import json
import argparse
from pathlib import Path

import pandas as pd


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return pd.DataFrame(records)


def safe_acc(df):
    if len(df) == 0:
        return 0.0
    return df["correct"].mean()


def summarize_group(df, group_cols):
    summary = (
        df.groupby(group_cols)
        .agg(
            total=("correct", "count"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean")
        )
        .reset_index()
    )

    summary["accuracy"] = summary["accuracy"].round(4)
    return summary


def add_task_group(df):
    """
    Convert detailed question_type into broader task groups:
    - total_count
    - behavior_presence
    - behavior_count
    """
    def map_group(qtype):
        if qtype == "total_count":
            return "total_count"
        elif qtype.endswith("_presence"):
            return "behavior_presence"
        elif qtype.endswith("_count"):
            return "behavior_count"
        else:
            return "other"

    df["task_group"] = df["question_type"].apply(map_group)
    return df


def add_gt_count_bin(df):
    """
    Bin gt_count for count-related difficulty analysis.
    """
    def bin_count(x):
        try:
            x = int(x)
        except Exception:
            return "unknown"

        if x == 0:
            return "0"
        elif x == 1:
            return "1"
        elif x == 2:
            return "2"
        elif x == 3:
            return "3"
        else:
            return ">=4"

    df["gt_count_bin"] = df["gt_count"].apply(bin_count)
    return df


def evaluate_results(result_jsonl, output_dir):
    result_jsonl = Path(result_jsonl)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(result_jsonl)

    if len(df) == 0:
        raise ValueError("The result file is empty.")

    # 防止重复运行推理脚本时，结果文件中出现重复 id
    if "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["id"], keep="last").copy()
        after = len(df)
        if before != after:
            print(f"Warning: duplicated ids detected. Kept last occurrence: {before} -> {after}")

    # 确保 correct 是 bool / 数值
    df["correct"] = df["correct"].astype(bool)

    # 添加任务大类
    df = add_task_group(df)

    # 添加 gt_count 分组
    df = add_gt_count_bin(df)

    # 1. Overall accuracy
    overall = pd.DataFrame([{
        "model_result_file": result_jsonl.name,
        "total": len(df),
        "correct": int(df["correct"].sum()),
        "accuracy": round(safe_acc(df), 4),
        "invalid_prediction": int((df["pred_answer_index"] == -1).sum()) if "pred_answer_index" in df.columns else None
    }])

    # 2. Accuracy by detailed question type
    by_question_type = summarize_group(
        df,
        ["question_type"]
    )

    # 3. Accuracy by broader task group
    by_task_group = summarize_group(
        df,
        ["task_group"]
    )

    # 4. Accuracy by behavior
    # total_count 的 behavior 是 all；这里保留 all，方便完整对比
    by_behavior = summarize_group(
        df,
        ["behavior"]
    )

    # 5. Accuracy by task group and behavior
    by_task_behavior = summarize_group(
        df,
        ["task_group", "behavior"]
    )

    # 6. Accuracy by question type and gt_count bin
    by_qtype_count_bin = summarize_group(
        df,
        ["question_type", "gt_count_bin"]
    )

    # 7. Accuracy for count questions only
    count_df = df[df["task_group"].isin(["total_count", "behavior_count"])].copy()

    by_count_task_gtbin = summarize_group(
        count_df,
        ["task_group", "gt_count_bin"]
    )

    # 8. Accuracy for presence questions only
    presence_df = df[df["task_group"] == "behavior_presence"].copy()

    by_presence_behavior = summarize_group(
        presence_df,
        ["behavior"]
    )

    # 9. Yes / No presence question confusion-style summary
    if len(presence_df) > 0:
        presence_confusion = (
            presence_df.groupby(["behavior", "gt_answer", "pred_answer"])
            .agg(total=("correct", "count"))
            .reset_index()
        )
    else:
        presence_confusion = pd.DataFrame()

    # 10. Prediction option distribution
    if "pred_letter" in df.columns:
        pred_distribution = (
            df.groupby(["question_type", "pred_letter"])
            .agg(total=("correct", "count"))
            .reset_index()
        )
    else:
        pred_distribution = pd.DataFrame()

    # 11. Error cases
    error_cases = df[df["correct"] == False].copy()

    # Save CSV files
    overall.to_csv(output_dir / "overall_accuracy.csv", index=False, encoding="utf-8-sig")
    by_question_type.to_csv(output_dir / "accuracy_by_question_type.csv", index=False, encoding="utf-8-sig")
    by_task_group.to_csv(output_dir / "accuracy_by_task_group.csv", index=False, encoding="utf-8-sig")
    by_behavior.to_csv(output_dir / "accuracy_by_behavior.csv", index=False, encoding="utf-8-sig")
    by_task_behavior.to_csv(output_dir / "accuracy_by_task_behavior.csv", index=False, encoding="utf-8-sig")
    by_qtype_count_bin.to_csv(output_dir / "accuracy_by_question_type_and_gt_count_bin.csv", index=False, encoding="utf-8-sig")
    by_count_task_gtbin.to_csv(output_dir / "accuracy_by_count_task_and_gt_count_bin.csv", index=False, encoding="utf-8-sig")
    by_presence_behavior.to_csv(output_dir / "accuracy_by_presence_behavior.csv", index=False, encoding="utf-8-sig")
    presence_confusion.to_csv(output_dir / "presence_confusion_summary.csv", index=False, encoding="utf-8-sig")
    pred_distribution.to_csv(output_dir / "prediction_distribution.csv", index=False, encoding="utf-8-sig")
    error_cases.to_csv(output_dir / "error_cases.csv", index=False, encoding="utf-8-sig")

    # Print main results
    print("\n========== Overall ==========")
    print(overall.to_string(index=False))

    print("\n========== Accuracy by task group ==========")
    print(by_task_group.to_string(index=False))

    print("\n========== Accuracy by question type ==========")
    print(by_question_type.to_string(index=False))

    print("\n========== Accuracy by behavior ==========")
    print(by_behavior.to_string(index=False))

    print("\n========== Accuracy by task group and behavior ==========")
    print(by_task_behavior.to_string(index=False))

    print("\n========== Count-task accuracy by gt_count bin ==========")
    print(by_count_task_gtbin.to_string(index=False))

    print(f"\nSaved all evaluation files to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result",
        type=str,
        required=True,
        help="Path to model result jsonl file, e.g., results_qwen25vl_7b.jsonl"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_qwen25vl_7b",
        help="Directory to save evaluation CSV files"
    )

    args = parser.parse_args()

    evaluate_results(
        result_jsonl=args.result,
        output_dir=args.output_dir
    )
