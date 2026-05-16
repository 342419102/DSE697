import json
import argparse
import re
from pathlib import Path
from collections import Counter

import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


LETTERS = ["A", "B", "C", "D"]


def build_prompt(question, choices):
    prompt = (
        "You are given a top-view image of broiler chickens.\n"
        "Only consider birds inside the central pen.\n\n"
        f"Question: {question}\n\n"
        "Choices:\n"
    )

    for i, choice in enumerate(choices):
        prompt += f"{LETTERS[i]}. {choice}\n"

    valid_letters = "/".join(LETTERS[:len(choices)])
    prompt += f"\nPlease answer with only one letter: {valid_letters}."

    return prompt


def parse_answer(text, num_choices):
    if text is None:
        return None, -1

    text = str(text).strip().upper()
    valid_letters = LETTERS[:num_choices]

    for letter in valid_letters:
        if text.startswith(letter):
            return letter, valid_letters.index(letter)

    pattern = r"\b(" + "|".join(valid_letters) + r")\b"
    match = re.search(pattern, text)
    if match:
        pred_letter = match.group(1)
        return pred_letter, valid_letters.index(pred_letter)

    for letter in valid_letters:
        if f"{letter}." in text or f"{letter})" in text:
            return letter, valid_letters.index(letter)

    return None, -1


def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def resolve_image_path(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return str(image_path.resolve())


def run_qwen3vl(
    jsonl_path,
    output_path,
    model_id="./weights/Qwen3-VL-8B-Instruct",
    limit=None,
    max_new_tokens=4,
    local_files_only=True,
    overwrite=True
):
    print(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only
    )

    print(f"Loading model: {model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
        local_files_only=local_files_only
    )

    model.eval()

    data = load_jsonl(jsonl_path)
    if limit is not None:
        data = data[:limit]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and overwrite:
        output_path.unlink()

    results = []
    error_counter = Counter()
    printed_error_count = 0

    for item in tqdm(data, desc="Running Qwen3-VL inference"):
        image_path = item["image"]
        question = item["question"]
        choices = item["choices"]

        try:
            image_path_abs = resolve_image_path(image_path)
            prompt = build_prompt(question, choices)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path_abs},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            inputs = inputs.to(model.device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            pred_letter, pred_index = parse_answer(
                output_text,
                num_choices=len(choices)
            )

            correct = pred_index == int(item["answer_index"])

            result = {
                "id": item["id"],
                "image": image_path,
                "image_name": item.get("image_name", ""),
                "date": item.get("date", ""),
                "question_type": item["question_type"],
                "behavior": item.get("behavior", ""),
                "question": question,
                "choices": choices,
                "gt_answer": item["answer"],
                "gt_answer_index": int(item["answer_index"]),
                "gt_count": item.get("gt_count", None),
                "model_output": output_text,
                "pred_letter": pred_letter,
                "pred_answer_index": pred_index,
                "pred_answer": choices[pred_index] if pred_index >= 0 else None,
                "correct": correct
            }

        except Exception as e:
            err = str(e)
            error_counter[err] += 1

            if printed_error_count < 5:
                print("\n[ERROR SAMPLE]")
                print("id:", item.get("id", ""))
                print("image:", image_path)
                print("error:", err)
                printed_error_count += 1

            result = {
                "id": item.get("id", ""),
                "image": image_path,
                "image_name": item.get("image_name", ""),
                "date": item.get("date", ""),
                "question_type": item.get("question_type", ""),
                "behavior": item.get("behavior", ""),
                "question": question,
                "choices": choices,
                "gt_answer": item.get("answer", ""),
                "gt_answer_index": int(item.get("answer_index", -1)),
                "gt_count": item.get("gt_count", None),
                "model_output": "",
                "pred_letter": None,
                "pred_answer_index": -1,
                "pred_answer": None,
                "correct": False,
                "error": err
            }

        results.append(result)

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    total = len(results)
    correct_num = sum(r["correct"] for r in results)
    acc = correct_num / total if total > 0 else 0

    print("\nFinished.")
    print(f"Total questions: {total}")
    print(f"Correct: {correct_num}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Saved to: {output_path}")

    if len(error_counter) > 0:
        print("\nError summary:")
        for err, count in error_counter.most_common(5):
            print(f"[{count}] {err[:500]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to broiler_vlm_vqa_mcq.jsonl"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results_qwen3vl_8b.jsonl",
        help="Output result jsonl file"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="./weights/Qwen3-VL-8B-Instruct",
        help="Local model path or Hugging Face model id"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N samples for debugging"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4,
        help="Maximum number of generated tokens"
    )

    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow loading files from Hugging Face online instead of local-only"
    )

    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Do not delete existing output file before running"
    )

    args = parser.parse_args()

    run_qwen3vl(
        jsonl_path=args.jsonl,
        output_path=args.output,
        model_id=args.model,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        local_files_only=not args.online,
        overwrite=not args.no_overwrite
    )