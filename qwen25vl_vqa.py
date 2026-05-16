import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


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

    text = text.strip().upper()
    valid_letters = LETTERS[:num_choices]

    # 优先判断开头
    for letter in valid_letters:
        if text.startswith(letter):
            return letter, valid_letters.index(letter)

    # 再判断文本中是否出现单独选项
    for letter in valid_letters:
        if f" {letter}" in text or f"{letter}." in text or f"{letter})" in text:
            return letter, valid_letters.index(letter)

    return None, -1


def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def run_qwen25vl(
    jsonl_path,
    output_path,
    model_id="./weights/Qwen2.5-VL-7B-Instruct",
    limit=None
):
    print(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Loading model: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
	local_files_only=True
    )

    data = load_jsonl(jsonl_path)

    if limit is not None:
        data = data[:limit]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for item in tqdm(data, desc="Running inference"):
        image_path = item["image"]
        question = item["question"]
        choices = item["choices"]

        prompt = build_prompt(question, choices)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

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
                "model_output": "",
                "pred_letter": None,
                "pred_answer_index": -1,
                "pred_answer": None,
                "correct": False,
                "error": str(e)
            }

        results.append(result)

        # 实时写入，防止中途出错丢失结果
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
        default="results_qwen25vl_7b.jsonl",
        help="Output result jsonl file"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model id"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N samples for debugging"
    )

    args = parser.parse_args()

    run_qwen25vl(
        jsonl_path=args.jsonl,
        output_path=args.output,
        model_id=args.model,
        limit=args.limit
    )
