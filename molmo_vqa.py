import json
import argparse
import re
from pathlib import Path
from collections import Counter

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

# Compatibility patch for Molmo + Transformers 5.x
# Some remote-code models such as MolmoForCausalLM may not define
# all_tied_weights_keys, while Transformers 5.x expects this attribute
# during model loading.
try:
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        def _get_all_tied_weights_keys(self):
            keys = getattr(self, "_tied_weights_keys", None)

            if keys is None:
                return {}

            if isinstance(keys, dict):
                return keys

            if isinstance(keys, (list, tuple, set)):
                return {k: k for k in keys}

            return {}

        PreTrainedModel.all_tied_weights_keys = property(_get_all_tied_weights_keys)

except Exception as e:
    print(f"Warning: failed to apply Molmo compatibility patch: {e}")

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


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

    # 1. 优先判断开头是否为 A/B/C/D
    for letter in valid_letters:
        if text.startswith(letter):
            return letter, valid_letters.index(letter)

    # 2. 解析 "Answer: C" / "The answer is C"
    pattern = r"\b(" + "|".join(valid_letters) + r")\b"
    match = re.search(pattern, text)
    if match:
        pred_letter = match.group(1)
        return pred_letter, valid_letters.index(pred_letter)

    # 3. 解析 C. / C)
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


def load_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return Image.open(image_path).convert("RGB")


def get_torch_dtype(dtype_name):
    if dtype_name == "auto":
        return "auto"
    elif dtype_name == "bf16":
        return torch.bfloat16
    elif dtype_name == "fp16":
        return torch.float16
    elif dtype_name == "fp32":
        return torch.float32
    else:
        raise ValueError("dtype must be one of: auto, bf16, fp16, fp32")


def move_batch_to_device(inputs, device, dtype=torch.bfloat16):
    """
    Move Molmo processor outputs to GPU and add batch dimension.
    Some fields may be None; these must be removed before model.generate_from_batch().
    """
    batch = {}

    for k, v in inputs.items():
        # 关键：过滤 None，否则 Molmo 内部可能调用 None.size() 报错
        if v is None:
            continue

        if torch.is_tensor(v):
            v = v.to(device)

            # 只转换浮点 tensor，避免 input_ids 这类整数 tensor 被错误转换
            if torch.is_floating_point(v):
                v = v.to(dtype=dtype)

            batch[k] = v.unsqueeze(0)
        else:
            batch[k] = v

    return batch


def get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_molmo_model(
    model_id,
    local_files_only=True,
    dtype_name="bf16",
    load_4bit=False
):
    dtype = get_torch_dtype(dtype_name)

    print(f"Loading processor: {model_id}")

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only
    )

    print(f"Loading model: {model_id}")

    if load_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError(
                "BitsAndBytesConfig is not available. Please install bitsandbytes first."
            )

        print("Using 4-bit quantized loading.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        # 注意：这里不要用 device_map='auto'，避免 Molmo remote code 与 accelerate 自动分配逻辑冲突
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map={"": 0},
            local_files_only=local_files_only
        )

    else:
        print(f"Using dtype: {dtype_name}")

        # 关键修正：不要使用 device_map='auto'
        # 否则可能触发 MolmoForCausalLM 没有 all_tied_weights_keys 的报错
        if dtype == "auto":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=local_files_only
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                local_files_only=local_files_only
            )

        if torch.cuda.is_available():
            model = model.to("cuda")

    model.eval()

    return processor, model


def run_molmo(
    jsonl_path,
    output_path,
    model_id="./weights/Molmo-7B-D-0924",
    limit=None,
    max_new_tokens=4,
    local_files_only=True,
    overwrite=True,
    dtype_name="bf16",
    load_4bit=False
):
    processor, model = load_molmo_model(
        model_id=model_id,
        local_files_only=local_files_only,
        dtype_name=dtype_name,
        load_4bit=load_4bit
    )

    device = get_model_device(model)
    print(f"Model device: {device}")

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

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        stop_strings="<|endoftext|>"
    )

    for item in tqdm(data, desc="Running Molmo-7B-D inference"):
        image_path = item["image"]
        question = item["question"]
        choices = item["choices"]

        try:
            image = load_image(image_path)
            prompt = build_prompt(question, choices)

            # Molmo 官方方式：processor.process(images=[image], text=prompt)
            inputs = processor.process(
                images=[image],
                text=prompt
            )

            if dtype_name == "bf16":
                input_dtype = torch.bfloat16
            elif dtype_name == "fp16":
                input_dtype = torch.float16
            else:
                input_dtype = torch.float32

            inputs = move_batch_to_device(
                inputs,
                device=device,
                dtype=input_dtype
            )

            with torch.inference_mode():
                output = model.generate_from_batch(
                    inputs,
                    generation_config,
                    tokenizer=processor.tokenizer
                )

            input_len = inputs["input_ids"].size(1)
            generated_tokens = output[0, input_len:]

            output_text = processor.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

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

        # 实时写入，防止中途崩溃丢失结果
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        default="results_molmo_7b.jsonl",
        help="Output result jsonl file"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="./weights/Molmo-7B-D-0924",
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
        "--dtype",
        type=str,
        default="bf16",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Inference dtype"
    )

    parser.add_argument(
        "--load_4bit",
        action="store_true",
        help="Load model in 4-bit mode to reduce GPU memory usage"
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

    run_molmo(
        jsonl_path=args.jsonl,
        output_path=args.output,
        model_id=args.model,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        local_files_only=not args.online,
        overwrite=not args.no_overwrite,
        dtype_name=args.dtype,
        load_4bit=args.load_4bit
    )