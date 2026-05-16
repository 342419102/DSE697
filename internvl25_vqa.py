import json
import argparse
import re
from pathlib import Path
from collections import Counter

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel


LETTERS = ["A", "B", "C", "D"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image,
    min_num=1,
    max_num=6,
    image_size=448,
    use_thumbnail=True
):
    """
    InternVL dynamic high-resolution preprocessing.
    max_num controls the maximum number of image tiles.
    For 24GB GPU, max_num=4 or 6 is safer.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )

    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))

    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )

        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image_for_internvl(
    image_path,
    input_size=448,
    max_num=6,
    dtype=torch.bfloat16
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=input_size)

    images = dynamic_preprocess(
        image,
        image_size=input_size,
        max_num=max_num,
        use_thumbnail=True
    )

    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)

    return pixel_values.to(dtype)


def load_internvl_model(
    model_id,
    dtype=torch.bfloat16,
    local_files_only=True,
    load_8bit=False,
    use_flash_attn=False
):
    print(f"Loading tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=local_files_only
    )

    print(f"Loading model: {model_id}")

    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "local_files_only": local_files_only
    }

    # InternVL 官方 quick start 支持 load_in_8bit；
    # 24GB 显存如果 full precision OOM，可以尝试 --load_8bit。
    if load_8bit:
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = "auto"
    else:
        # 单卡直接 cuda；如果 OOM，可改用 --load_8bit
        pass

    try:
        kwargs["use_flash_attn"] = use_flash_attn
        model = AutoModel.from_pretrained(model_id, **kwargs).eval()
    except TypeError:
        # 部分环境 / remote code 不接受 use_flash_attn 时，自动回退
        kwargs.pop("use_flash_attn", None)
        model = AutoModel.from_pretrained(model_id, **kwargs).eval()

    if not load_8bit:
        model = model.cuda()

    return tokenizer, model


def run_internvl25(
    jsonl_path,
    output_path,
    model_id="./weights/InternVL2_5-8B",
    limit=None,
    max_new_tokens=4,
    local_files_only=True,
    overwrite=True,
    dtype_name="bf16",
    max_num=6,
    load_8bit=False,
    use_flash_attn=False
):
    if dtype_name == "bf16":
        dtype = torch.bfloat16
    elif dtype_name == "fp16":
        dtype = torch.float16
    else:
        raise ValueError("dtype_name must be 'bf16' or 'fp16'.")

    tokenizer, model = load_internvl_model(
        model_id=model_id,
        dtype=dtype,
        local_files_only=local_files_only,
        load_8bit=load_8bit,
        use_flash_attn=use_flash_attn
    )

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

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False
    }

    for item in tqdm(data, desc="Running InternVL2.5-8B inference"):
        image_path = item["image"]
        question = item["question"]
        choices = item["choices"]

        try:
            pixel_values = load_image_for_internvl(
                image_path=image_path,
                input_size=448,
                max_num=max_num,
                dtype=dtype
            ).cuda()

            prompt = "<image>\n" + build_prompt(question, choices)

            with torch.inference_mode():
                output_text = model.chat(
                    tokenizer,
                    pixel_values,
                    prompt,
                    generation_config
                )

            output_text = str(output_text).strip()

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

        # 避免长时间循环中显存碎片持续累积
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
        default="results_internvl25_8b.jsonl",
        help="Output result jsonl file"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="./weights/InternVL2_5-8B",
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
        "--max_num",
        type=int,
        default=6,
        help="Maximum number of image tiles for dynamic preprocessing. Use 4 or 6 for 24GB GPU."
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Inference dtype"
    )

    parser.add_argument(
        "--load_8bit",
        action="store_true",
        help="Use 8-bit loading if full precision causes CUDA OOM"
    )

    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use flash attention if installed and supported"
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

    run_internvl25(
        jsonl_path=args.jsonl,
        output_path=args.output,
        model_id=args.model,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        local_files_only=not args.online,
        overwrite=not args.no_overwrite,
        dtype_name=args.dtype,
        max_num=args.max_num,
        load_8bit=args.load_8bit,
        use_flash_attn=args.use_flash_attn
    )