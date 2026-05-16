import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 修改为你自己的任意一张测试图片
image_path = "/home/jiewu/pycharm-2025.1/workspace/DSE697/images/D02_20260215171317_000002.jpg"

prompt = (
    "You are given a top-view image of broiler chickens.\n"
    "Only consider birds inside the central pen.\n\n"
    "Question: How many birds are in the central pen?\n\n"
    "Choices:\n"
    "A. 10\n"
    "B. 11\n"
    "C. 12\n"
    "D. 13\n\n"
    "Please answer with only one letter: A, B, C, or D."
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }
]

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

print("Generating...")

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
)

print("Model output:", output_text[0])
