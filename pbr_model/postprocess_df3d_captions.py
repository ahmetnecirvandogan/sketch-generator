import os
import json
import time
import torch
from PIL import Image
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def postprocess_df3d_captions(batch_size=1):
    """
    Stage 3: Post-processing VLM captioning.
    Walks through the dataset/df3d directory and generates per-render captions 
    using Qwen2-VL-7B-Instruct.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen2-VL-7B-Instruct on {device}...")
    
    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    dataset_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "df3d")
    
    # Walk through the dataset
    samples_to_process = []
    for root, dirs, files in os.walk(dataset_root):
        if "metadata.json" in files and "render.png" in files:
            meta_path = os.path.join(root, "metadata.json")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # Idempotency check: skip if already captioned by VLM
            if "vlm_caption_at" not in meta:
                samples_to_process.append(root)

    print(f"Found {len(samples_to_process)} samples needing VLM captioning.")
    
    count = 0
    for sample_dir in samples_to_process:
        render_path = os.path.join(sample_dir, "render.png")
        meta_path = os.path.join(sample_dir, "metadata.json")
        prompt_path = os.path.join(sample_dir, "prompt.txt")
        
        try:
            # Prepare the prompt for Qwen2-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": render_path},
                        {"type": "text", "text": "Describe this 3D garment in detail. Focus ONLY on the garment's shape, silhouette, fabric texture, and specific patterns or graphics. Do NOT mention lighting, shadows, or highlights. Be concise but descriptive for a text-to-image model."},
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # Inference
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Update Metadata
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            old_text = meta.get("text", "")
            meta["vlm_caption"] = output_text
            meta["vlm_caption_at"] = datetime.now().isoformat()
            meta["text"] = f"{output_text}. {old_text}" # Prepend the rich VLM caption
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=4)
                
            # Update prompt.txt
            with open(prompt_path, 'w') as f:
                f.write(output_text)

            count += 1
            if count % 10 == 0:
                print(f"  Processed {count}/{len(samples_to_process)} samples... (Last: {output_text[:50]}...)")
                
        except Exception as e:
            print(f"Error processing {sample_dir}: {e}")

    print(f"\nFinished! Processed {count} samples.")

if __name__ == "__main__":
    postprocess_df3d_captions()
