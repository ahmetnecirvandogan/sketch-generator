import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def caption_df3d_textures():
    """
    Offline captioning pass for DF3D textures.
    Uses BLIP to describe the contents of every texture.png and saves to a JSON registry.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BLIP model on {device}...")
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", 
        use_safetensors=True
    ).to(device)

    # Use the project-relative path (works via the symlink we created)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df3d_root = os.path.join(base_dir, "meshes", "df3d", "all")
    output_json = os.path.join(base_dir, "meshes", "df3d", "captions.json")
    
    captions = {}
    
    # Load existing if any to resume
    if os.path.exists(output_json):
        with open(output_json) as f:
            captions = json.load(f)

    folders = sorted([f for f in os.listdir(df3d_root) if os.path.isdir(os.path.join(df3d_root, f))])
    
    print(f"Found {len(folders)} garments. Starting captioning...")
    
    count = 0
    try:
        for folder in folders:
            if folder in captions:
                continue
                
            tex_path = os.path.join(df3d_root, folder, f"{folder}_tex.png")
            if os.path.exists(tex_path):
                raw_image = Image.open(tex_path).convert('RGB')
                
                # Generate caption
                inputs = processor(raw_image, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=40)
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                captions[folder] = caption
                count += 1
                if count % 10 == 0:
                    print(f"  Processed {count} textures... (Last: {folder}: {caption})")
                    # Periodically save
                    with open(output_json, 'w') as f:
                        json.dump(captions, f, indent=4)
                        
    except KeyboardInterrupt:
        print("\nStopping and saving progress...")
    
    with open(output_json, 'w') as f:
        json.dump(captions, f, indent=4)
        
    print(f"\nDone! Saved {len(captions)} captions to {output_json}")

if __name__ == "__main__":
    caption_df3d_textures()
