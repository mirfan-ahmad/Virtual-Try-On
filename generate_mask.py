from PIL import Image
from model.cloth_masker import AutoMasker
from diffusers.image_processor import VaeImageProcessor
import numpy as np
import os


def load_pkl():
    repo_path = "."
    global automasker
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device='cuda',
    )

def generate_mask(person_image_path, cloth_image_path, cloth_type="upper"):
    load_pkl()
    person_image = Image.open(person_image_path).convert("RGB")
    cloth_image = Image.open(cloth_image_path).convert("RGB")

    person_image = person_image.resize(cloth_image.size, Image.LANCZOS)
    result = automasker(person_image, cloth_type)
    mask = result['mask']

    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(mask.astype(np.uint8))

    mask = mask.resize((768, 1024), Image.LANCZOS)
    mask_rgb = mask.convert("RGB")
    mask_rgb.save("agnostic_mask.png")
    return "agnostic_mask.png"


if __name__ == "__main__":
    person_image_path = r"dataset\test\image\03921_00.jpg"
    cloth_image_path = r"dataset\test\cloth\08015_00.jpg"

    mask_image = generate_mask(person_image_path, cloth_image_path)
    
