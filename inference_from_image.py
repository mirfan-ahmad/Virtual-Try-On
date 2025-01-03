import os
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image, ImageFilter
from model.pipeline import CatVTONPipeline
from generate_mask import generate_mask
from arguments import parse_args


def preprocess_images(person_image_path, cloth_image_path, height, width):
    vae_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )

    assert os.path.exists(person_image_path), f"Person image {person_image_path} does not exist."
    assert os.path.exists(cloth_image_path), f"Cloth image {cloth_image_path} does not exist."

    person_image = Image.open(person_image_path)
    cloth_image = Image.open(cloth_image_path)
    mask_image = Image.open(generate_mask(person_image_path, cloth_image_path, cloth_type="upper"))

    person_processed = vae_processor.preprocess(person_image, height, width)[0]
    cloth_processed = vae_processor.preprocess(cloth_image, height, width)[0]
    mask_processed = mask_processor.preprocess(mask_image, height, width)[0]

    return person_processed, cloth_processed, mask_processed, person_image, cloth_image


def repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 50
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result


def to_pil_image(images):
    """Convert tensor or numpy array images to PIL format."""
    if isinstance(images, list):  # Check if input is a list
        return [Image.fromarray(image.astype("uint8")) if isinstance(image, np.ndarray) else image for image in images]
    else:
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            return [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            return [Image.fromarray(image) for image in images]


@torch.no_grad()
def generate_viton_output(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize pipeline
    pipeline = CatVTONPipeline(
        attn_ckpt_version="vitonhd",
        attn_ckpt=args.resume_path,
        base_ckpt=args.base_model_path,
        weight_dtype={
            "no": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.mixed_precision],
        device="cuda",
        skip_safety_check=True,
    )

    # Preprocess images
    person_processed, cloth_processed, mask_processed, person_image, cloth_image = preprocess_images(
        args.person_image, args.cloth_image, args.height, args.width
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    results = pipeline(
        person_processed.unsqueeze(0),
        cloth_processed.unsqueeze(0),
        mask_processed.unsqueeze(0),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    )

    # Convert images to PIL format for visualization and further processing
    person_images = [person_image]
    cloth_images = [cloth_image]

    # Process results
    for i, result in enumerate(results):
        person_name = os.path.basename(args.person_image).split('.')[0]  # Get the person's name from the image file
        output_path = os.path.join(args.output_dir, f"{person_name}_result.png")

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        result_image = result if isinstance(result, Image.Image) else to_pil_image([result])[0]

        if args.repaint:
            person_image = person_images[0].resize(result_image.size, Image.LANCZOS)
            mask = Image.open("agnostic_mask.png").resize(result_image.size, Image.NEAREST)
            result_image = repaint(person_image, mask, result_image)

        if args.concat_eval_results:
            w, h = result_image.size
            resized_person_image = person_images[0].resize((w, h), Image.LANCZOS)
            resized_cloth_image = cloth_images[0].resize((w, h), Image.LANCZOS)

            concated_result = Image.new("RGB", (w * 3, h))
            concated_result.paste(resized_person_image, (0, 0))
            concated_result.paste(resized_cloth_image, (w, 0))
            concated_result.paste(result_image, (w * 2, 0))
            result_image = concated_result

        result_image.save(output_path)

    return output_path


if __name__ == "__main__":
    args = parse_args()
    try:
        print(generate_viton_output(args))
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists("agnostic_mask.png"):
            os.remove("agnostic_mask.png")
