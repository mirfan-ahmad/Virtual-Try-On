
# Virtual Try-On Application

A cutting-edge virtual try-on application for trying clothing on images with seamless results. This repository provides scripts for setting up the environment, running the Gradio application, and performing local inference on single images or entire folders.

---

## Table of Contents

- [Virtual Try-On Application](#virtual-try-on-application)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Setup Instructions](#setup-instructions)
  - [Using the Gradio Application](#using-the-gradio-application)
  - [Local Inference](#local-inference)
    - [Infer from a Single Image](#infer-from-a-single-image)
    - [Infer on a Whole Folder](#infer-on-a-whole-folder)
  - [Command-Line Arguments](#command-line-arguments)
    - [Required Arguments](#required-arguments)
    - [Optional Arguments](#optional-arguments)

---

## Features

- Try clothing on images with high-quality results.
- Flexible inference options:
  - Use the Gradio interface for an interactive experience.
  - Perform local inference for batch processing.
- Support for mixed precision and advanced settings for efficient computation.
- User-friendly commands for seamless integration.

---

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Set Up Virtual Environment**

   Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

   Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-Trained Checkpoints**

   Run the following script to download necessary model checkpoints:
   ```bash
   python download_ckpts.py
   ```

4. **Launch the Gradio Application**

   Start the Gradio-based user interface:
   ```bash
   python gradio_app.py
   ```

---

## Using the Gradio Application

Once the Gradio application is running, open the provided URL in your browser to access the virtual try-on interface. Here, you can upload images and customize settings to try clothing interactively.

---

## Local Inference

For advanced users, local inference scripts are provided for processing single images or entire folders.

### Infer from a Single Image

Run the `inference_from_image.py` script with the following command:

```bash
python inference_from_image.py --person_image <path> --cloth_image <path> --output_dir <path> --cloth_type [upper | lower | overall] --mixed_precision [no | fp16 | bf16] --allow_tf32 --concat_eval_results --repaint --width <512> --height <368> --num_inference_steps <50> --seed <555>
```

**Example Command**:
```bash
python inference_from_image.py --person_image ./data/person.jpg --cloth_image ./data/cloth.jpg --output_dir ./output --cloth_type upper --mixed_precision fp16 --allow_tf32 --width 512 --height 368
```

### Infer on a Whole Folder

Run the `inference_from_dir.py` script to process all images in a folder:

```bash
python inference_from_dir.py --person_image <path/to/folder> --cloth_image <path/to/folder> --output_dir <path> --cloth_type [upper | lower | overall] --dataloader_num_workers <num_of_cores> --batch_size <8> --guidance_scale <2.5> --mixed_precision [no | fp16 | bf16] --allow_tf32 --concat_eval_results --repaint --width <512> --height <368> --num_inference_steps <50> --seed <555>
```

**Example Command**:
```bash
python inference_from_dir.py --person_image ./data/persons/ --cloth_image ./data/cloths/ --output_dir ./output --cloth_type overall --dataloader_num_workers 4 --batch_size 8 --guidance_scale 2.5 --mixed_precision fp16 --allow_tf32 --width 512 --height 368
```

---

## Command-Line Arguments

### Required Arguments

| Argument         | Description                         |
|------------------|-------------------------------------|
| `--person_image` | Path to the person image or folder. |
| `--cloth_image`  | Path to the clothing image or folder. |
| `--output_dir`   | Directory where output images will be saved. |

### Optional Arguments

| Argument                 | Description                                                  |
|--------------------------|--------------------------------------------------------------|
| `--cloth_type`           | Type of clothing: `upper`, `lower`, or `overall`.           |
| `--mixed_precision`      | Mixed precision mode: `no`, `fp16`, or `bf16`.              |
| `--allow_tf32`           | Enables TensorFloat-32 computation.                         |
| `--concat_eval_results`  | Concatenate evaluation results into a single output.         |
| `--repaint`              | Enable repainting for refined results.                      |
| `--width`                | Image width for processing. Default: 512.                   |
| `--height`               | Image height for processing. Default: 368.                  |
| `--num_inference_steps`  | Number of inference steps. Default: 50.                     |
| `--seed`                 | Seed for reproducibility. Default: 555.                     |
| `--dataloader_num_workers` | Number of workers for data loading (folder inference only). |
| `--batch_size`           | Batch size for folder inference. Default: 8.                |
| `--guidance_scale`       | Guidance scale for folder inference. Default: 2.5.          |


Enjoy using the Virtual Try-On application! 🎉
