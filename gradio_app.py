import gradio as gr
from API import generate_viton_output

def viton_interface(person_image, cloth_image, output_dir, height, width, base_model_path,
                     resume_path, mixed_precision, num_inference_steps, guidance_scale, seed,
                     repaint, concat_eval_results, cloth_type="upper"):
    output = generate_viton_output(
        person_image=person_image,
        cloth_image=cloth_image,
        output_dir=output_dir,
        height=height,
        width=width,
        base_model_path=base_model_path,
        resume_path=resume_path,
        mixed_precision=mixed_precision,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        repaint=repaint,
        concat_eval_results=concat_eval_results,
        cloth_type=cloth_type
    )
    return output


def main():
    with gr.Blocks() as app:
        gr.Markdown("# 2D Virtual Try-On (VITON)\nUpload a person image and a clothing image to generate the virtual try-on output.")

        with gr.Row():
            person_image = gr.Image(type="filepath", label="Person Image")
            cloth_image = gr.Image(type="filepath", label="Cloth Image")

        with gr.Row():
            output_dir = gr.Textbox(value="output", label="Output Directory")
            height = gr.Number(value=512, label="Height")
            width = gr.Number(value=384, label="Width")

        with gr.Row():
            base_model_path = gr.Textbox(value="runwayml/stable-diffusion-inpainting", label="Base Model Path")
            resume_path = gr.Textbox(value="zhengchong/CatVTON", label="Resume Path")
            mixed_precision = gr.Dropdown(choices=["fp16", "bf16", "no"], value="fp16", label="Mixed Precision")

        with gr.Row():
            num_inference_steps = gr.Slider(minimum=0, maximum=150, value=100, step=1, label="Number of Inference Steps")
            guidance_scale = gr.Slider(minimum=0.0, maximum=10.0, value=2.5, step=0.1, label="Guidance Scale")
            seed = gr.Number(value=555, label="Seed")

        with gr.Row():
            repaint = gr.Checkbox(value=False, label="Repaint")
            concat_eval_results = gr.Checkbox(value=True, label="Concatenate Evaluation Results")
        
        with gr.Row():
            cloth_type = gr.Dropdown(choices=["upper", "lower", "overall", "inner", "outer"], label="Cloth Type", value="Upper")

        output_image = gr.Image(label="VITON Output")
        generate_button = gr.Button("Generate")

        generate_button.click(
            fn=viton_interface,
            inputs=[
                person_image, cloth_image, output_dir, height, width, base_model_path,
                resume_path, mixed_precision, num_inference_steps, guidance_scale, seed,
                repaint, concat_eval_results, cloth_type
            ],
            outputs=[output_image]
        )

    return app

if __name__ == "__main__":
    app = main()
    app.launch(share=True)
