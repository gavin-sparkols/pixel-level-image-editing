from __future__ import annotations

import json

from PIL import Image

from .core import process_pair


def _run(before, after, mask, mode, threshold, min_area_ratio, dilate):
    if before is None or after is None:
        raise ValueError("Upload both Before and After images")
    mask_path = mask if isinstance(mask, str) else None
    output = process_pair(
        Image.fromarray(before).convert("RGB"),
        Image.fromarray(after).convert("RGB"),
        mask_path=mask_path,
        mode=mode,
        threshold=float(threshold),
        min_area_ratio=float(min_area_ratio),
        dilate=int(dilate),
    )
    return output.result, output.mask, output.heatmap_before, output.heatmap_after, json.dumps(output.metrics, indent=2)


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("Install the UI dependencies with: pip install -e '.[app]'") from exc

    with gr.Blocks(title="Pixel-Level Image Editing") as demo:
        gr.Markdown(
            "# Pixel-Level Image Editing\n"
            "Upload a before/after pair. Untouched pixels are restored exactly from the Before image."
        )
        with gr.Row():
            before = gr.Image(label="Before", type="numpy")
            after = gr.Image(label="After", type="numpy")
            mask = gr.File(label="Optional binary mask (SAM 3.1 compatible)", type="filepath")
        with gr.Row():
            mode = gr.Radio(["paste", "recolor"], value="paste", label="Edit mode")
            threshold = gr.Slider(1, 100, value=24, step=1, label="Automatic mask threshold")
            min_area_ratio = gr.Slider(0, 0.02, value=0.0005, step=0.0001, label="Minimum component ratio")
            dilate = gr.Slider(0, 30, value=5, step=1, label="Mask expansion")
        run = gr.Button("Process", variant="primary")
        with gr.Row():
            result = gr.Image(label="Pixel-preserved result")
            output_mask = gr.Image(label="Edit mask")
        with gr.Row():
            heatmap_before = gr.Image(label="Before cleanup: pixel delta")
            heatmap_after = gr.Image(label="After cleanup: pixel delta")
        report = gr.Code(label="Validation report", language="json")
        run.click(
            _run,
            inputs=[before, after, mask, mode, threshold, min_area_ratio, dilate],
            outputs=[result, output_mask, heatmap_before, heatmap_after, report],
        )
    demo.launch(server_name=host, server_port=port, share=share)

