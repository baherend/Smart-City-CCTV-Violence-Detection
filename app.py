import os

import gradio as gr

from SCVD_Professional_GUI import CSS, demo


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    )
