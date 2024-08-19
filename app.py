import os
import sys
import subprocess
import gradio as gr

repo_dir = os.path.dirname(__file__)
sys.path.extend((repo_dir, os.getcwd()))
from src.models.utils.model import load_ensemble_model
from src.models.utils.logging import get_logger
from src.models.taskers.inferencer import infer

logger = get_logger("Application", is_stream=True)

cmd = ['bash', 'src/prepare.sh', '--platform', 'gradio']
subprocess.run(cmd, shell=False, capture_output=False, stdout=None)
logger.info("Environment is set up")

model, cfg, saved_cfg, llm_tokenizer = load_ensemble_model(
    os.path.join(repo_dir, 'src/models/vavsp_llm.yaml')
)
logger.info("Loaded model")


def predict(
        video_path,
        time_interval
):
    progress = gr.Progress()
    if video_path is None:
        logger.warning('Upload/record a video. The pipeline is crashed')
        exit(1)
    output_file = "results/output.mp4"
    if os.path.isfile(output_file):
        os.remove(path=output_file)
    try:
        output_file = infer(
            video_path=video_path,
            progress=progress,
            time_interval=time_interval,
            model=model,
            cfg=cfg,
            saved_cfg=saved_cfg,
            llm_tokenizer=llm_tokenizer,
        )
    except RuntimeError:
        logger.critical('Runtime error caught while inferencing.')
        return video_path

    if not os.path.isfile(output_file):
        return video_path

    return output_file


if __name__ == "__main__":
    app = gr.Interface(
        fn=predict,
        inputs=[
            gr.Video(
                sources=['upload', 'webcam'],
                format='mp4'
            ),
            gr.Slider(
                minimum=1, maximum=200,
                value=3, step=1,
                label='Second',
            ),
        ],
        outputs=gr.Video(),
        title="Demo project",
        description="Vietnamese Automatic Speech Recognition Utilizing Audio and Visual Data"
    )

    app.launch(share=True)