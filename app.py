import os
import sys
import subprocess
import gradio as gr

repo_dir = os.path.dirname(__file__)
sys.path.extend((repo_dir, os.getcwd()))
from src.models.utils.model import load_ensemble_model, load_feature_extractor
from src.models.utils import vsp_llm as custom_utils
from src.models.utils.logging import get_logger
from src.models.taskers.inferencer import infer

logger = get_logger("Application", is_stream=True)

def setup_environment():
    cmd = ['bash', 'scripts/prepare.sh', '--platform', 'gradio']
    subprocess.run(cmd, shell=False, capture_output=False, stdout=None)

setup_environment()
logger.info("Environment is set up")

extractor = load_feature_extractor(
        os.path.join(repo_dir, 'src/models/checkpoints/large_vox_iter5.pt'),
        12, custom_utils=custom_utils
)
logger.info("Loaded feature extractor")

model, cfg, saved_cfg, llm_tokenizer = load_ensemble_model(
    os.path.join(repo_dir, 'vavsp_llm.yaml')
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
            extractor=extractor
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
                minimum=1, maximum=999,
                value=3, step=1,
                label='Second',
            ),
        ],
        outputs=gr.Video(),
        title="Demo project",
        description="Vietnamese Automatic Speech Recognition Utilizing Audio and Visual Data"
    )

    app.queue().launch(share=True)