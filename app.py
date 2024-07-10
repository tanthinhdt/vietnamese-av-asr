import gradio as gr
import os
import subprocess


def setup_environment():
    cmd = [
        'bash', 'scripts/prepare.sh',
        '--platform', 'any'
    ]
    subprocess.run(cmd, shell=False, capture_output=False, stdout=None)


def predict(video_path: str):
    output_file = os.path.join('results', 'output.mp4')
    if os.path.isfile(output_file):
        os.remove(output_file)

    inference_file = 'src/models/inferences/main.py'
    cmd = [
        'python',
        inference_file,
        video_path,
        '--clear-fragments',
        '--decode'
    ]

    try:
        subprocess.run(
            cmd,
            shell=False,
            stdout=None,
            capture_output=False
        )
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        print(f"Standard Output: {e.stdout}")

    if not os.path.isfile(output_file):
        return video_path

    return output_file


if __name__ == "__main__":
    setup_environment()
    app = gr.Interface(
        fn=predict,
        inputs=gr.Video(),
        outputs=gr.Video(),
        title="Demo project",
        description="Vietnamese Automatic Speech Recognition Utilizing Audio and Visual Data"
    )

    app.launch(share=True)