import sys
import os

sys.path.append(os.getcwd())
import subprocess
import gradio as gr


def predict(video_path: str):
    inference_file = 'src/models/inferences/main.py'
    cmd = [
        'python',
        inference_file,
        video_path,
        '--clear-fragments',
        '--decode',
    ]
    if not os.access(inference_file, os.X_OK):
        os.chmod(inference_file, 0o755)

    try:
        subprocess.run(
            cmd,
            shell=False,
            check=True,
            stdout=None,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        print(f"Standard Output: {e.stdout}")
        return f"Error: {e.stdout}"

    if not os.path.isfile('results/output.mp4'):
        return video_path
    return 'results/output.mp4'


if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict,
        inputs=gr.Video(),
        outputs=gr.Video(),
        title="Demo project",
        description="Vietnamese Automatic Speech Recognition Utilizing Audio and Visual Data"
    )

    iface.launch(share=True)