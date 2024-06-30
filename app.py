import sys
import os

sys.path.append(os.getcwd())
import subprocess
import gradio as gr


def predict(video_path: str):
    output_file = 'results/output.mp4'
    if os.path.isfile(output_file):
        os.remove(output_file)

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

    if not os.path.isfile(output_file):
        return video_path
    return output_file


if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict,
        inputs=gr.Video(),
        outputs=gr.Video(),
        title="Demo project",
        description="Vietnamese Automatic Speech Recognition Utilizing Audio and Visual Data"
    )

    iface.launch(share=True)