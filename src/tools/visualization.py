import torch
import numpy as np
from imageio import mimsave
from PIL import Image, ImageDraw, ImageFont


def unnormalize_image(image: np.ndarray, std: tuple, mean: tuple) -> np.ndarray:
    image = (image * std) + mean
    image = (image * 255).astype("uint8")
    return image.clip(0, 255)


def save_as_gif(
    video_tensor: torch.Tensor,
    save_path: str = "sample.gif",
    std: tuple = None,
    mean: tuple = None,
):
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_image(
            image=video_frame.permute(1, 2, 0).numpy(),
            std=std,
            mean=mean,
        )
        frames.append(frame_unnormalized)
    kargs = {"duration": 0.25}
    mimsave(save_path, frames, "GIF", **kargs)
    return save_path


def display_gif(gif_path: str) -> Image:
    return Image(filename=gif_path)


def draw_text_on_image(
    image: np.ndarray,
    text: str,
    position: tuple = (20, 20),
    color: tuple = (0, 0, 255),
    font_size: int = 20,
) -> np.ndarray:
    font = ImageFont.truetype(
        font="fonts/OpenSans-Regular.ttf",
        size=font_size,
    )
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    draw.text(
        xy=position,
        text=text,
        fill=color,
        font=font,
    )
    return np.array(pil_image)
