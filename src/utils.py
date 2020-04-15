import cv2
import numpy as np
import torch
from typing import Tuple


def change_range(img: np.ndarray, in_range: Tuple[int, int], out_range: Tuple[int, int]):
    """

    :param img:
    :param in_range:
    :param out_range:
    :return:
    """
    if in_range != out_range:
        scale = (np.float32(out_range[1]) - np.float32(out_range[0])) / (
                    np.float32(in_range[1]) - np.float32(in_range[0]))
        bias = np.float32(out_range[0]) - np.float32(in_range[0]) * scale
        img = img * scale + bias

    return img


def load_img(img_path: str, to_rgb: bool = True, color_img: bool = True):
    """
    Loads grayscale/color image from the given path.

    Returns 3-dimensional tensor of shape (height, width, 3) in case of color image
    and (height, width, 1) -- for grayscale one.
    :param img_path:
    :param to_rgb:
    :param color_img:
    :return:
    """
    is_color = cv2.IMREAD_COLOR if color_img else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(img_path, is_color)

    if img is None:
        raise RuntimeError(f"Image wasn't loaded from '{img_path}'!")

    if to_rgb and color_img:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not color_img and len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)

    assert len(img.shape) == 3, "Image must be 3d array!"

    return img

def save_img(img: np.ndarray, dest: str, to_bgr=True):
    if to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dest, img)

def central_crop(img: np.ndarray, crop_size: int):
    """

    :param img:
    :param crop_size:
    :return:
    """
    h, w = img.shape[:2]
    center_x, center_y = int(np.ceil(w / 2)), int(np.ceil(h / 2))
    left, right = center_x - crop_size // 2, center_x + crop_size // 2
    up, bottom = center_y - crop_size // 2, center_y + crop_size // 2

    crop = img[up:bottom, left:right]
    assert crop.shape[0] == crop_size and crop.shape[1] == crop_size, f"Size of the cropped image doesn't match {crop_size}"

    return crop


def numpy2tensor(img: np.ndarray,
                 in_range: Tuple[int, int] = (0, 255),
                 out_range: Tuple[int, int] = (-1, 1),
                 device: torch.device = 'cuda'
                 ):
    """
    Converts image array from one range to another, cast it to torch.Tensor,
     transforms from (H, W, C) to (B, C, H, W) and place on the specified device.

    :param img:
    :param in_range:
    :param out_range:
    :param device:
    :return:
    """
    img = change_range(img, in_range=in_range, out_range=out_range)
    img = np.transpose(img, axes=(2, 0, 1))
    img = np.expand_dims(img, axis=0)
    tensor_img = torch.as_tensor(img, dtype=torch.float32, device=device)

    return tensor_img


def tensor2numpy(tensor: torch.Tensor,
                 in_range: Tuple[int, int] = (-1, 1),
                 out_range: Tuple[int, int] = (0, 255)
                 ):
    """
    Casts image tensor to numpy array and converts it from one range to another.

    :param tensor:
    :param in_range:
    :param out_range:
    :return:
    """
    img = tensor.cpu().numpy()[0]
    img = np.transpose(img, axes=(1, 2, 0))
    img = change_range(img, in_range=in_range, out_range=out_range).astype(np.uint8)
    return img

