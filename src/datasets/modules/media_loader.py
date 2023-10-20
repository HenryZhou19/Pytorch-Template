import os

import cv2
import numpy as np


def load_video(filepath: str, gray_out=False, dtype=np.uint8) -> np.ndarray:
    """
    RGB 24bits video only?
    if gray_out:
        return np.ndarray [C(gray)=1, frame_length, H, W] uint8
    else:
        return np.ndarray [C(RGB)=3, frame_length, H, W] uint8
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    capture = cv2.VideoCapture(filepath)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    C = 1 if gray_out else 3
    video_array = np.zeros((frame_count, frame_height, frame_width, C), dtype=dtype)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            capture.release()
            raise ValueError("Failed to load frame #{} of {}.".format(count, filepath))
        if gray_out:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[..., None]
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_array[count] = frame
    capture.release()
    video_array = video_array.transpose((3, 0, 1, 2))

    return video_array


def _convert_cv2_image_array(image_array, gray_out) -> np.ndarray:
    raw_gray = len(image_array.shape) == 2
    if gray_out:
        if raw_gray:
            image_array = image_array[..., None]
        else:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)[..., None]
    else:
        if raw_gray:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        else:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return image_array


def load_image(filepath: str, gray_out=False) -> np.ndarray:
    """
    if gray_out or raw_gray:
        return np.ndarray [C(gray)=1, H, W] uint8, 16, ...
    else:
        return np.ndarray [C(RGB)=3, H, W] uint8, 16, ...
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    image_array = _convert_cv2_image_array(cv2.imread(filepath, cv2.IMREAD_UNCHANGED), gray_out)
    image_array = image_array.transpose((2, 0, 1))

    return image_array


def load_tif_images(filepath, gray_out=False) -> np.ndarray:
    """
    if gray_out:
        return np.ndarray [C(gray)=1, frame_length, H, W] uint8
    else:
        return np.ndarray [C(RGB)=3, frame_length, H, W] uint8
    """
    assert filepath.endswith('tif') or filepath.endswith('tiff')
    _, array_tuple = cv2.imreadmulti('C2-!220118 cos7 wt er endo int2s 015.tif', flags=cv2.IMREAD_UNCHANGED)
    array_list = []
    for image_array in array_tuple:
        array_list.append(_convert_cv2_image_array(image_array, gray_out))
    images_array = np.stack(array_list).transpose((3, 0, 1, 2))

    return images_array
