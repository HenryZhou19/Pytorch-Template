import os

import cv2
import numpy as np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

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
    video_array = np.zeros((frame_count, frame_height, frame_width, C), dtype=dtype)  # [frame_length, H, W, C]

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            capture.release()
            raise ValueError(f'Failed to load frame #{count} of {filepath}.')
        if gray_out:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[..., None]
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_array[count] = frame
    capture.release()
    video_array = video_array.transpose((3, 0, 1, 2))

    return video_array


def save_video(video_array: np.ndarray, filepath: str, fps):
    """
    RGB 24bits video only?
    video_array: np.ndarray [C(gray, RGB)=1 or 3, frame_length, H, W] uint8
    """
    video_array = video_array.transpose((1, 2, 3, 0))
    
    gray_in = video_array.shape[-1] == 1
    container_format = filepath.split('.')[-1]
    
    if container_format == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif container_format == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        raise ValueError(f'Unsupported file format: {container_format}')
    
    out = cv2.VideoWriter(filepath, fourcc, fps, tuple(video_array.shape[1:3]))
    for frame_array in video_array:  # [frame_length, H, W, C]
        if gray_in:
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2BGR)
        else:
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        out.write(frame_array)
    out.release()


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
