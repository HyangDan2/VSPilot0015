import numpy as np
import cv2
from winsdk.windows.graphics.imaging import (
    SoftwareBitmap, BitmapPixelFormat, BitmapAlphaMode
)
from winsdk.windows.storage.streams import Buffer
from PySide6.QtGui import QImage

def sbmp_to_gray(sbmp: SoftwareBitmap) -> np.ndarray:
    try:
        gray8 = SoftwareBitmap.convert(sbmp, BitmapPixelFormat.GRAY8, BitmapAlphaMode.IGNORE)
        w, h = gray8.pixel_width, gray8.pixel_height
        buf = Buffer(w * h)
        gray8.copy_to_buffer(buf)
        arr = np.frombuffer(memoryview(buf), dtype=np.uint8, count=w*h).reshape(h, w)
        return arr
    except Exception:
        pass
    rgba = SoftwareBitmap.convert(sbmp, BitmapPixelFormat.RGBA8, BitmapAlphaMode.IGNORE)
    w, h = rgba.pixel_width, rgba.pixel_height
    buf = Buffer(w * h * 4)
    rgba.copy_to_buffer(buf)
    arr = np.frombuffer(memoryview(buf), dtype=np.uint8, count=w*h*4).reshape(h, w, 4)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)

def numpy_to_qimage(img: np.ndarray) -> QImage:
    if img.ndim == 2:
        h, w = img.shape
        return QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, w*ch, QImage.Format.Format_RGB888).copy()
