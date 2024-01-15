import cv2
import numpy as np


def cv2d_draw_mask_contour(image: np.ndarray, mask: np.ndarray, color: list, brush_size: int = 1) -> np.ndarray:
    """
    Draw mask contours on a given image with specified color
    :param image: numpy image of [H,W,C]
    :param mask: mask of [H,W] with [0,X] where X any value will be treated as mask
    :param color: color to draw mask with [R,G,B]
    :param brush_size: thickness of the edges
    :return: new mask
    """
    image = image.astype(np.uint8)
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, color, brush_size)
    return image

