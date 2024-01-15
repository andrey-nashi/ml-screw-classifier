import numpy as np
import cv2
import math

def cv2d_remove_blobs(mask: np.ndarray, blob_min_size: int, mode: int = 0):
    """
    Remove all blobs that have less pixels than the given threshold
    :param mask: binary mask 0|255
    :param blob_min_size: min size of the blob
    :param mode - size check mode 
    - MODE_AREA=0 (S < T)
    - MODE_RESOLUTION=1 (w & h < T)
    - MODE_DIAGONAL=2 (sq(w2 + h2) < T)
    :return: binary mask with small blobs removed
    """
    MODE_AREA = 0
    MODE_RESOLUTION = 1
    MODE_DIAGONAL = 2
    
    mask_output = mask.copy()
    mask_output = mask_output.astype(np.uint8)

    ret, labels = cv2.connectedComponents(mask_output)
    for label in range(1, ret):
        xy = np.argwhere(labels == label).T
        
        blob_area = len(xy[0])
        bbox_x_min = int(min(xy[1]))
        bbox_y_min = int(min(xy[0]))
        bbox_x_max = int(max(xy[1]))
        bbox_y_max = int(max(xy[0]))
        
        blob_h = bbox_x_max - bbox_x_min
        blob_w = bbox_y_max - bbox_y_min
        blob_d = math.sqrt(blob_h * blob_h + blob_w * blob_w)

        if mode == MODE_AREA and blob_area < blob_min_size:
            mask_output[labels == label] = 0
        if mode == MODE_RESOLUTION and blob_w < blob_min_size and blob_h < blob_min_size:
            mask_output[labels == label] = 0
        if mode == MODE_DIAGONAL and blob_d < blob_min_size:
            mask_output[labels == label] = 0

    return mask_output