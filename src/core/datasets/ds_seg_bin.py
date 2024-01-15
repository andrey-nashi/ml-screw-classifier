import os
import cv2
import json
import numpy as np
import random
from .ds_base import convert_image2tensor, normalize_numpy
from .ds_base import AbstractDataset


class DatasetSegmentationBinary(AbstractDataset):

    SERIAL_KEY_DATASET = "db"
    SERIAL_KEY_IMAGE = "image"
    SERIAL_KEY_MASK = "mask"
    FLAG = "is_real"

    BK_IMAGE = "image"
    BK_MASK = "mask"
    BATCH_KEYS = [BK_IMAGE, BK_MASK]

    NORMALIZE_NONE = 0
    NORMALIZE_255 = 1
    NORMALIZE_MINMAX = 2

    def __init__(self, transform_func: callable = None, norm_image_mode: bool = NORMALIZE_255, norm_mask_mode: bool = NORMALIZE_255):
        """
        Dataset for binary segmentation task where mask is given as 0/255 image
        :param transform_func: should be a callable method with arguments 'image' and 'mask'
        which would return a dictionary with 'image' and 'mask' fields.
        Albumentations style transformation function
        """
        super().__init__(transform_func)
        self.path_root_dir = None
        self.norm_image_mode = norm_image_mode
        self.norm_mask_mode = norm_mask_mode

    def load_from_json(self, path_file: str, path_root_dir: str = None) -> bool:
        """
        Load binary segmentation dataset from a JSON file specified by path
        :param path_file: absolute path to JSON file, must have the following format.
        {"dataset": [{"image": <path_to_image>, "mask": <path_to_mask>}, {...}]}
        :param path_root_dir: path to root directory with images or masks, it will
        be used as prefix to all paths in the JSON dataset
        :return: True if success, False if failed
        """
        if True:
        #try:
            f = open(path_file, "r")
            data = json.load(f)
            f.close()

            self.path_root_dir = path_root_dir

            for sample in data[self.SERIAL_KEY_DATASET]:
                path_image = sample[self.SERIAL_KEY_IMAGE]
                path_mask = sample[self.SERIAL_KEY_MASK]
                flag = sample[self.FLAG]

                self.samples_table.append({self.SERIAL_KEY_IMAGE: path_image, self.SERIAL_KEY_MASK: path_mask, self.FLAG: flag})
            return True
        #except Exception:
        #    return False

    def save_to_json(self, path_file: str, **kwargs):
        """
        Save this dataset into a JSON file by the given path.
        :param path_file: path to the output JSON file
        :param kwargs:
        :return: True if success, False if failed
        """
        try:
            f = open(path_file, "w")
            json.dump({self.SERIAL_KEY_DATASET: self.samples_table}, f)
            f.close()
            return True
        except Exception:
            return False

    def __getitem__(self, sample_index: int):
        """
        Overloded method to be used by torch.Dataloader. Will output a list of two elements
        :param sample_index: index of the sample
        :return: list of two items
        - [image, mask] where both image and mask are numpy arrays or Tensors (depending on the is_to_tensor flag)
        - [image, None]
        """
        path_image = self.samples_table[sample_index][self.SERIAL_KEY_IMAGE]
        path_mask = self.samples_table[sample_index][self.SERIAL_KEY_MASK]

        # ---- Dataset with both image and mask are specified
        if path_image is not None and path_mask is not None:

            if self.path_root_dir is not None:
                path_image = os.path.join(self.path_root_dir, path_image)
                path_mask = os.path.join(self.path_root_dir, path_mask)

            image = cv2.imread(path_image)

            if self.samples_table[sample_index][self.SERIAL_KEY_MASK] != "":
                mask = cv2.imread(path_mask)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]))

            transformed = self.transform_func(image=image, mask=mask)
            transformed_image = transformed[self.BK_IMAGE]
            transformed_mask = transformed[self.BK_MASK]

            transformed_image = normalize_numpy(transformed_image, self.norm_image_mode)
            transformed_mask = normalize_numpy(transformed_mask, self.norm_mask_mode)

            if self.is_to_tensor:
                transformed_image = convert_image2tensor(transformed_image)
                transformed_mask = convert_image2tensor(transformed_mask)

            return transformed_image, transformed_mask

        # ---- Dataset with only image specified, can be used in testing
        if path_image is not None and path_mask is None:
            if self.path_root_dir is not None:
                path_image = os.path.join(self.path_root_dir, path_image)

            image = cv2.imread(path_image)
            transformed = self.transform_func(image=image)
            transformed_image = transformed[self.BK_IMAGE]

            transformed_image = normalize_numpy(transformed_image, self.norm_image_mode)

            if self.is_to_tensor:
                transformed_image = convert_image2tensor(transformed_image)

            return transformed_image, None

    def oversample(self, factor: int):
        candidates = []
        for sample in self.samples_table:
            if sample[self.FLAG] == 1 and sample[self.BK_MASK] != "":
                candidates.append(sample)

        for i in range(0, factor):
            random.shuffle(candidates)
            for sample in candidates:
                x = sample.copy()
                self.samples_table.append(x)

        