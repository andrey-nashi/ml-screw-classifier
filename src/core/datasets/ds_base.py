import copy
import random
import torch
import numpy as np

from torch.utils.data import Dataset

# -----------------------------------------------------------------------------------------
def convert_image2tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array that represents an image like (H,W) or (H,W,3) into tensor like (1,H,W) or (3,H,W).
    :param image: image to be transformed
    :return: Tensor or None if dimensions don't match
    """
    if image.ndim == 3:
        transformed_image = torch.from_numpy(image)
        transformed_image = transformed_image.permute(2, 0, 1)
        return transformed_image.float()
    if image.ndim == 2:
        transformed_image = torch.from_numpy(image)
        transformed_image = torch.unsqueeze(transformed_image, 0)
        return transformed_image.float()
    return None


def convert_list2tensor(data: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(data)
    return tensor


def normalize_numpy(array, mode):
    if mode == 0: return array
    if mode == 1: return array / 255
    if mode == 2: return (array - np.min(array)) / (np.max(array) - np.min(array))


# -----------------------------------------------------------------------------------------


class AbstractDataset(Dataset):

    def __init__(self, transform_func: callable = None):
        super(AbstractDataset, self).__init__()
        self.transform_func = transform_func
        self.is_to_tensor = True
        self.samples_table = []

    def switch_to_tensor(self, forced_flag: bool = None):
        """
        Switch to tensor flag, which specifies whether this dataset will output
        torch.Tensor or numpy array when __getitem__ method is called.
        Default state is true, switch is done automatically if forced_flag is None
        :param forced_flag: if not None, force state
        :return:
        """
        if forced_flag is None: self.is_to_tensor = not self.is_to_tensor
        else: self.is_to_tensor = forced_flag

    # -----------------------------------------------------------------------------------------

    def __len__(self):
        return len(self.samples_table)

    def __getitem__(self, sample_index: int):
        if sample_index < len(self.samples_table):
            return self.samples_table
        else:
            return None

    def __add__(self, dataset):
        assert type(self) == type(dataset)

        output = self.__copy__()
        output.samples_table = self.samples_table + dataset.samples_table
        return output

    def __copy__(self):
        new_obj = self.__class__()
        for k, v in vars(self).items():
            try:
                setattr(new_obj, k, copy.deepcopy(v))
            except:
                pass
        return new_obj

    # -----------------------------------------------------------------------------------------
    def get_data_source(self, index_start: int, element_count: int = 1):
        assert element_count >= 1
        assert index_start >= 0
        assert index_start < len(self.samples_table)

        output = self.samples_table[index_start:index_start + element_count]
        return output

    @property
    def size(self):
        return len(self.samples_table)

    def set_transform_func(self, transform_func: callable):
        self.transform_func = transform_func

    def load_from_json(self, path_file: str, **kwargs):
        return

    def save_to_json(self, path_file: str, **kwargs):
        return

    def split(self, ratio_list: list) -> list:
        """
        Split this dataset into multiple new datasets in a given ratios.
        :param ratio_list: list of float values, sum should be 1
        :return: list of datasets
        """
        total = sum(ratio_list)
        if total != 1: return []

        output = []
        for i in range(0, len(ratio_list)):
            if i == 0:
                limit_min =0
                limit_max = int(ratio_list[i] * len(self.samples_table))
            elif i == len(ratio_list) - 1:
                limit_min = int(sum(ratio_list[0:(len(ratio_list) - 1)]) * len(self.samples_table))
                limit_max = len(self.samples_table)
            else:
                limit_min = int(sum(ratio_list[0:i]) * len(self.samples_table))
                limit_max = int(sum(ratio_list[0:i+1]) * len(self.samples_table))
            dataset = self.__copy__()
            dataset.samples_table = self.samples_table[limit_min:limit_max]
            output.append(dataset)

        return output

    def shuffle(self):
        random.shuffle(self.samples_table)
