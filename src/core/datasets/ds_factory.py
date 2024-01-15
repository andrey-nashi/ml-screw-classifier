import inspect

from .ds_seg_bin import DatasetSegmentationBinary


class DatasetFactory:

    _LIST_DATASETS = [
        DatasetSegmentationBinary,
     ]

    _TABLE_DATASETS = {m.__name__:m for m in _LIST_DATASETS}

    @staticmethod
    def create_dataset(dataset_name, dataset_args):
        if dataset_name in DatasetFactory._TABLE_DATASETS:
            return DatasetFactory._TABLE_DATASETS[dataset_name](**dataset_args)

        else:
            raise NotImplemented

    @staticmethod
    def get_dataset_args(dataset_name):
        if dataset_name in DatasetFactory._TABLE_DATASETS:
            target_dataset = DatasetFactory._TABLE_DATASETS[dataset_name]

        output = {}
        model_args = inspect.signature(target_dataset.__init__).parameters
        for arg_name, arg in model_args.items():
            if arg_name != "self":
                output[arg_name] = {"ann": str(arg.annotation), "default": arg.default}

        return output
