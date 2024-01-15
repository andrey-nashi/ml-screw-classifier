import torch
import albumentations as alb
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from core.datasets.ds_seg_bin import DatasetSegmentationBinary
from core.model.smp import SmpModel_Light


#---------------------------------------------------------------------------------------

def main(path_dir_root, path_json_train, path_json_val, image_size, batch_size, max_epochs):

    transform_train = alb.Compose([
        alb.RandomRotate90(),
        alb.Flip(),
        alb.Transpose(),
        alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        alb.Resize(image_size, image_size, p=1, always_apply=True),
    ])

    transform_val = alb.Compose([alb.Resize(image_size, image_size, p=1, always_apply=True)])

    # ---- Datasets and loaders
    dataset_train = DatasetSegmentationBinary(transform_train)
    dataset_train.load_from_json(path_json_train, path_dir_root)
    dataset_train.oversample(3)
    dataset_valid = DatasetSegmentationBinary(transform_val)
    dataset_valid.load_from_json(path_json_val, path_dir_root)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)

    loss_f = smp.losses.DiceLoss("binary", from_logits=False)
    model = SmpModel_Light(model_name="Unet", encoder_name="resnet50", in_channels=3, out_classes=1, loss_func=loss_f, activation="sigmoid")
    model.set_optimizer(torch.optim.SGD, lr=0.01)


    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

#---------------------------------------------------------------------------------------

if __name__ == '__main__':
    DEFAULT_DIR_ROOT = "../data"
    DEFAULT_JSON_TRAIN = "../data/dataset/ds-train.json"
    DEFAULT_JSON_TEST = "../data/dataset/ds-test.json"
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_EPOCHS = 100
    DEFAULT_IMG_SIZE = 512

    main(DEFAULT_DIR_ROOT, DEFAULT_JSON_TRAIN, DEFAULT_JSON_TEST, DEFAULT_IMG_SIZE, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS)
