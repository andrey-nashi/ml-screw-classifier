import numpy as np
import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

from .model_raw import SmpModel

class SmpModel_Light(pl.LightningModule):

    def __init__(self, model_name: str, encoder_name: str, in_channels: int, out_classes: int, loss_func: callable = None, activation: str = None):
        """
        Initialize segmentation model with given architecture, encoder, number of channels.
        :param model_name: model architecture [Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN]
        :param encoder_name: encoder for the given model. See https://github.com/qubvel/segmentation_models.pytorch for more details.
        Recommended encoders are
        * resnet18, resnet34, resnet50, resnet101, resnet152
        * efficientnet-b0 -> efficientnet-b7
        :param in_channels: number of channels, 3 for RGB input
        :param out_classes: number of output channels, 1 for binary segmentation
        :param loss_func: loss function (DICE is recommended)
        :param is_save_log: if True will save logs via PL hook, False, skip calculating and saving metrics
        :param kwargs:
        """
        super().__init__()

        # ---- Force pytorch lighting to ignore saving loss function, and other flags
        self.save_hyperparameters(ignore=["loss_func"])

        # ---- Create smp model with given parameters
        self.model = SmpModel(model_name=model_name, encoder_name=encoder_name, in_channels=in_channels,
                              out_classes=out_classes, activation=activation)

        # ---- Initialize imagenet like mean and standard deviation, will not work for non-3 channels _most_likely_
        # ---- FIXME add non-3 channel support
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_func = loss_func

        self.optimizer = None
        self.optimizer_lr = None

    def set_loss_func(self, loss_func: callable):
        self.loss_func = loss_func

    def set_optimizer(self, optimizer: callable, lr: float):
        self.optimizer = optimizer
        self.optimizer_lr = lr

    def forward(self, image: torch.Tensor):
        # ---- This check is useful for testing
        if image.device != self.device:
            image = image.to(self.device)
        #image = (image - self.mean) / self.std
        mask = self.model.forward(image)

        return mask

    def training_step(self, batch, batch_index: int):
        image = batch[0]
        mask_gt = batch[1]

        assert image.ndim == 4
        assert image.max() <= 1.0 and image.min() >= -1
        assert mask_gt.max() <= 1.0 and mask_gt.min() >= 0

        mask_pr = self.forward(image)

        loss = self.loss_func(mask_pr, mask_gt)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        return

    def validation_step(self, batch, batch_index: int):
        image = batch[0]
        mask_gt = batch[1]

        assert image.ndim == 4
        assert image.max() <= 1.0 and image.min() >= -1
        assert mask_gt.max() <= 1.0 and mask_gt.min() >= 0

        mask_pr = self.forward(image)
        loss = self.loss_func(mask_pr, mask_gt)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        return

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), self.optimizer_lr)
        return optimizer

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict on a single image
        :param image: [H, W, 3] image UINT8
        :return: numpy array of the binary mask [H, W] mask(x,y) in [0|255]
        """
        transformed_image = torch.from_numpy(image / 255)
        transformed_image = transformed_image.permute(2, 0, 1)
        transformed_image = transformed_image.float()
        transformed_image = (transformed_image - self.mean) / self.std

        if self.training: self.eval()
        with torch.no_grad():
            model_output = self.forward(transformed_image)
            model_output = model_output[0][0].cpu().numpy()
            model_output = model_output * 255

        model_output = model_output.astype(np.uint8)
        return model_output