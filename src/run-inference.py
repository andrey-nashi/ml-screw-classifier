import gradio as gr
import numpy as np
import torch
import albumentations as alb

from core.model.smp import SmpModel_Light

from core.utils.cv2d_draw import cv2d_draw_mask_contour

IMAGE_SIZE = 512
PATH_MODEL = "../data/output/weights/unet_resnet_100.ckpt"

transform = alb.Compose([alb.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1, always_apply=True)])
model = SmpModel_Light(model_name="Unet", encoder_name="resnet50", in_channels=3, out_classes=1, loss_func=None, activation="sigmoid")
model = model.load_from_checkpoint(PATH_MODEL)
model.cuda()
model.eval()


def predict(image: np.ndarray):
    print("Received request")

    input = transform(image=image)["image"]

    transformed_image = torch.from_numpy(input / 255)
    transformed_image = transformed_image.permute(2, 0, 1)
    transformed_image = transformed_image.float()
    transformed_image.cuda()
    transformed_image = torch.unsqueeze(transformed_image, 0)

    with torch.no_grad():
        mask_pr = model(transformed_image)
        mask_pr = mask_pr.cpu().numpy()
        mask_pr_out = mask_pr[0][0] * 255
        vis_image = cv2d_draw_mask_contour(input, mask_pr_out, [0, 255, 0], 2)

        return np.asarray(vis_image)
    
inputs = gr.Image()
interface = gr.Interface(fn=predict, inputs=inputs, outputs="image")

#interface.launch(server_name="0.0.0.0")

interface.launch(share=True)
#---------------------------------------------------------------------------------------

#path_in = "test_174.png"

#path_out = "out.png"

#x = cv2.imread(path_in)
#y = predict(x)
#cv2.imwrite(path_out, y)