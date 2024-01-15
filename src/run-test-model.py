import os

import cv2
import numpy as np
import torch
import albumentations as alb
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from core.datasets.ds_seg_bin import DatasetSegmentationBinary
from core.model.smp import SmpModel_Light

from core.utils.cv2d_draw import cv2d_draw_mask_contour
from core.utils.cv2d_utils import cv2d_remove_blobs
from core.utils.stat_clf import ml_stat_binary_classification

#---------------------------------------------------------------------------------------

def compare_as_classifier(mask_gt, mask_pr, blob_min_size: int):
    output = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    mask_pr_bin = mask_pr.copy()
    mask_pr_bin[mask_pr_bin >= 128] = 255
    mask_pr_bin[mask_pr_bin < 128] = 0

    mask_pr_bin = cv2d_remove_blobs(mask_pr_bin, blob_min_size)

    if np.max(mask_gt) == 255 and np.max(mask_pr_bin) == 255:
        output["tp"] += 1
    if np.max(mask_gt) == 0 and np.max(mask_pr_bin) == 0:
        output["tn"] += 1
    if np.max(mask_gt) == 255 and np.max(mask_pr_bin) == 0:
        output["fn"] += 1
    if np.max(mask_gt) == 0 and np.max(mask_pr_bin) == 255:
        output["fp"] += 1

    return output


def run_test(path_dir_root: str, path_json_test: str, path_model: str, path_output: str, image_size: int):

    transform_val = alb.Compose([alb.Resize(image_size, image_size, p=1, always_apply=True)])
    dataset_valid = DatasetSegmentationBinary(transform_val)
    dataset_valid.load_from_json(path_json_test, path_dir_root)
    valid_dataloader = DataLoader(dataset_valid, batch_size=8, shuffle=False, num_workers=4)

    model = SmpModel_Light(model_name="Unet", encoder_name="resnet50", in_channels=3, out_classes=1, loss_func=None, activation="sigmoid")
    model = model.load_from_checkpoint(path_model)

    model.cuda()
    model.eval()


    result_clf = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}


    with torch.no_grad():
        index_g = 0

        for batch in valid_dataloader:

            image = batch[0].cuda()
            mask_gt = batch[1].cuda()

            mask_pr = model(image)

            mask_pr = mask_pr.cpu().numpy()
            mask_gt = mask_gt.cpu().numpy()
            image = image.cpu().numpy()

            b, x, y, z = mask_pr.shape

            for bid in range(0, b):
                mask_pr_out = mask_pr[bid][0] * 255
                mask_gt_out = mask_gt[bid][0] * 255
                image_out = image[bid][0] * 255
                
                info = dataset_valid.get_data_source(index_g)
      
                file_name = os.path.basename(info[0]["image"])
                print("Saving result for", file_name)

                # ---- Predicted masks
                file_path = os.path.join(path_output["mask_pr"], file_name)
                cv2.imwrite(file_path, mask_pr_out)

                # ---- Visualized mask contours
                image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
                vis_image = cv2d_draw_mask_contour(image_out, mask_pr_out, [0, 0, 255], 2)
                vis_image = cv2d_draw_mask_contour(vis_image, mask_gt_out, [0, 255, 0], 2)

                file_path = os.path.join(path_output["vis"], file_name)
                cv2.imwrite(file_path, vis_image)


                result = compare_as_classifier(mask_gt_out, mask_pr_out, 30)
                for k in result_clf: 
                    result_clf[k] += result[k]


                index_g += 1


    print("-------------------------------------------------")
    print(result_clf)
    print(ml_stat_binary_classification(**result_clf))

#---------------------------------------------------------------------------------------



if __name__ == '__main__':
    DEFAULT_IMAGE_SIZE = 512
    DEFAULT_DIR_ROOT = "../data"
    DEFAULT_JSON_TEST = "../data/dataset/ds-test.json"
    DEFAULT_MODEL_PATH = "../data/output/weights/unet_resnet_100.ckpt"
    DEFAULT_OUTPUT = {
        "mask_pr": "../data/output/mask_pr",
        "vis": "../data/output/vis",
    }

    for d in DEFAULT_OUTPUT:
        if not os.path.exists(DEFAULT_OUTPUT[d]):
            os.makedirs(DEFAULT_OUTPUT[d])

    run_test(DEFAULT_DIR_ROOT, DEFAULT_JSON_TEST, DEFAULT_MODEL_PATH, DEFAULT_OUTPUT, DEFAULT_IMAGE_SIZE)