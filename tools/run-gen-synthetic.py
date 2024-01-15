import os
import cv2
import numpy as np
import random


# -------------------------------------------------------------------------


class Tools:

    @staticmethod
    def add_gaussian_noise(image, mean=0, sigma=25):
        row, col  = image.shape
        gauss = np.random.normal(mean, sigma, (row, col))
        noisy = np.clip(image + gauss, 0, 255)
        return noisy.astype(np.uint8)

    @staticmethod
    def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
        row, col = image.shape
        noisy = np.copy(image)

        # Salt noise
        salt = np.random.rand(row, col) < salt_prob
        noisy[salt, :] = 255

        # Pepper noise
        pepper = np.random.rand(row, col) < pepper_prob
        noisy[pepper, :] = 0

        return noisy.astype(np.uint8)

# -------------------------------------------------------------------------


class SyntheticGenerator:

  
    def __init__(self, templates: list, path_root: str, texture_path: str = None):
        self._template_db = []
        self._texture_path = texture_path
        self._texture = None
        self._load_templates(templates, path_root)

    def _load_templates(self, templates: list, path_root: str):
        for template in templates:
            path_template = template[0]
            path_mask = template[1]

            image = cv2.imread(os.path.join(path_root, path_template))
            mask = cv2.imread(os.path.join(path_root, path_mask))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            self._template_db.append([image, mask])

        if os.path.exists(self._texture_path):
            self._texture = cv2.imread(self._texture_path)
            self._texture = cv2.cvtColor(self._texture, cv2.COLOR_BGR2GRAY)

    def _generate_background(self, width: int, height: int):
        INTENSITY_MIN = 180
        INTENSITY_MAX = 210

        intensity = random.randint(INTENSITY_MIN, INTENSITY_MAX)

        background = np.full((height, width), intensity)
        background = Tools.add_gaussian_noise(background)
        background = cv2.GaussianBlur(background, (5, 5), 0)
        return background

    def _generate_indent(self, image: np.ndarray, mask: np.ndarray):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        screw_contour = contours[0]

        index_max = screw_contour.shape[0]
        index = random.randint(0, index_max)
        xy = screw_contour[index][0]

        indent_radius = random.randint(20, 33)

        mask_obj = mask.copy()
        mask_obj = cv2.circle(mask_obj, xy, indent_radius, 0, thickness=-1)

        mask_defect = np.zeros(mask_obj.shape, dtype=np.uint8)
        mask_defect = cv2.circle(mask_defect, xy, indent_radius, 255, thickness=-1)
        mask_defect[mask == 0] = 0

        return image, mask_obj, mask_defect

    def _generate_scratch(self, image: np.ndarray, mask: np.ndarray):
        mask_eroded = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=8)
        x = np.argwhere(mask_eroded == 255)
        index_max = x.shape[0]
        index = random.randint(0, index_max)
        xy = [x[index][1], x[index][0]]

        image_defect = image.copy()
        image_defect = Tools.add_gaussian_noise(image_defect, 0, 50)
        image_defect = cv2.GaussianBlur(image_defect, (5, 5), 0)

        if self._texture is not None:
            texture = cv2.resize(self._texture, image_defect.shape)
            image_defect = cv2.addWeighted(image_defect, 0.4, texture, 0.6, 0)

            brightness = random.randint(0, 15)
            brightness_matrix = np.ones(image_defect.shape, dtype=np.uint8) * brightness
            image_defect = cv2.add(image_defect, brightness_matrix)

        a = random.randint(20, 40)
        b = random.randint(20, 40) + 5
        mask_defect = np.zeros(mask.shape, dtype=np.uint8)
        mask_defect = cv2.ellipse(mask_defect, xy, [a,b], 45, 0, 360, 255, thickness=-1)
        mask_defect[mask == 0] = 0

        image_output = image.copy()
        image_output[mask_defect == 255] = image_defect[mask_defect == 255]

        return image_output, mask, mask_defect

    def _assemble(self, image_background: np.ndarray, image_object: np.ndarray, mask_object: np.ndarray):
        output = image_background.copy()
        output[mask_object == 255] = 0
        output[mask_object == 255] = image_object[mask_object == 255]
        return output

    def generate(self):
        image_index = random.randint(0, len(self._template_db) - 1)
        image = self._template_db[image_index][0]
        mask = self._template_db[image_index][1]

        height, width = image.shape

        background = self._generate_background(width, height)

        mode = random.randint(0, 1)

        if mode == 0:
            image_obj, mask_obj, mask_defect = self._generate_indent(image, mask)
        if mode == 1:
            image_obj, mask_obj, mask_defect = self._generate_scratch(image, mask)

        image_out = self._assemble(background, image_obj, mask_obj)
        return image_out, mask_defect

# -------------------------------------------------------------------------

def main():
    PATH_ROOT = "../data/synth"
    PATH_TEXTURE = "../data/synth/texture.png"
    TEMPLATES = [
        ["template01.png", "mask01.png"],
        ["template02.png", "mask02.png"]
    ]

    output = "./synthetic"
    if not os.path.exists(output):
        os.makedirs(output)

    for i in range(0, 200):


        generator = SyntheticGenerator(TEMPLATES, PATH_ROOT, PATH_TEXTURE)
        image_out, mask_defect = generator.generate()

        file_image = "synth-" + str(i).zfill(3) + "-img.png"
        file_mask = "synth-" + str(i).zfill(3) + "-mask.png"

        cv2.imwrite(os.path.join(output, file_image), image_out)
        cv2.imwrite(os.path.join(output, file_mask), mask_defect)

# -------------------------------------------------------------------------


if __name__ == '__main__':
    main()
