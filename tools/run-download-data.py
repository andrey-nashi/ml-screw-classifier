import os.path

import gdown
import zipfile

GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/1FVx-1is8EYGliE56XxyG6rsh6W1K-YqK/view?usp=sharing"
DOWNLOAD_PATH = "../data"

def download_data(url: str, path_dir: str):
    path_file = os.path.join(path_dir, "screws.zip")
    # ---- Download from google drive by the provided URL
    gdown.download(url, path_file, quiet=False, fuzzy=True)

    print(path_file)
    # ---- Unzip, make the dataset nice
    with zipfile.ZipFile(path_file, 'r') as zip_ref:
        zip_ref.extractall(path_dir)


download_data(GOOGLE_DRIVE_URL, DOWNLOAD_PATH)
