import os.path

import gdown
import tarfile

DOWNLOAD_PATH = "../data/"

GOOGLE_DRIVE_OUTPUT = "https://drive.google.com/file/d/1qCQsKZfWfOmvpCfAKsfiPEId9Wp74XHY/view?usp=sharing"
FILE_OUTPUT_NAME = "output.tar.gz"

GOOGLE_DRIVE_DATA = "https://drive.google.com/file/d/1iCpyfVszYy0GfT4o3HabMiuJaDVnMPVd/view?usp=sharing"
FILE_DATA_NAME = "data.tar.gz"

# ---------------------------------------------------------------------------
def download_data(url: str, path_dir: str, file_name: str):
    path_file = os.path.join(path_dir, file_name)
    # ---- Download from google drive by the provided URL
    gdown.download(url, path_file, quiet=False, fuzzy=True)

def extract(source: str, destination: str):
    file = tarfile.open(source)
    file.extractall(destination)
    file.close()

# ---------------------------------------------------------------------------

print(">>> DOWNLOADING DELIVERABLES - WEIGHTS & IMAGES")
download_data(GOOGLE_DRIVE_OUTPUT, DOWNLOAD_PATH, FILE_OUTPUT_NAME)

print(">>> EXTRACTING  DELIVERABLES - WEIGHTS & IMAGES")
extract(os.path.join(DOWNLOAD_PATH, FILE_OUTPUT_NAME), DOWNLOAD_PATH)

# ---------------------------------------------------------------------------

print(">>> DOWNLOADING DATA - IMAGES & MASKS")
download_data(GOOGLE_DRIVE_OUTPUT, DOWNLOAD_PATH, FILE_OUTPUT_NAME)

print(">>> EXTRACTING DATA - IMAGES & MASKS")
extract(os.path.join(DOWNLOAD_PATH, FILE_OUTPUT_NAME), DOWNLOAD_PATH)