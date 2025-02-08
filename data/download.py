"""Download Ames Housing dataset from public source."""
import urllib.request
import os

URL = "https://raw.githubusercontent.com/jamesturk/agate/master/examples/ames_housing.csv"
# Alternative: kaggle datasets download -d prevek18/ames-housing-dataset

DEST = os.path.join(os.path.dirname(__file__), "train.csv")

if os.path.exists(DEST):
    print("Dataset already exists.")
else:
    print("Downloading Ames Housing dataset...")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved to {DEST}")
