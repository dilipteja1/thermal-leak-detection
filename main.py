import os
import sys
import argparse
from pathlib import Path

src = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src)
sys.path.append(os.path.join(src, "dataset_processing"))
sys.path.append(os.path.join(src, 'dataset_processing', 'utils'))
from dataset_processing.loader import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data set loader for pre processing")
    parser.add_argument("--input_dir", "--input-dir", help="the dataset directory to be processed")
    parser.add_argument("--output_dir", "--output-dir", help="output directory")
    args = parser.parse_args()
    reader = DataLoader()
    reader.load()
    reader.annotate()