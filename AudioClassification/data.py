import logging
import os
import pickle
from pathlib import Path
import numpy as np
import csv
import random, string, os, tqdm, pickle, argparse

logging.basicConfig(level=logging.INFO)

location = os.environ['UNTAR_LOCATION']
target_location = os.environ['LOCATION']
file_name = os.environ['UNTAR_FILE']


def create_output_paths(output_path, mixed=False):
    Path(output_path + "train/male/").mkdir(parents=True, exist_ok=True)
    Path(output_path + "train/female/").mkdir(parents=True, exist_ok=True)

    Path(output_path + "test/male/").mkdir(parents=True, exist_ok=True)
    Path(output_path + "test/female/").mkdir(parents=True, exist_ok=True)

    Path(output_path + "valid/male/").mkdir(parents=True, exist_ok=True)
    Path(output_path + "valid/female/").mkdir(parents=True, exist_ok=True)

    if mixed:
        Path(output_path + "train/mixed/").mkdir(parents=True, exist_ok=True)
        Path(output_path + "test/mixed/").mkdir(parents=True, exist_ok=True)
        Path(output_path + "valid/mixed/").mkdir(parents=True, exist_ok=True)
