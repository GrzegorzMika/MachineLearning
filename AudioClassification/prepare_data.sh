#!/bin/bash

source paths.sh
python data.py --length 60.0 --num_train_samples 80 --num_valid_samples 10 --num_test_samples 10 --simultaneous True
python TFRecords.py