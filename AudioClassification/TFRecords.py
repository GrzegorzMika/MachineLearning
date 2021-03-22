import logging
import os

import librosa
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tqdm import tqdm
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)

target_location = os.environ['LOCATION']


def _bytes_feature(value):
    try:
        value = value.numpy()
    except AttributeError:
        pass
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    return tf.io.serialize_tensor(array)


def gen_example(sound_clip, sr, label, shape):
    return {
        'x': _int64_feature(shape[0]),
        'y': _int64_feature(shape[1]),
        'sr': _int64_feature(sr),
        'label': _int64_feature(label),
        'feature': _bytes_feature(serialize_array(sound_clip))
    }


def gen_tfr(tfr_dir, csv_path, output_name, max_files, seed=42):
    logging.info(f"Parsing {csv_path}")

    df = pd.read_csv(csv_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    splits = len(df) // max_files + 1
    if len(df) % max_files == 0:
        splits -= 1

    logging.info(f" Using {splits} shard(s) for {len(df)} files, with up to {max_files} samples per shard")
    file_counter = 0
    for i in tqdm(range(splits)):
        filename = f"{tfr_dir}{i + 1}of{splits}_{output_name}.tfrecords"
        writer = tf.io.TFRecordWriter(filename)

        current_shard_count = 0
        while current_shard_count < max_files:
            index = i * max_files + current_shard_count
            if index == len(df):
                break

            row = df.iloc[index]

            sound_clip, sr = librosa.load(row[0], sr=22050)
            sound_clip = np.expand_dims(sound_clip, axis=1)
            if sound_clip.shape[0] != 1323000:
                logging.warning(f"{row[0]} was not of fit shape: {sound_clip.shape}")
                current_shard_count += 1
                continue

            data = gen_example(sound_clip, sr, row[1], sound_clip.shape)

            out = tf.train.Example(feature=tf.train.Features(feature=data))
            writer.write(out.SerializeToString())
            current_shard_count += 1
            file_counter += 1
        writer.close()

    logging.info(f"Parsed {str(file_counter)} files for {output_name}")


def generate_monitoring_sample(tfr_dir, csv_path, output_name, number_of_samples, use_all=True):
    logging.info(f"Parsing {csv_path} to enable logging some statistics")

    df = pd.read_csv(csv_path)
    y = df.pop('label').to_frame()

    if use_all:
        df_x = df
        df_y = y
    else:
        test_size = number_of_samples / len(df)
        _, df_x, _, df_y = sklearn.model_selection.train_test_split(df, y, test_size)

    x = np.empty(shape=(len(df_x), 1323000, 1), dtype=np.float64)
    y = []

    for i, row in df.iterrows():
        sound_clip, sr = librosa.load(row[0], sr=22050)
        sound_clip = np.expand_dims(sound_clip, axis=1)
        label = df_y.iloc[i]
        x[i] = sound_clip
        y.append(label[0])

    y = np.asarray(y, dtype='int8')
    np.save(os.path.join(tfr_dir, output_name + '_y_monitor.npy'), y)
    np.save(os.path.join(tfr_dir, output_name + '_x_monitor.npy'), x)


def main(args):
    tfr_dir = args['output_path']
    Path(tfr_dir).mkdir(parents=True, exist_ok=True)

    gen_tfr(tfr_dir=tfr_dir, csv_path=args['test_list'], output_name="test", max_files=args['test_max'],
            seed=args['seed'])
    gen_tfr(tfr_dir=tfr_dir, csv_path=args['validation_list'], output_name="valid", max_files=args['valid_max'],
            seed=args['seed'])
    gen_tfr(tfr_dir=tfr_dir, csv_path=args['train_list'], output_name="train", max_files=args['train_max'],
            seed=args['seed'])

    if args['use_monitoring']:
        generate_monitoring_sample(tfr_dir=tfr_dir, csv_path=args['test_list'], output_name="test",
                                   number_of_samples=args['test_monitor'], use_all=args['use_all_samples'])
        generate_monitoring_sample(tfr_dir=tfr_dir, csv_path=args['validation_list'], output_name="valid",
                                   number_of_samples=args['valid_monitor'], use_all=args['use_all_samples'])
        generate_monitoring_sample(tfr_dir=tfr_dir, csv_path=args['train_list'], output_name="train",
                                   number_of_samples=args['train_monitor'], use_all=args['use_all_samples'])


parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', dest='output_path', default=os.path.join(target_location, 'dataset', 'tfr_dir'),
                    help='Base path for the dataset')
parser.add_argument('--train_list', dest='train_list', default=os.path.join(target_location, 'custom_train.csv'),
                    help="CSV file that stores the training files")
parser.add_argument('--validation_list', dest='validation_list',
                    default=os.path.join(target_location, 'custom_valid.csv'),
                    help="CSV file that stores the validation files")
parser.add_argument('--test_list', dest='test_list', default=os.path.join(target_location, 'custom_test.csv'),
                    help="CSV file that stores the test files")
parser.add_argument('--files_per_train_shard', dest='train_max', type=int, default=50,
                    help='Number of files for the TFRecord file')
parser.add_argument('--files_per_test_shard', dest='test_max', type=int, default=50,
                    help='Number of files for the TFRecord file')
parser.add_argument('--files_per_valid_shard', dest='valid_max', type=int, default=50,
                    help='Number of files for the TFRecord file')
parser.add_argument('--use_monitoring_samples', dest='use_monitoring', type=bool, default=True,
                    help='Whether to create an additional numpy array that contains samples that can be used to generate live statistics during training')
parser.add_argument('--use_all_samples', dest='use_all_samples', type=bool, default=False,
                    help='For small datasets use all available subset samples to generate monitoring data')
parser.add_argument('--num_train_monitor', dest='train_monitor', type=int, default=25,
                    help='Number of train samples to store in a numpy array to observe live training statistic on')
parser.add_argument('--num_test_monitor', dest='test_monitor', type=int, default=25,
                    help='Number of test samples to store in a numpy array to observe live training statistics on')
parser.add_argument('--num_valid_monitor', dest='valid_monitor', type=int, default=25,
                    help='Number of valid samples to store in a numpy array to observe live training statistics on')

args, unknown = parser.parse_known_args()
args = args.__dict__

if __name__ == '__main__':
    main(args)
