import argparse
import csv
import logging
import os
import pickle
import random
from pathlib import Path

from pydub import AudioSegment
from pydub.utils import which
from tqdm import tqdm

AudioSegment.converter = which("ffmpeg")

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


def build_speech_lists(dirs):
    speech_data = []

    for subdir in dirs:
        tmp = [f for f in Path(subdir).glob('**/*.flac') if (os.path.getsize(f) / 5) > 1]
        speech_data.extend(tmp)

    return speech_data


def parse_speakers(speaker_list, seed=42):
    random.seed(seed)
    with open(speaker_list, 'rb') as f:
        speakers = pickle.load(f)

    male_speakers, female_speakers = speakers

    male_speakers.sort()
    female_speakers.sort()
    random.shuffle(male_speakers)
    random.shuffle(female_speakers)

    male_train = male_speakers[:int(0.8 * len(male_speakers))]
    male_valid = male_speakers[int(0.8 * len(male_speakers)):int(0.9 * len(male_speakers))]
    male_test = male_speakers[int(0.9 * len(male_speakers)):]

    female_train = female_speakers[:int(0.8 * len(female_speakers))]
    female_valid = female_speakers[int(0.8 * len(female_speakers)):int(0.9 * len(female_speakers))]
    female_test = female_speakers[int(0.9 * len(female_speakers)):]

    logging.info(
        f"Number of female train, validation, and test speakers: {len(female_train)}, {len(female_valid)}, {len(female_test)}")
    logging.info(
        f"Number of male train, validation, and test speakers: {len(male_train)}, {len(male_valid)}, {len(male_test)}")

    return build_speech_lists(female_train), \
           build_speech_lists(female_valid), \
           build_speech_lists(female_test), \
           build_speech_lists(male_train), \
           build_speech_lists(male_valid), \
           build_speech_lists(male_test)


def generate_simultaneous_speech_overlay(base_sound, speech_list, seed=42):
    random.seed(seed)

    speeches = []
    for path in speech_list:
        speech = AudioSegment.from_file(path)
        speech = augment_speech(speech)
        speeches.append(speech)

    entry_point = random.uniform(0, 0.2 * len(base_sound))

    output = base_sound
    for s in speeches:
        output = output.overlay(s, position=entry_point)

    return output


def generate_successive_speech_overlay(base_sound, speech_list, seed=42):
    random.seed(seed)

    speeches = []
    for path in speech_list:
        speech = AudioSegment.from_file(path)
        speech = augment_speech(speech)
        speeches.append(speech)

    entry_point = random.uniform(0, 0.2 * len(base_sound))

    output = base_sound
    for k, s in enumerate(speeches):
        output = output.overlay(s, position=entry_point)
        entry_point += len(s) + random.uniform(0, 5000)

    return output


def augment_speech(speech):
    if random.getrandbits(1):
        speech = speech.apply_gain(random.uniform(-1, 1))

    if random.getrandbits(1):
        speech = speech[random.uniform(0, 0.5 * len(speech)):]
    else:
        speech = speech[:random.uniform(0, 0.5 * len(speech))]

    return speech


def generate_speech_overlay(base_sound, speech_list, simultaneous):
    if simultaneous:
        return generate_simultaneous_speech_overlay(base_sound, speech_list)
    else:
        return generate_successive_speech_overlay(base_sound, speech_list)


def get_speech_samples(gender, subset, number_of_samples, paths):
    if gender == 'F':
        if subset == 'train':
            return [random.choice(paths['female_train']) for _ in range(number_of_samples)]
        elif subset == 'valid':
            return [random.choice(paths['female_valid']) for _ in range(number_of_samples)]
        elif subset == 'test':
            return [random.choice(paths['female_test']) for _ in range(number_of_samples)]
        else:
            raise ValueError('Unknown subset!')
    elif gender == 'M':
        if subset == 'train':
            return [random.choice(paths['male_train']) for _ in range(number_of_samples)]
        elif subset == 'valid':
            return [random.choice(paths['male_valid']) for _ in range(number_of_samples)]
        elif subset == 'test':
            return [random.choice(paths['male_test']) for _ in range(number_of_samples)]
        else:
            raise ValueError('Unknown subset!')
    else:
        raise ValueError('Unknown gender!')


def generate_sample(subset, output_path, filename, paths, length=60.0, simultan=True):
    base_sound = AudioSegment.silent(duration=length * 1000)

    gender = random.choice(['M', 'F'])
    speeches = get_speech_samples(gender=gender, subset=subset, number_of_samples=random.randint(6, 10), paths=paths)
    simultaneous = random.getrandbits(1) if simultan else 0

    output = generate_speech_overlay(base_sound=base_sound, speech_list=speeches, simultaneous=simultaneous)

    if output.duration_seconds == length:
        subdir = "female" if gender == 'F' else 'male'
        output_name = os.path.join(output_path, subset, subdir, filename + '.flac')
        output.export(output_name, format='flac', parameters=['-ar', '22050'])
        return output_name, subdir

    logging.warning("Running an additional generation, sample not long enough")
    return generate_sample(subset, output_path, filename, length, simultan)


def gen_csv(data_dict, outpath, outname):
    with open(outpath + outname, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['path', 'label'])
        filewriter.writerows(data_dict.items())


parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', dest='output_path', default=os.path.join(target_location, 'dataset'),
                    help='Base path for the dataset')
parser.add_argument('--speaker_file', dest='speaker_file', default=os.path.join(target_location, 'speaker.pkl'),
                    help='Pickle file that stores the speakers')
parser.add_argument('--seed', dest='seed', type=int, default=42, help='Seed for reproducibility')
parser.add_argument('--num_train_samples', dest='train_samples', type=int, default=80,
                    help='Number of samples in the training subset')
parser.add_argument('--num_test_samples', dest='test_samples', type=int, default=10,
                    help='Number of samples in the test subset')
parser.add_argument('--num_valid_samples', dest='validation_samples', type=int, default=10,
                    help='Number of samples in the validation subset')
parser.add_argument('--csv_dir', dest='csv_dir', default=target_location,
                    help='The csv files that contain sample|label mappings are stored there')
parser.add_argument('--csv_flag', dest='csv_flag', default='custom_',
                    help='A name that will be appended to the csv file to differentiate it from other ones')
parser.add_argument('--mixed', dest='mixed', type=int, default=0,
                    help="If set then create a third category that mixes male and female speakers")
parser.add_argument('--simultaneous', dest='simultan', type=bool, default=True,
                    help="If set then overlay two speeches at the same timestamp")
parser.add_argument('--length', dest='length', type=float, default=60.0,
                    help="Length of sample audio in seconds")

args, unknown = parser.parse_known_args()
args = args.__dict__

if __name__ == '__main__':
    create_output_paths(output_path=args['output_path'], mixed=args['mixed'])
    female_train, female_valid, female_test, male_train, male_valid, male_test = parse_speakers(
        speaker_list=args['speaker_file'])
    paths = {
        'female_train': female_train,
        'female_valid': female_valid,
        'female_test': female_test,
        'male_train': male_train,
        'male_valid': male_valid,
        'male_test': male_test
    }

    samples_dict = {}
    for i in tqdm(range(1, args['train_samples'] + 1)):
        out_name, label = generate_sample(subset="train", output_path=args['output_path'], filename=str(i),
                                          length=args['length'], simultan=args['simultan'], paths=paths)
        samples_dict[out_name] = 0 if label == "male" else 1
    gen_csv(data_dict=samples_dict, outpath=args['csv_dir'], outname=args['csv_flag'] + "train.csv")

    samples_dict = {}
    for i in tqdm(range(1, args['test_samples'] + 1)):
        out_name, label = generate_sample(subset="test", output_path=args['output_path'], filename=str(i),
                                          length=args['length'], simultan=args['simultan'], paths=paths)
        samples_dict[out_name] = 0 if label == "male" else 1
    gen_csv(data_dict=samples_dict, outpath=args['csv_dir'], outname=args['csv_flag'] + "test.csv")

    samples_dict = {}
    for i in tqdm(range(1, args['validation_samples'] + 1)):
        out_name, label = generate_sample(subset="valid", output_path=args['output_path'], filename=str(i),
                                          length=args['length'], simultan=args['simultan'], paths=paths)
        samples_dict[out_name] = 0 if label == "male" else 1
    gen_csv(data_dict=samples_dict, outpath=args['csv_dir'], outname=args['csv_flag'] + "valid.csv")