import logging
import os
import pickle

logging.basicConfig(level=logging.INFO)

location = os.environ['UNTAR_LOCATION']
target_location = os.environ['LOCATION']
file_name = os.environ['UNTAR_FILE']

dirs = os.listdir(os.path.join(location, file_name))

with open(os.path.join(location, "SPEAKERS.TXT"), "r") as rf:
    text = rf.read()

text = text.splitlines()
text = text[12:]

available_speakers = {}
male_speaker = []
female_speaker = []

logging.info("Finding available speakers...")
for line in text:
    if file_name in line:
        line = line.split('|')
        id = int(line[0])
        available_speakers[id] = line[1].strip()

logging.info("Splitting available speakers into male and female speakers...")
for d in dirs:
    gender = available_speakers[int(d)]
    if gender == 'M':
        male_speaker.append(os.path.join(location, file_name, d))
    else:
        female_speaker.append(os.path.join(location, file_name, d))

with open(os.path.join(target_location, "speaker.pkl"), "wb") as wf:
    pickle.dump([male_speaker, female_speaker], wf)
