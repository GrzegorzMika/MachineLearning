import logging
import os
import pickle

location = os.environ['UNTAR_LOCATION']
file_name = os.environ['UNTAR_FILE']

dirs = os.listdir(location)

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
        male_speaker.append(os.path.join(location, d))
    else:
        female_speaker.append(os.path.join(location, d))

with open(os.path.join(location, "speaker.pkl"), "wb") as wf:
    pickle.dump([male_speaker, female_speaker], wf)
