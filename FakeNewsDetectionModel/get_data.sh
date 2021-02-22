#!/bin/bash

wget https://raw.githubusercontent.com/susanli2016/NLP-with-Python/master/data/corona_fake.csv
python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords'); nltk.download('punkt')"