from collections import Counter

import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

from text_scores import METHODS, calculate_score, NEGATIONS, INTERROGATIVES, POWER_WORDS, \
    CASUAL_WORDS, TENTATIVE_WORDS, EMOTION_WORDS, lexical_richness
from utils import log_name

stop_words = set(stopwords.words('english'))


@log_name
def load_and_clean(path, random_state=123):
    data = pd.read_csv(path)
    data['label'] = data['label'].str.replace('fake', 'FAKE', case=False).str.replace('true', 'TRUE', case=False)
    data['source'] = data['source'].str.replace('facebook', 'Facebook')
    data.loc[5]['label'] = 'FAKE'
    data.loc[15]['label'] = 'TRUE'
    data.loc[43]['label'] = 'FAKE'
    data.loc[131]['label'] = 'TRUE'
    data.loc[242]['label'] = 'FAKE'
    data.text.fillna(data.title, inplace=True)
    data.title.fillna('missing', inplace=True)
    data.source.fillna('missing', inplace=True)
    return data.sample(frac=1, random_state=random_state).reset_index(drop=True)


@log_name
def capital_letters_in_text(data):
    data['text_num_uppercase'] = data['text'].str.count(r'[A-Z]')
    data['text_len'] = data['text'].str.len()
    data['text_pct_uppercase'] = data.text_num_uppercase.div(data.text_len)
    return data


@log_name
def capital_letters_in_title(data):
    data['title_num_uppercase'] = data['title'].str.count(r'[A-Z]')
    data['title_len'] = data['title'].str.len()
    data['title_pct_uppercase'] = data.title_num_uppercase.div(data.title_len)
    return data


@log_name
def stop_words_text(data):
    data['text_num_stop_words'] = data['text'].str.split().apply(lambda x: len(set(x).intersection(stop_words)))
    data['text_word_count'] = data['text'].apply(lambda x: len(str(x).split()))
    data['text_pct_stop_words'] = data['text_num_stop_words'] / data['text_word_count']
    return data


@log_name
def stop_words_title(data):
    data['title_num_stop_words'] = data['title'].str.split().apply(lambda x: len(set(x).intersection(stop_words)))
    data['title_word_count'] = data['title'].apply(lambda x: len(str(x).split()))
    data['title_pct_stop_words'] = data['title_num_stop_words'] / data['title_word_count']
    return data


@log_name
def part_of_speech(data):
    token = data.apply(lambda row: word_tokenize(row['title']), axis=1)
    token = token.apply(lambda row: pos_tag(row))
    tag_count_df = pd.DataFrame(token.map(lambda x: Counter(tag[1] for tag in x)).to_list()).fillna(0)
    data = pd.concat([data, tag_count_df], axis=1)
    return data


@log_name
def negations_and_interrogatives(data):
    data['num_negation'] = data['text'].str.lower().str.count(NEGATIONS)
    data['num_interrogatives_title'] = data['title'].str.lower().str.count(INTERROGATIVES)
    data['num_interrogatives_text'] = data['text'].str.lower().str.count(INTERROGATIVES)
    return data


@log_name
def specific_words(data):
    data['num_powerWords_text'] = data['text'].str.lower().str.count(POWER_WORDS)
    data['num_casualWords_text'] = data['text'].str.lower().str.count(CASUAL_WORDS)
    data['num_tentativeWords_text'] = data['text'].str.lower().str.count(TENTATIVE_WORDS)
    data['num_emotionWords_text'] = data['text'].str.lower().str.count(EMOTION_WORDS)
    return data


@log_name
def text_scores(data, methods=METHODS.keys()):
    for m in methods:
        data[m] = calculate_score(data, m)
    return data


@log_name
def preprocess_data(data):
    data = capital_letters_in_title(data)
    data = capital_letters_in_text(data)
    data = stop_words_title(data)
    data = stop_words_text(data)
    data = part_of_speech(data)
    data = negations_and_interrogatives(data)
    data = specific_words(data)
    data = text_scores(data)
    data = lexical_richness(data)
    return data
