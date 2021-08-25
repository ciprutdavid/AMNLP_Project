import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import numpy as np
import time

DATA_PATH = "./data/wiki/all"
PROCESSED_DATA_PATH = "./data/wiki/processed"
STOPWORDS_LIST = stopwords.words('english')
VAL_SET_SIZE = 500
TRAIN_DATA_PATH = "./data/wiki/train"
VAL_DATA_PATH  = "./data/wiki/test"


def ends_with_punctuation(string):
    return re.match(".*[?.:;!]$", string) is not None


def unwanted_line(string):
    return string.startswith("<doc") or string.startswith("</doc>") or string == "\n" or not ends_with_punctuation(
        string)


def preprocess_wiki():
    data = open(DATA_PATH, 'r')
    processed = open(PROCESSED_DATA_PATH, 'a')
    num_lines =0
    for paragraph in data:
        if has_recurring_span(paragraph):
            num_lines += 1
            processed.write(paragraph)
    return num_lines

def split_train_validation():
    num_paragraphs = preprocess_wiki()
    VALIDATION_DOC_ID = np.random.choice(num_paragraphs, VAL_SET_SIZE, replace=False)
    data = open(PROCESSED_DATA_PATH, 'r')
    train = open(TRAIN_DATA_PATH, 'a')
    val = open(VAL_DATA_PATH, 'a')
    content = data.readlines()
    for i,line in enumerate(content):
        if i in VALIDATION_DOC_ID:
            val.write(line)
        else:
            train.write(line)


def has_recurring_span(paragraph):
    if unwanted_line(paragraph):
        return False
    paragraph_list = [word for word in strip_punctuation(paragraph).lower().split() if word not in STOPWORDS_LIST]
    paragraph_set = set(paragraph_list)
    return len(paragraph_set) != len(paragraph_list)


def strip_punctuation(string):
    return re.sub(r'[.,;:!?]', '', string)


if __name__ == "__main__":
    split_train_validation()