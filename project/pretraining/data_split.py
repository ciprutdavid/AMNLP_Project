import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import time
import os

DATA_PATH = "E:/Studies/TAU/NLP/all"
PROCESSED_DATA_PATH = "E:/Studies/TAU/NLP/processed"
STOPWORDS_LIST = stopwords.words('english')
nltk.download('punkt')

VAL_SET_SIZE = 500
TRAIN_DATA_PATH = "E:/Studies/TAU/NLP/train"
VAL_DATA_PATH  = "E:/Studies/TAU/NLP/test"


def ends_with_punctuation(string):
    return re.match(".*[?.:;!]$", string) is not None


def unwanted_line(string):
    return string.startswith("<doc") or string.startswith("</doc>") or string == "\n" or not ends_with_punctuation(
        string)


def preprocess_wiki():
    if os.stat(PROCESSED_DATA_PATH).st_size > 0:
        return sum(1 for _ in open(PROCESSED_DATA_PATH, 'r' ,errors="ignore"))
    data = open(DATA_PATH, 'r' ,errors="ignore")
    processed = open(PROCESSED_DATA_PATH, 'a')
    num_lines =0
    for paragraph in data:
        if has_recurring_span(paragraph):
            num_lines += 1
            processed.write(paragraph)
    return num_lines


def data_reader(ob):
    for line in ob:
        if line:
            yield line
        else:
            raise EOFError

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

def get_word_histogram(line, tokenizer):
    raise NotImplementedError

def find_maximal_ngrams(line, hist, tokenizer):
    raise NotImplementedError

def select_masked_tokens(line, ngrams_list):
    # TODO: add [QUESTION] token for n-1 of the occurrences for 1 (or more) repeated words from ngram_list
    raise NotImplementedError

def add_question_tokens():
    data = open(PROCESSED_DATA_PATH, 'r') # Consider to split train and validation 'lazy'ly after this function.
    word_tokenizer = WordPunctTokenizer()
    reader = data_reader(data)
    while True:
        try:
            line = next(reader)
            hist = get_word_histogram(data, word_tokenizer)
            ngrams_list = find_maximal_ngrams(data, hist, word_tokenizer)
            processed_line = select_masked_tokens(line, ngrams_list)
            # TODO: Maybe here it's a good place to add (stochaticly) to train/validation
        except EOFError:
            break


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