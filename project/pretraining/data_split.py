import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import WordPunctTokenizer
import time
import os
from collections import Counter
from itertools import dropwhile
from functools import reduce

DATA_PATH = "E:/Studies/TAU/NLP/all"
PROCESSED_DATA_PATH = "E:/Studies/TAU/NLP/processed"
STOPWORDS_LIST = stopwords.words('english') + ['-', '"', '(', ')', '[' ,']']
nltk.download('punkt')
TRAIN_DATA_PATH = "E:/Studies/TAU/NLP/train"
VAL_DATA_PATH  = "E:/Studies/TAU/NLP/test"
VAL_SET_SIZE = 500
MAX_SPAN_LEN = 10

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
    try:
        for line in ob:
            if line:
                yield line
    except StopIteration:
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

def find_ngrams_positions(word_list, spans, ngrams_counter):
    num_words = len(word_list)
    ngram_list = list(map(tuple, ngrams_counter.keys()))
    output = {ng : [] for ng in ngram_list}
    ngram_wc = list(map(len, ngram_list))
    word_list = list(map(lambda w : w.lower(), word_list))
    for idx, word in enumerate(word_list):
        for ngram_idx, ng in enumerate(ngram_list):
            if idx + ngram_wc[ngram_idx] <= num_words and word_list[idx: idx + ngram_wc[ngram_idx]] == list(ng):
                output[ng].append((idx, (spans[idx][0], spans[idx + ngram_wc[ngram_idx] - 1][1])))
    assert all([len(output[key]) == ngrams_counter[key] for key in output])
    return output

def remove_duplicate_ngrams(ng, all_positions, is_masked):
    word_num = len(ng)
    unique = []
    last_idx = -1
    for (word_id, (st, end)) in all_positions:
        if not any(is_masked[word_id:(word_id + word_num)]) and (last_idx < 0 or word_id - last_idx >= word_num):
            last_idx = word_id
            unique.append((word_id, (st, end)))

    if len(unique) >= 2:
        for idx, pos in unique:
            is_masked[idx:(idx + word_num)] = [True] * word_num
    else:
        unique = []
    return unique

def filter_irrelevant_spans(ngrams_list):
    interrupters = ['.', ',', '?', '!']
    bad_suffixes_and_prefixes = ['-', "&", '=', '$', '\'']
    doubles = ['"']
    parenthesis = [('[', ']'), ('(', ')'), ('<', '>'), ('{', '}')]
    filters = [lambda t: all(ch not in t for ch in interrupters),
               lambda t: any(ch.isalnum() for ch in t),
               lambda t: all(ch.isascii() for ch in t),
               lambda t: any(len(ch) > 1 for ch in t),
               lambda t: any(ch not in STOPWORDS_LIST for ch in t),
               lambda t: all(t[0] != ch and t[-1] != ch for ch in bad_suffixes_and_prefixes),
               lambda t: all(t.count(ch) % 2 == 0 for ch in doubles),
               lambda t: all(t.count(l) == t.count(r) for (l,r) in parenthesis)
               ]
    ngrams_list = list(filter(lambda ng : all(f(ng) for f in filters), ngrams_list))
    return Counter(ngrams_list)

def find_all_recurring_spans(line, tokenizer):
    words_count = int(len(line.split()) * 0.15)
    line = line.lower()
    spans = list(tokenizer.span_tokenize(line))
    word_list = [line[start:end]for (start,end) in spans]
    ngrams_pos = {}
    is_masked = [False] * len(word_list)
    for n in range(MAX_SPAN_LEN, 0, -1):
        ngrams = nltk.ngrams(word_list, n)
        ngram_freq = filter_irrelevant_spans(ngrams)
        for k,_ in dropwhile(lambda v : v[1] > 1 , ngram_freq.most_common()):
            del ngram_freq[k]
        ngrams_pos_n = find_ngrams_positions(word_list, spans, ngram_freq)
        for ng in ngrams_pos_n:
            ngrams_pos_n[ng] = remove_duplicate_ngrams(ng, ngrams_pos_n[ng], is_masked)
        ngrams_pos.update({ng : l for ng, l  in ngrams_pos_n.items() if len(l) > 1})
    return ngrams_pos

def mask_passage(line, ngram_positions):
    answer_positions = np.random.randint(0,len(ngram_positions))
    for idx,position in enumerate(ngram_positions):
        if idx == answer_positions: continue
        line = line[:position[0]] + "[QUESTION]" + line[position[1]:]
    return line

def select_masked_tokens(line, ngrams_list):
    # TODO: add [QUESTION] token for n-1 of the occurrences for 1 (or more) repeated words from ngram_list
    raise NotImplementedError

def add_question_tokens(limit = np.inf):
    data = open(PROCESSED_DATA_PATH, 'r') # Consider to split train and validation 'lazy'ly after this function.
    word_tokenizer = WordPunctTokenizer()
    reader = data_reader(data)
    count = 0
    all_ngrams = set()
    while count < limit:
        try:
            line = next(reader)
            rec_spans = find_all_recurring_spans(line, word_tokenizer)
            all_ngrams.update(rec_spans.keys())
            # processed_line = select_masked_tokens(line, ngrams_list)
            # TODO: Maybe here it's a good place to add (stochaticly) to train/validation
            count += 1
        except EOFError:
            break
    return all_ngrams

def has_recurring_span(paragraph):
    if unwanted_line(paragraph):
        return False
    paragraph_list = [word for word in strip_punctuation(paragraph).lower().split() if word not in STOPWORDS_LIST]
    paragraph_set = set(paragraph_list)
    return len(paragraph_set) != len(paragraph_list)

def strip_punctuation(string):
    return re.sub(r'[.,;:!?]', '', string)

if __name__ == "__main__":
    with open(PROCESSED_DATA_PATH, 'r') as data:
        all_ngrams = add_question_tokens(100)
        for s in all_ngrams:
            print(s)