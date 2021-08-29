import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import WordPunctTokenizer
import time
import os
from collections import Counter, OrderedDict
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
MAX_TOKENS_TO_MASK = 30

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


def create_dataset(limit = np.inf):
    data = open(PROCESSED_DATA_PATH, 'r') # Consider to split train and validation 'lazy'ly after this function.
    word_tokenizer = WordPunctTokenizer()
    with open(PROCESSED_DATA_PATH, 'r') as reader:
        count = 0
        # all_ngrams = set() # debug usage
        while count < limit:
            line = reader.readline()
            if line:
                line_instance = Paragraph(line)
                line_instance.find_all_recurring_spans()
                # print(line_instance.get_ngrams_positions())
                # all_ngrams.update(rec_spans.keys()) # debug_usage
                # processed_line = mask_tokens_stochastically(line, rec_spans)
                # line_instance.sample_ngrams_to_mask()
                # line_instance.mask_recurring_spans()

                # TODO: Maybe here it's a good place to add (stochaticly) to train/validation
                count += 1
            else: break

class Paragraph:
    def __init__(self, line, mask = "[QUESTION]"):
        self.line = line
        self.tokenizer = WordPunctTokenizer()
        line = self.line.lower()
        self.spans = list(self.tokenizer.span_tokenize(line))
        self.word_list = [line[start:end] for (start, end) in self.spans]
        self.tokens_count = len(self.word_list)
        self.is_masked = [0] * self.tokens_count
        self.mask = mask
        self.mask_len = len(mask)

    def find_all_recurring_spans(self):
        ngrams_pos = {}
        for n in range(MAX_SPAN_LEN, 0, -1):
            ngrams_list = nltk.ngrams(self.word_list, n)
            ngram_freq = self.filter_irrelevant_spans(ngrams_list)
            for k, _ in dropwhile(lambda v: v[1] > 1, ngram_freq.most_common()):
                del ngram_freq[k]
            ngrams_pos_n = self.find_ngrams_positions(ngram_freq)
            for ng in ngrams_pos_n:
                ngrams_pos_n[ng] = self.remove_duplicate_ngrams(ng, ngrams_pos_n[ng])
            ngrams_pos.update({ng: l for ng, l in ngrams_pos_n.items() if len(l) > 1})
        self.ngram_pos = ngrams_pos

    def get_ngrams_positions(self):
        return self.ngram_pos

    def filter_irrelevant_spans(self, ngrams_list):
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
                   lambda t: all(t.count(l) == t.count(r) for (l, r) in parenthesis)
                   ]
        ngrams_list = list(filter(lambda ng: all(f(ng) for f in filters), ngrams_list))
        return Counter(ngrams_list)

    def find_ngrams_positions(self, ngrams_counter):
        num_words = len(self.word_list)
        ngram_list = list(map(tuple, ngrams_counter.keys()))
        output = {ng: [] for ng in ngram_list}
        ngram_wc = list(map(len, ngram_list))
        word_list = list(map(lambda w: w.lower(), self.word_list))
        for idx, word in enumerate(word_list):
            for ngram_idx, ng in enumerate(ngram_list):
                if idx + ngram_wc[ngram_idx] <= num_words and word_list[idx: idx + ngram_wc[ngram_idx]] == list(ng):
                    output[ng].append((idx, (self.spans[idx][0], self.spans[idx + ngram_wc[ngram_idx] - 1][1])))
        assert all([len(output[key]) == ngrams_counter[key] for key in output])
        return output

    def remove_duplicate_ngrams(self, ng, all_positions):
        word_num = len(ng)
        unique = []
        last_idx = -1
        for (word_id, (st, end)) in all_positions:
            if not any(self.is_masked[word_id:(word_id + word_num)]) and (last_idx < 0 or word_id - last_idx >= word_num):
                last_idx = word_id
                unique.append((word_id, (st, end)))

        if len(unique) >= 2:
            for idx, pos in unique:
                self.is_masked[idx:(idx + word_num)] = [True] * word_num
        else:
            unique = []
        return unique

    def sample_ngrams_to_mask(self, p = 0.67):
        ngrams_to_mask = []
        spans_to_ngrams = {}
        while len(ngrams_to_mask) == 0:
            for ng, occur_lst in self.ngram_pos.items():
                if np.random.rand() <= p and ng not in ngrams_to_mask:
                    label_idx = np.random.randint(len(occur_lst))
                    # TODO: Consider to change here to indices of ngrams instead of coping
                    X = occur_lst[:]
                    X.pop(label_idx)
                    y = occur_lst[label_idx]
                    ngrams_to_mask.append({ng : (X, y)})
                    for (w_id, pos) in  X:
                        spans_to_ngrams[pos] = ng
        self.ngrams_to_mask = ngrams_to_mask
        self.spans_to_ngrams = OrderedDict(sorted(spans_to_ngrams.items()))

    def mask_recurring_spans(self):
        limit = min(MAX_TOKENS_TO_MASK, int(self.tokens_count * 0.15))
        masked_line = self.line[:]
        offset = 0
        for (s, e), ng in self.spans_to_ngrams.items():
            span_length = e - s


if __name__ == "__main__":
    num_runs = 10
    st_time = time.time()
    all_ngrams = create_dataset(num_runs)
    en_time = time.time()
    print("%d Lines were processed in %.2f seconds" % (num_runs, (en_time - st_time)))