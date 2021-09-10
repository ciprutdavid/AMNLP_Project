import re
import numpy as np
import nltk
from nltk.corpus import stopwords
import torch
import torch.nn.functional as F
from nltk.util import ngrams
from nltk.tokenize import WordPunctTokenizer, word_tokenize
from nltk.tokenize import TreebankWordTokenizer as twt
import time
import os
from collections import Counter, OrderedDict
from itertools import dropwhile, takewhile
from functools import reduce
from transformers import AutoTokenizer
from splinter_tokenizer import SplinterTokenizer


DATA_PATH = "E:/Studies/TAU/NLP/all"
PROCESSED_DATA_PATH = "E:/Studies/TAU/NLP/processed"
# PROCESSED_DATA_PATH = "../data/processed"
nltk.download('punkt')
STOPWORDS_LIST = stopwords.words('english') + ['-', '"', '(', ')', '[' ,']']
TRAIN_DATA_PATH = "E:/Studies/TAU/NLP/train"
VAL_DATA_PATH  = "E:/Studies/TAU/NLP/test"
VAL_SET_SIZE = 500
MAX_SPAN_LEN = 10
MAX_TOKENS_TO_MASK = 30
MAX_LENGTH = 512

QUESTION_TOKEN = "<extra_id_0>"
QUESTION_ID = 32099

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
    double = [el for el in paragraph_list if paragraph_list.count(el) > 1]
    return len(paragraph_set) != len(paragraph_list)

def strip_punctuation(string):
    return re.sub(r'[.,;:!?]', '', string)

def create_dataset(limit = np.inf):
    with open(PROCESSED_DATA_PATH, 'r') as reader:
        count = 0
        timer_all = {i: 0 for i in range(7)}
        max_histogram = [0] * 11 # debug usage only
        while count < limit:
            line = reader.readline()
            if line:
                line_instance = Paragraph(line)
                timer = line_instance.find_all_recurring_spans()
                for i in range(7):
                    timer_all[i] += timer[i]
                max_ngram = line_instance.sample_ngrams_to_mask()
                max_histogram[max_ngram] += 1
                line_instance.mask_recurring_spans()
                # TODO: Maybe here it's a good place to add (stochasticly) to train/validation
                count += 1
            else: break
        return timer_all, max_histogram

class Paragraph:
    def __init__(self, line, mask):
        self.line = line
        self.tokenizer = WordPunctTokenizer()
        line = self.line.lower()
        self.spans = list(self.tokenizer.span_tokenize(line))
        self.word_list = [line[start:end] for (start, end) in self.spans]
        self.tokens_count = len(self.word_list)
        self.is_masked = [False] * self.tokens_count
        self.mask = mask
        self.mask_len = len(mask)

    def find_all_recurring_spans(self):
        self.ngrams_pos = {}
        for n in range(MAX_SPAN_LEN, 0, -1):

            # interval 0:
            ngram_counter = Counter(nltk.ngrams(self.word_list, n))

            # interval 1:
            ngrams_list = {}
            for k, c in takewhile(lambda v: v[1] > 1, ngram_counter.most_common()):
                ngrams_list[k] = c
            if len(ngrams_list) == 0: continue

            # interval 2:
            ngrams_freq = self.filter_irrelevant_spans(ngrams_list)
            if len(ngrams_freq) == 0: continue

            ngram_pos_n = {}
            for ng in ngrams_freq:
                contained = [True for larger_ng in self.ngrams_pos.keys() if self.is_contained(ng, larger_ng)]
                if len(contained) == 0: ngram_pos_n[ng] = []

            if len(ngram_pos_n) == 0: continue
            # interval 5:
            self.ngrams_pos.update(ngram_pos_n)

        if len(self.ngrams_pos) == 0: return 0
        self.find_ngrams_positions()
        return len(self.ngrams_pos)

    def get_ngrams_positions(self):
        return self.ngrams_pos

    def is_contained(self, ng1, ng2):
        if len(ng1) <= len(ng2):
            for i in range(len(ng2)):
                if ng1 == ng2[i: i + len(ng1)]:
                    return True
        return False

    def filter_irrelevant_spans(self, ngrams_list):
        if len(ngrams_list) == 0: return
        interrupters = ['.', ',', '?', '!']
        bad_suffixes_and_prefixes = ['-', "&", '=', '$', '\'']
        doubles = ['"']
        parenthesis = [('[', ']'), ('(', ')'), ('<', '>'), ('{', '}')]
        filters = [
                   # does not contain an interrupter
                   lambda t: all(ch not in t for ch in interrupters),
                   # at least one word is num
                   lambda t: any(ch.isalnum() for ch in t),
                   # all is ascii
                   lambda t: all(ch.isascii() for ch in t),
                   # not only one words
                   lambda t: any(len(ch) > 1 for ch in t),
                   # does not composed of stopwords only
                   lambda t: any(ch not in STOPWORDS_LIST for ch in t),
                   lambda t: all(t[0] != ch and t[-1] != ch for ch in bad_suffixes_and_prefixes),
                   lambda t: all(t.count(ch) % 2 == 0 for ch in doubles),
                   lambda t: all(t.count(l) == t.count(r) for (l, r) in parenthesis)
                   ]
        valid_keys = list(filter(lambda ng: all(f(ng) for f in filters), ngrams_list.keys()))
        ngrams_freq = Counter()
        for k in valid_keys:
            ngrams_freq[k] = ngrams_list[k]
        return ngrams_freq

    def find_ngrams_positions(self):
        if len(self.ngrams_pos) == 0: return
        num_words = len(self.word_list)
        ngram_pos = list(map(tuple, self.ngrams_pos.keys()))
        # self.ngram_pos_n = {ng: [] for ng in ngram_list}
        ngram_wc = list(map(len, self.ngrams_pos))
        word_list = list(map(lambda w: w.lower(), self.word_list))
        for idx, word in enumerate(word_list):
            for ngram_idx, ng in enumerate(ngram_pos):
                if idx + ngram_wc[ngram_idx] <= num_words and word_list[idx: idx + ngram_wc[ngram_idx]] == list(ng):
                    self.ngrams_pos[ng].append((idx, (self.spans[idx][0], self.spans[idx + ngram_wc[ngram_idx] - 1][1])))

    def sample_ngrams_to_mask(self, p = 1.):
        self.chosen_ngrams = set()
        ngrams_to_mask = []
        spans_to_ngrams = {}
        max_ngram = 0
        for ng, occur_lst in self.ngrams_pos.items():
            if np.random.rand() <= p:
                max_ngram = max(max_ngram, len(ng))
                label_idx = np.random.randint(len(occur_lst))
                # TODO: Consider to change here to indices of ngrams instead of coping
                X = occur_lst[:]
                X.pop(label_idx)
                y = occur_lst[label_idx]
                self.chosen_ngrams.add(ng)
                ngrams_to_mask.append({ng : (X, y)})
                for (w_id, pos) in  X:
                    spans_to_ngrams[pos] = ng
        self.ngrams_to_mask = ngrams_to_mask
        self.spans_to_ngrams = OrderedDict(sorted(spans_to_ngrams.items()))
        return max_ngram, len(self.spans_to_ngrams)

    def mask_recurring_spans(self):
        if len(self.spans_to_ngrams) == 0: return
        limit = min(MAX_TOKENS_TO_MASK, int(self.tokens_count * 0.15))
        masked_line = ""
        last_idx = 0
        num_masked = len(self.spans_to_ngrams.items())
        for (s, e), ng in self.spans_to_ngrams.items():
            masked_line += self.line[last_idx:s] + self.mask
            last_idx = e

        self.masked_line = masked_line + self.line[last_idx:]

    def _get_range_indices(self, l, pattern):
        for i in range(len(l) - len(pattern) + 1):
            if l[i:i + len(pattern)] == pattern:
                return [i, i + len(pattern)]
        return -1

    # for debug only
    def handeler(self, tokenized_rep, str):
        tokenized_ngram = tokenizer(str).input_ids[:-1]
        pos = self._get_range_indices(tokenized_rep, tokenized_ngram)
        if pos != -1: return pos

    # for debug only
    def _handle_edge_cases(self, tokenized_rep, ngram_str):
        pos = self.handeler(tokenized_rep, "\"" + ngram_str + "\"", )
        if pos != -1: return pos
        pos = self.handeler(tokenized_rep, "\"" + ngram_str)
        if pos != -1: return pos
        pos = self.handeler(tokenized_rep, ngram_str + "\"")
        if pos != -1: return pos
        pos = self.handeler(tokenized_rep, "(" + ngram_str + ")")
        if pos != -1: return pos
        pos = self.handeler(tokenized_rep, "(" + ngram_str)
        if pos != -1: return pos
        pos = self.handeler(tokenized_rep, ngram_str + ")")
        return pos

    def _answer_chars_to_tokens(self, text, answer_spans, tokens):
        # convert char indices of answers to token indices
        tok_spans = []
        for s, e in answer_spans:

            answer_tokens = [
                i for i, (s_tok, e_tok) in enumerate(tokens.offset_mapping)
                if e_tok >= s and s_tok <= e
            ]
            if len(answer_tokens) == 0:
                print('Warning!')
                print('text:', text, 'answer_spans:', answer_spans)
                raise Exception
            tok_spans += [(answer_tokens[0], answer_tokens[-1])]


    def get_splinter_data(self, tokenizer):
        out_dict = {'masked_line' : self.masked_line,
                    }


        tokenized_rep = tokenizer(self.masked_line.lower())
        tokenized_line = tokenized_rep.input_ids
        if len(tokenized_line) > MAX_LENGTH:
            print("debug print")
            return None
        debug_counter = 0
        valid_labels = {}
        for ngram in self.ngrams_pos:
            if ngram not in self.chosen_ngrams:
                continue
            debug_counter += len(self.ngrams_pos[ngram])
            ngram_str = ' '.join(ngram)
            tokenized_ngram = tokenizer(ngram_str).input_ids[:-1]
            pos = self._get_range_indices(tokenized_line, tokenized_ngram)
            if pos == -1:
                return None
                # pos = self._handle_edge_cases(tokenized_rep, ngram_str)
                # if pos == -1:
                #     return None, en_time
            valid_labels[ngram] = pos
            # need to save questions that are relevant to this label
        pos_vec = [valid_labels[ng] for ng in self.spans_to_ngrams.values()]
        # st_vec, en_vec = list(zip(*pos_vec))
        out_dict['labels'] = list(zip(*pos_vec))
        #for debug
        if (debug_counter - len(valid_labels.keys()) != len(out_dict['labels'][0])):
            print(1)

        return out_dict





