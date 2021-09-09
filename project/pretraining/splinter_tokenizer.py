import torch.utils.data
from nltk import WordPunctTokenizer, word_tokenize, TreebankWordTokenizer
from transformers import AutoTokenizer, T5Tokenizer, PreTrainedTokenizer
import time
import t5_baseline_pretrain_dataset as baseline_data


class SplinterTokenizer:

    def __init__(self):
        self.word_tokenizer = TreebankWordTokenizer()
        self.t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')

    def __call__(self, text, padding='max_length', truncation=True, max_length=512, return_tensors='pt'):
        if type(text) == str:
            tokenized_array = self.tokenize(text)
            string = self.t5_tokenizer.convert_tokens_to_string(tokenized_array)
            return self.t5_tokenizer(string, padding='max_length', truncation=truncation, max_length=max_length,
                                     return_tensors=return_tensors)
        else:
            multiple_text = []
            for line in text:
                multiple_text.append(self.t5_tokenizer.convert_tokens_to_string(self.tokenize(line)))
            return self.t5_tokenizer.batch_encode_plus(multiple_text, padding='max_length', truncation=truncation,
                                                       max_length=max_length, return_tensors=return_tensors)

    def tokenize(self, text):
        split_tokens = []
        for token in self.word_punc_tokenizer.tokenize(text):
            for sub_token in self.t5_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

