import sentencepiece as spm
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace


SERVER_PATH = "/home/yandex/AMNLP2021/data/wiki/all"

class Tokenizer:

    def __init__(self,tokenizer,vocab=None):

        self.wordpiece_tokenizer = 0
        self.sentencepiece_tokenizer = 0
        self.vocab = 0

    def build_vocab(self):
        print()





if __name__ == "__main__":
    sentence = "this is some sentence [UNK]"

    pre_tok = Whitespace()
    print(pre_tok.pre_tokenize_str(sentence))