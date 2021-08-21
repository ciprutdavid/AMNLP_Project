import sentencepiece as spm
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace


SERVER_DATA_PATH = "/home/yandex/AMNLP2021/data/wiki/all"
