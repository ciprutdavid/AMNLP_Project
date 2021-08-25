import transformers
import transformers.models.t5.tokenization_t5 as T5_Tokenizer
from transformers import AutoTokenizer

def get_t5_tokenizer():
    return AutoTokenizer.from_pretrained('t5-base')
