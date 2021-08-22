import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer



SERVER_DATA_PATH = "/home/yandex/AMNLP2021/data/wiki/all"
SMALL_DATA = "/home/david/PycharmProjects/AMNLP_Project/project/pretraining/small_data"


# sp_trainer_config ={
#     'input':SMALL_DATA,
#     'model_prefix':'m',
#     'train_extremely_large_corpus':'true'
# }
#
# # f"--input={SMALL_DATA} --model_prefix=m"
# spm.SentencePieceTrainer.train(sp_trainer_config)
# sp = spm.SentencePieceProcessor()
# sp.load('m.model')
# print(sp.get_piece_size())


bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
bert_tokenizer.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(
    vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]","[QUESTION]"]
)
files = [SMALL_DATA]
bert_tokenizer.train(files, trainer)

bert_tokenizer.save("data/bert-all.json")
