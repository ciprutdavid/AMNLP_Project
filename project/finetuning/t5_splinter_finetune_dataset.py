import torch.utils.data
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
import time
import numpy as np

DATA_PATH = "../../data/splinter_data/squad"
SEED = [42]
EXAMPLES = [16, 32, 64, 128, 256, 512, 1024]
train_file_name = lambda seed, examples: f"squad-train-seed-{seed}-num-examples-{examples}.jsonl"
DEV_FILE_NAME = "dev.jsonl"

def find_start_end_indices(context_ids,answer_ids):
    context_ids = context_ids[:-1] # remove id 1 from the end which the tokenizer adds
    answer_ids = answer_ids[:-1] # remove id 1 from the end which the tokenizer adds
    for i in range(len(context_ids) - len(answer_ids) + 1):
        if context_ids[i:i+len(answer_ids)] == answer_ids:
            start_index = i
            end_index = i + len(answer_ids) - 1
            return start_index,end_index
    return None


def create_squad_train(seed, examples, tokenizer=AutoTokenizer.from_pretrained('t5-base')):
    data = []
    labels = []
    if seed not in SEED:
        raise Exception("seed needs to be 42,43,44,45 or 46")
    if examples not in EXAMPLES:
        raise Exception("examples needs to be 16,32,64,128,256,512 or 1024")
    path = DATA_PATH + '/' + train_file_name(seed, examples)
    with open(path, 'r') as file:
        for item in file:
            item_dict = json.loads(item)
            if 'context' not in item_dict:
                continue
            else:
                if len(tokenizer(item_dict['context'])['input_ids']) > 512: continue
                data.append(item_dict['context'] + " </s> " + item_dict['qas'][0]['question'] + " " + "<extra_id_0>")
                labels.append("<extra_id_0> " + item_dict['qas'][0]['answers'][0] + " </s>")
    return data, labels


def create_squad_val(size=1000, tokenizer=AutoTokenizer.from_pretrained('t5-base')):  # TODO : finish val data creation
    data = []
    labels = []
    curr_size = 0
    path = DATA_PATH + '/' + DEV_FILE_NAME
    with open(path, 'r') as file:
        for idx, item in enumerate(file):
            if curr_size == size:
                break
            else:
                item_dict = json.loads(item)
                if idx == 0 or len(tokenizer(item_dict['context'])['input_ids']) > 512: continue
                curr_size += 1
                data.append(item_dict['context'] + " " + item_dict['qas'][0]['question'] + " </s> " + "<extra_id_0>")
                labels.append("<extra_id_0> " + item_dict['qas'][0]['answers'][0] + " </s>")
    return data, labels


class SquaDataset(Dataset):
    def __init__(self, examples, labels):
        super(SquaDataset, self).__init__()
        self.examples = examples
        self.labels = labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item], self.labels[item]


class SquaDataColate:  # TODO : finish data colate

    def __init__(self, tokenizer,device='cuda'):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        X,y = map(list,zip(*batch))
        tokenized_X = self.tokenizer.batch_encode_plus(X, padding='max_length', truncation=True, max_length=512,
                                                       return_tensors='pt')
        tokenized_y = self.tokenizer.batch_encode_plus(y, padding='max_length', truncation=True, max_length=512,
                                                       return_tensors='pt')
        arg_dict = {
            'input_ids': tokenized_X['input_ids'].to(self.device),'labels': tokenized_y['input_ids'].to(self.device)
        }
        return arg_dict


if __name__ == "__main__":

    from transformers import T5Config,T5ForConditionalGeneration,AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration(T5Config())

    path = DATA_PATH + '/' + train_file_name(42, 16)
    with open(path, 'r') as file:
        for idx,item in enumerate(file):
            if idx == 0 : continue
            item_dict = json.loads(item)
            qas = item_dict['qas'][0]
            answer = qas['answers'][0]
            tokenized_answer = tokenizer(answer).input_ids
