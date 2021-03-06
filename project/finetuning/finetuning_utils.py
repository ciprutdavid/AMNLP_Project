import torch.utils.data
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
from transformers import Trainer
import torch.nn.functional as F

DATA_PATH = "data/splinter_data/squad"
SEED = [42,43,44,45,46]
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
    start_labels = []
    end_labels = []
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
                masked_context = item_dict['context'] + " </s> " + item_dict['qas'][0]['question'] + " " + "<extra_id_0>"
                if len(tokenizer(masked_context)['input_ids']) > 512: continue
                labels = find_start_end_indices(tokenizer(item_dict['context'])['input_ids'],
                                                    tokenizer(item_dict['qas'][0]['answers'][0])['input_ids'])
                if labels is None: continue
                start_labels.append(labels[0])
                end_labels.append(labels[1])
                data.append(masked_context)
    return data, start_labels, end_labels


def create_squad_val(size=1000, tokenizer=AutoTokenizer.from_pretrained('t5-base')):  # TODO : finish val data creation
    data = []
    start_labels = []
    end_labels = []
    curr_size = 0
    path = DATA_PATH + '/' + DEV_FILE_NAME
    with open(path, 'r') as file:
        for idx, item in enumerate(file):
            if curr_size == size:
                break
            else:
                item_dict = json.loads(item)
                if 'context' not in item_dict: continue
                masked_context = item_dict['context'] + " " + item_dict['qas'][0]['question'] + " </s> " + "<extra_id_0>"
                if idx == 0 or len(tokenizer(masked_context)['input_ids']) > 512: continue
                labels = find_start_end_indices(tokenizer(item_dict['context'])['input_ids'],
                                                    tokenizer(item_dict['qas'][0]['answers'][0])['input_ids'])
                if labels is None: continue
                start_labels.append(labels[0])
                end_labels.append(labels[1])
                data.append(masked_context)
                curr_size += 1
    return data, start_labels, end_labels


class SquaDataset(Dataset):
    def __init__(self, examples, start_labels,end_labels):
        super(SquaDataset, self).__init__()
        self.examples = examples
        self.start_labels = start_labels
        self.end_labels = end_labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item], self.start_labels[item], self.end_labels[item]


class SquaDataColate:

    def __init__(self, tokenizer,device='cuda'):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        examples,start_labels,end_labels = map(list,zip(*batch))
        tokenized_X = self.tokenizer.batch_encode_plus(examples, padding='max_length', truncation=True, max_length=512,
                                                       return_tensors='pt')
        labels = torch.LongTensor(start_labels + end_labels)
        arg_dict = {
            'input_ids': tokenized_X['input_ids'].to(self.device),'labels': labels.to(self.device)
        }
        return arg_dict


class QASS_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs, labels)
        return (loss, outputs) if return_outputs else loss

