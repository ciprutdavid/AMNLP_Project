from torch.utils.data import Dataset, DataLoader
import numpy as np

def load_data(load_path):
    lines = []
    with open(load_path,'r') as f:
        for line in f:
            lines.append(line)
    return lines

class WikiDataset(Dataset):
    def __init__(self, train):
        super(WikiDataset, self).__init__()
        self.train = train

    def __len__(self):
        return len(self.train)

    def __getitem__(self, item):
        return self.train[item]


class T5_Collate(object):
    def __init__(self, tokenizer,device='cuda'):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        X = []
        y = []
        for idx in range(len(batch)):
            masked, mask = self.mask_span(batch[idx])
            X.append(masked)
            y.append(mask)
        tokenized_X = self.tokenizer(X, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        tokenized_y = self.tokenizer(y, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        arg_dict = {
            'input_ids' : tokenized_X['input_ids'].to(self.device),
            # 'attention_mask' : tokenized_X['attention_mask'].to(self.device),
            'labels' : tokenized_y['input_ids'].to(self.device)
            # 'decoder_attention_mask' : tokenized_y['attention_mask'].to(self.device)
        }
        return arg_dict

    def mask_span(self, paragraph):
        mask = ""
        split = paragraph.split()
        cursor = 0
        id = 0
        while cursor < len(split):
            if np.random.binomial(1, 0.3) == 1:
                span_length = np.random.geometric(0.4)
                id_token = self.extra_id(id) + " "
                id += 1
                mask += id_token
                if span_length >= len(split) - cursor:
                    mask += " ".join(split[cursor:]) + " "
                    split[cursor:] = [id_token]
                    cursor += span_length + 1
                else:
                    mask += " ".join(split[cursor:cursor + span_length]) + " "
                    split[cursor:cursor + span_length] = [id_token]
                    cursor += span_length + 1
            else:
                cursor += 1
        return " ".join(split), mask + "</s>"

    def extra_id(self, id):
        return f"<extra_id_{id}>"


def WikiDataloader(dataset, collate_fn, batch_size=100, shuffle=False):
    return DataLoader(WikiDataset(dataset), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
