from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

SERVER_PATH = "/home/yandex/AMNLP2021/davidciprut/AMNLP_Project/data/wiki/0"
LOCAL_PATH = "/home/david/PycharmProjects/AMNLP_Project/data/wiki/0"


class WikiDataset(Dataset):
    def __init__(self, train):
        super(WikiDataset, self).__init__()
        self.train = train

    def __len__(self):
        return len(self.train)

    def __getitem__(self, item):
        return self.train[item]
        # masked_paragraph, mask_labels = self.Mask(paragraph)
        # tokenized_input = self.tokenizer(masked_paragraph,return_special_tokens_mask=True, padding='max_length', truncation=True, return_tensors='pt',
        #                                  max_length=512).to(self.device)
        # tokenized_labels = self.tokenizer(mask_labels, return_special_tokens_mask=True,padding='max_length', truncation=True, return_tensors='pt',
        #                                   max_length=512).to(self.device)
        # return tokenized_input, tokenized_labels


class T5_Collate(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        masked = [self.mask_span(batch[i]) for i in range(len(batch))]
        return self.tokenizer(masked,padding='max_length', truncation=True,max_length=512, return_tensors='pt')

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


if __name__ == "__main__":
    train_data = []
    with open(LOCAL_PATH,'r') as f:
        for line in f:
            train_data.append(line)

    from transformers import T5Config, T5Model, T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast, AutoTokenizer, \
        AutoModel
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms import Compose
    import time



    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    dataset = WikiDataset(train_data)
    collator = T5_Collate(tokenizer)
    train_dl = WikiDataloader(train_data, collate_fn=collator, batch_size=64, shuffle=False)
    a = next(iter(train_dl))
    print()
