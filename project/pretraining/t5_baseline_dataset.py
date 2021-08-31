from transformers import T5Config, T5Model, T5Tokenizer, T5TokenizerFast
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import time

SERVER_PATH = "/home/yandex/AMNLP2021/davidciprut/AMNLP_Project/data/wiki/0"
LOCAL_PATH = "/home/david/PycharmProjects/AMNLP_Project/data/wiki/0"


class WikiDataset(Dataset):
    def __init__(self, train, tokenizer, device):
        self.train = train
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.train)

    def __getitem__(self, item):
        paragraph = self.train[item]
        # start = time.time()
        masked_paragraph, mask_labels = self.Mask(paragraph)
        end = time.time()
        # print("Masking : " + str(end-start))
        # start = time.time()
        tokenized_input = self.tokenizer(masked_paragraph, padding='max_length', truncation=True, return_tensors='pt',
                                         max_length=512).to(self.device)
        tokenized_labels = self.tokenizer(mask_labels, padding='max_length', truncation=True, return_tensors='pt',
                                          max_length=512).to(self.device)
        # end = time.time()
        # print("Tokenization : " + str(end-start))
        return tokenized_input, tokenized_labels

    def Mask(self, paragraph):
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
        if mask == "</s>":
            self.Mask(paragraph)
        else:
            return  " ".join(split),  mask + "</s>"

    def extra_id(self,id):
        return f"<extra_id_{id}>"


def WikiDataloader(data, tokenizer, device='cuda', batch_size=100,num_workers=0):
    return DataLoader(WikiDataset(data, tokenizer, device=device), batch_size=batch_size,num_workers=num_workers)


# if __name__ == "__main__":
#
#
#     data_list = []
#     idx = 0
#     with open(LOCAL_PATH, 'r') as f:
#         for line in f:
#             data_list.append(line)
#             idx += 1
#             if idx == 10000:
#                 break
#
#
#     config = {
#         "train_data": data_list,
#         "val_data": data_list,
#         "tokenizer": T5TokenizerFast.from_pretrained("t5-base")
#     }
#     batch_size=1000
#     start = time.time()
#     dl = WikiDataloader(data=data_list, tokenizer=T5TokenizerFast.from_pretrained("t5-base"), device='cpu',
#                         batch_size=batch_size,num_workers=0)
#     end = time.time()
#     print("Dataloader Initialization : " + str(end - start))
#
#     start = time.time()
#     out=next(iter(dl))
#     end = time.time()
#     print(f"Getting Batch data of size {batch_size} : " + str(end-start))
