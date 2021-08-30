from transformers import T5Config, T5Model, T5Tokenizer
import numpy as np
import torch

config = {
    "train_path": "",
    "val_path": "",
    "tokenizer": T5Tokenizer.from_pretrained("t5-base")
}

class T5PretrainingDataset:
    def __init__(self, train_path, val_path, tokenizer):
        self.train_path = train_path
        self.val_path = val_path
        self.tokenizer = tokenizer
        self.train = open(self.train_path, 'r')
        self.val = open(self.val_path, 'r')

    def get_batch(self,size,type="train",device='cuda'):
        raw_batch_X, raw_batch_y = self.get_raw_batch(size,type)
        X = self.tokenizer(raw_batch_X,padding=True, return_tensors='pt')
        y = self.tokenizer(raw_batch_y,padding=True, return_tensors='pt')
        return X,y

    def get_raw_batch(self, size, type="train"):
        masked_examples = []
        mask_labels = []
        file_ = self.train if type == "train" else self.val
        examples = []
        example_num = 0
        while example_num < size:
            try:
                examples.append(file_.readline())
                example_num += 1
            except EOFError:
                file_.close()
                self.train = open(self.train_path) if type == "train" else open(self.val_path)
                break
        for example in examples:
            masked_example, mask_label = self.mask_paragraph(example)
            masked_examples.append(masked_example)
            mask_labels.append(mask_label)
        return masked_examples, mask_labels

    def extra_id(self, id):
        return f"<extra_id_{id}>"

    def mask_paragraph(self, paragraph):
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
            self.mask_paragraph(paragraph)
        else:
            return " ".join(split), mask + "</s>"
