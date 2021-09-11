import time
import numpy as np
import torch
from paragraph import Paragraph
from torch.utils.data import Dataset, DataLoader
import pickle
from transformers import AutoTokenizer
from splinter_tokenizer import SplinterTokenizer

#PROCESSED_DATA_PATH = "E:/Studies/TAU/NLP/processed"
PROCESSED_DATA_PATH = "/home/yandex/AMNLP2021/benzeharia/project/AMNLP_Project/project/data/processed"

t5_tokenizer = AutoTokenizer.from_pretrained('t5-base', cache_dir='../data/t5_tokenizer_cache/')
PROCESSED_DATA_SIZE = 17610994
QUESTION_TOKEN = "<extra_id_0>"
QUESTION_ID = 32099
VALID_LINES_RATIO = 0.67
VALIDATION_SIZE = 500
P_VALIDATION = VALIDATION_SIZE / PROCESSED_DATA_SIZE * VALID_LINES_RATIO
DIM = 512

class SplinterCollate:
    def __init__(self, tokenizer, device='cuda'):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        input_dict = {'input_ids': torch.LongTensor([], size=(0, DIM)), 'attention_mask': torch.empty(size=(0, DIM)),
                      'labels': {'start_labels': torch.empty(size=(0, DIM)), 'end_labels': torch.empty(size=(0, DIM))}}
        for example_dict in batch:
            tokenized_masked_line = self.tokenizer(example_dict['masked_line'], padding='max_length', truncation=True,
                                                   max_length=512, return_tensors='pt')
            input_dict['input_ids'] = torch.vstack((input_dict['input_ids'], tokenized_masked_line['input_ids']))
            input_dict['attention_mask'] = torch.vstack((input_dict['attention_mask'], tokenized_masked_line['attention_mask']))
            input_dict['labels']['start_labels'] = torch.vstack((input_dict['labels']['start_labels'], example_dict['start_labels']))
            input_dict['labels']['end_labels'] = torch.vstack((input_dict['labels']['end_labels'], example_dict['end_labels']))
        return input_dict


class SplinterDataset(Dataset):
    def __init__(self, num_runs=np.inf, mask=QUESTION_TOKEN, start_idx=0):
        super(SplinterDataset, self).__init__()
        self.num_runs = num_runs
        self.mask = mask
        self.all_line_ob_train = []
        self.all_line_ob_validation = []
        st_time = time.time()
        self.train = self._create_dataset(start_idx=start_idx)
        en_time = time.time()
        self.show_progress_n = 1000
        self.save_checkpoint_n = 10000
        print("%d Lines were processed in %.2f seconds" % (num_runs, (en_time - st_time)))

    def _create_dataset(self, start_idx):
        with open(PROCESSED_DATA_PATH, 'r', errors='ignore') as reader:
            count = 0
            for i in range(start_idx):
                reader.readline()
                count += 1
            st_time = time.time()
            self.train_file_idx = 0
            self.validation_file_idx = 0
            prob_count = 0
            too_many_to_mask = 0
            self.train_indices = []
            self.validation_indices = []
            while count <= self.num_runs:
                count += 1
                if count % 250000 == 0:
                    self.save_train_checkpoint()
                    self.save_validation_checkpoint()
                    self.show_progress(count, st_time)
                line = reader.readline()
                if line:
                    line_instance = Paragraph(line, self.mask)
                    num_rec_spans = line_instance.find_all_recurring_spans()
                    if num_rec_spans == 0:
                        prob_count += 1
                        continue
                    max_ngram, num_to_mask = line_instance.sample_ngrams_to_mask()
                    if max_ngram == 0 or num_to_mask > 35:
                        prob_count += 1
                        continue
                    line_instance.mask_recurring_spans()
                    paragraph_entry = line_instance.get_splinter_data(tokenizer=t5_tokenizer)
                    if paragraph_entry == None:
                        prob_count += 1
                        continue
                    if np.random.rand() > P_VALIDATION:
                        self.train_indices.append(count - 1)
                        self.all_line_ob_train.append(paragraph_entry)
                    else:
                        self.validation_indices.append(count - 1)
                        self.all_line_ob_validation.append(paragraph_entry)

                    # TODO: Maybe here it's a good place to add (stochasticly) to train/validation

                else:
                    break
            self.save_train_checkpoint()
            self.save_validation_checkpoint()
            print(prob_count)
            print(too_many_to_mask)

    def save_train_checkpoint(self):
        with open('../data/splinter_data/train/all_train_paragraphs_{}.pkl'.format(self.train_file_idx), 'wb+') as out_f:
            pickle.dump(self.all_line_ob_train, out_f, pickle.HIGHEST_PROTOCOL)
        with open('../data/new_train_indices/train_indices.pkl', 'wb+') as out_f:
            pickle.dump(self.train_indices, out_f, pickle.HIGHEST_PROTOCOL)
        self.train_file_idx += 1
        self.all_line_ob_train = []

    def save_validation_checkpoint(self):
        with open('../data/splinter_data/validation/all_validation_paragraphs_{}.pkl'.format(self.validation_file_idx), 'wb+') as out_f:
            pickle.dump(self.all_line_ob_validation, out_f, pickle.HIGHEST_PROTOCOL)
        with open('../data/new_val_indices/val_indices.pkl', 'wb+') as out_f:
            pickle.dump(self.validation_indices, out_f, pickle.HIGHEST_PROTOCOL)
        self.validation_file_idx += 1
        self.all_line_ob_validation = []

    def show_progress(self, count, st_time):
        ovrl_time = time.time() - st_time
        time_left = (ovrl_time / count) * (self.num_runs - count)
        print("%d (%.2f%%) paragraphs were processed at %.2fs (%.2fs per line)" %
              (count, 100 * count / self.num_runs, ovrl_time, ovrl_time / count))
        print("     Expected to finish in %.2f minutes" % (time_left / 60))

    def select_ngrams(self):
        pass

    def mask_spans_all(self):
        pass

def prepare_data_for_pretraining():
    paths = {
              'train' : {'first' : lambda x : '../data/splinter_data/train/all_train_paragraphs_{}.pkl'.format(x),
                  'second': lambda x: '../data/splinter_data/train_2/train_outputs/new_all_train_paragraphs_{}.pkl'.format(x)},
              'validation' : {'first' : lambda x : '../data/splinter_data/validation/all_validation_paragraphs_{}.pkl'.format(x),
                  'second': lambda x: '../data/splinter_data/validation_2/validation_outputs/new_all_validation_paragraphs_{}.pkl'.format(x)}
    }
    for data, part in [(a,b) for a in ['train', 'validation'] for b in ['first', 'second']]:
        count = 0
        while True:
            try:
                p = paths[data][part](count)
                with open(p, 'rb+') as f:
                    print(p)
                count+=1
            except FileNotFoundError:
                print(count)
                break


if __name__ == '__main__':
    ds = SplinterDataset(10000000)

