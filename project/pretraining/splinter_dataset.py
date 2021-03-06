import time
import numpy as np
import torch
from paragraph import Paragraph
from torch.utils.data import Dataset
import pickle
from transformers import AutoTokenizer

PROCESSED_DATA_PATH = "project/data/processed"

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
        masked_line_batch, st_labels, en_labels = list(zip(*batch))
        masked_line_batch = self.tokenizer(list(masked_line_batch), padding='max_length', truncation=True, max_length=DIM,
                                                       return_tensors='pt')

        labels_batch = torch.cat((torch.cat(st_labels),torch.cat(en_labels)))
        masked_line_batch['input_ids'] =  masked_line_batch['input_ids'].to(self.device)
        masked_line_batch['attention_mask'] = masked_line_batch['attention_mask'].to(self.device)
        masked_line_batch['labels'] = labels_batch.to(self.device)

        return masked_line_batch

class SplinterDatasetWrapper(Dataset):
    def __init__(self, data_type):
        super(SplinterDatasetWrapper, self).__init__()
        self.data = prepare_data_for_pretraining(data_type)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        masked_line= self.data[item]['masked_line']
        raw_st_labels, raw_en_labels= self.data[item]['labels']
        st_labels = torch.LongTensor(raw_st_labels)
        en_labels = torch.LongTensor(raw_en_labels)
        return (masked_line, st_labels, en_labels)

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
        self.t5_tokenizer = AutoTokenizer.from_pretrained('t5-base', cache_dir='../data/t5_tokenizer_cache/')
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
                    paragraph_entry = line_instance.get_splinter_data(tokenizer= self.t5_tokenizer)
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

def prepare_data_for_pretraining(data_type = 'train'):
    paths = {
              'train' : {'first' : lambda x : '../data/splinter_data/train/all_train_paragraphs_{}.pkl'.format(x),
                  'second': lambda x: '../data/splinter_data/train_2/train_outputs/new_all_train_paragraphs_{}.pkl'.format(x)},
              'validation' : {'first' : lambda x : '../data/splinter_data/validation/all_validation_paragraphs_{}.pkl'.format(x),
                  'second': lambda x: '../data/splinter_data/validation_2/validation_outputs/new_all_validation_paragraphs_{}.pkl'.format(x)}
    }
    # for data, part in [(a,b) for a in ['train', 'validation'] for b in ['first', 'second']]:
    output = []
    for part in ['first', 'second']:
        count = 0
        while True:
            try:
                p = paths[data_type][part](count)
                with open(p, 'rb+') as f:
                    output.extend(pickle.load(f))
                count+=1
            except FileNotFoundError:
                print(count)
                break
    return output

