import time
import numpy as np
import torch
from paragraph import Paragraph
from torch.utils.data import Dataset, DataLoader
import pickle
from transformers import AutoTokenizer
from splinter_tokenizer import SplinterTokenizer

PROCESSED_DATA_PATH = "E:/Studies/TAU/NLP/processed"
# PROCESSED_DATA_PATH = "../data/processed"

t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
PROCESSED_DATA_SIZE = 17610994
QUESTION_TOKEN = "<extra_id_0>"
QUESTION_ID = 32099
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
    def __init__(self, num_runs=np.inf, mask=QUESTION_TOKEN):
        super(SplinterDataset, self).__init__()
        self.num_runs = num_runs
        self.mask = mask
        self.all_line_ob = []
        st_time = time.time()
        self.train = self._create_dataset()
        en_time = time.time()
        print("%d Lines were processed in %.2f seconds" % (num_runs, (en_time - st_time)))

    def _create_dataset(self):
        with open(PROCESSED_DATA_PATH, 'r', errors='ignore') as reader:
            count = 0
            st_time = time.time()
            file_idx = 0
            prob_count = 0
            too_many_to_mask = 0
            while count <= self.num_runs:
                count += 1
                line = reader.readline()
                if line:
                    line_instance = Paragraph(line, self.mask)
                    num_rec_spans = line_instance.find_all_recurring_spans()
                    if num_rec_spans == 0: continue
                    max_ngram, num_to_mask = line_instance.sample_ngrams_to_mask()
                    if num_to_mask > 35:
                        continue
                    if max_ngram == 0: continue
                    line_instance.mask_recurring_spans()
                    paragraph_entry = line_instance.get_splinter_data(tokenizer=t5_tokenizer)
                    if paragraph_entry == None:
                        prob_count += 1
                        continue
                    self.all_line_ob.append(paragraph_entry)
                    # TODO: Maybe here it's a good place to add (stochasticly) to train/validation

                    if count % 1000 == 0:
                        ovrl_time = time.time() - st_time
                        time_left = (ovrl_time / count) * (PROCESSED_DATA_SIZE - count)
                        print("%d (%.2f%%) paragraphs were processed at %.2fs (%.2fs per line)" %
                              (count, 100 * count / PROCESSED_DATA_SIZE, ovrl_time, ovrl_time / count))
                        print("     Expected to finish in %.2f minutes" % (time_left / 60))
                        if count % 1000 == 0:
                            with open('all_paragraphs_{}.pkl'.format(file_idx), 'wb+') as out_f:
                                pickle.dump(self.all_line_ob, out_f, pickle.HIGHEST_PROTOCOL)
                else:
                    break
            ovrl_time = time.time() - st_time
            time_left = (ovrl_time / count) * (PROCESSED_DATA_SIZE - count)
            print("%d (%.2f%%) paragraphs were processed at %.2fs (%.2fs per line)" %
                  (count, 100 * count / PROCESSED_DATA_SIZE, ovrl_time, ovrl_time / count))
            print("     Expected to finish in %.2f minutes" % (time_left / 60))

            with open('all_paragraphs_{}.pkl'.format(file_idx), 'wb+') as out_f:
                pickle.dump(self.all_line_ob, out_f, pickle.HIGHEST_PROTOCOL)
            print(prob_count)
            print(too_many_to_mask)

    def select_ngrams(self):
        pass

    def mask_spans_all(self):
        pass


if __name__ == '__main__':
    ds = SplinterDataset(1000)
