import time
import numpy as np
import torch
from paragraph import Paragraph
from torch.utils.data import Dataset, DataLoader
import pickle

PROCESSED_DATA_PATH = "E:/Studies/TAU/NLP/processed"
# PROCESSED_DATA_PATH = "../data/processed"

PROCESSED_DATA_SIZE = 17610994

class SplinterCollate:
    def __init__(self, tokenizer, device='cuda'):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        X = []
        y = []
        # for b in batch:
        #     #line_mask = self.get_masked_line(b)
        #     X.append(line_mask)
        #     y.append(line_mask.label)
        tokenized_X = self.tokenizer.batch_encode_plus(X, padding='max_length', truncation=True, max_length=512,
                                                       return_tensors='pt')
        tokenized_y = self.tokenizer.batch_encode_plus(y, padding='max_length', truncation=True, max_length=512,
                                                       return_tensors='pt')

        arg_dict = {
            'input_ids': tokenized_X['input_ids'].to(self.device),
            # 'decoder_input_ids': tokenized_y['input_ids'].to(self.device),
            'labels':tokenized_y['input_ids'].to(self.device)
        }
        return arg_dict

class SplinterDataset(Dataset):
    def __init__(self, num_runs = np.inf, mask = "[QUESTION]"):
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
            # timer_all = {i: 0 for i in range(7)} # debug usage only
            # max_histogram = [0] * 11  # debug usage only
            st_time = time.time()
            file_idx = 0
            while count < self.num_runs:
                line = reader.readline()
                if line:
                    line_instance = Paragraph(line, self.mask)
                    timer = line_instance.find_all_recurring_spans()
                    # for i in range(7):
                    #     timer_all[i] += timer[i]
                    # max_ngram = line_instance.sample_ngrams_to_mask()
                    # max_histogram[max_ngram] += 1
                    line_instance.sample_ngrams_to_mask()
                    line_instance.mask_recurring_spans()
                    # paragraph_entry = line_instance.get_splinter_data()
                    self.all_line_ob.append(line_instance)
                    # TODO: Maybe here it's a good place to add (stochasticly) to train/validation
                    count += 1
                    if count % 50000 == 0:
                        ovrl_time = time.time() - st_time
                        time_left =  (ovrl_time/count) * (PROCESSED_DATA_SIZE - count)
                        print("%d (%.2f%%) paragraphs were processed at %.2fs (%.2fs per line)" %
                              (count, 100 * count/PROCESSED_DATA_SIZE, ovrl_time, ovrl_time/count))
                        print("     Expected to finish in %.2f minutes" % (time_left / 60))
                        if count % 100000 == 0:
                            with open('all_paragraphs_{}.pkl'.format(file_idx), 'wb+') as out_f:
                                pickle.dump(self.all_line_ob, out_f, pickle.HIGHEST_PROTOCOL)
                else:
                    break

    def select_ngrams(self):
        pass

    def mask_spans_all(self):
        pass

if __name__ == '__main__':
    ds = SplinterDataset()