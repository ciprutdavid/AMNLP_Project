import numpy as np
import json

PROCESSED_DATA_PATH = "E:/Studies/TAU/NLP/processed"
TRAIN_DATA_PATH = "E:/Studies/TAU/NLP/train"
VAL_DATA_PATH  = "E:/Studies/TAU/NLP/test"
VAL_INDICES_PATH  ="../data/val_indices.json"

VALIDATION_SIZE = 500
PROCESSED_DATA_SIZE = 17610994

def select_validation_indices(len_of_processed_data = PROCESSED_DATA_SIZE, validation_size = VALIDATION_SIZE):
    val_indices = sorted(np.random.choice(len_of_processed_data, validation_size))
    val_indices_dict = {'Validation_indices' : list(map(int, val_indices))}
    with open(VAL_INDICES_PATH, 'w+') as f:
        json.dump(val_indices_dict, f)

def split_train_validation(OUTPUT_DIR = "../data/"):
    with open("../data/val_indices.json") as f:
        val_dict = json.load(f)
    val_indices = val_dict['Validation_indices']
    max_val_idx = val_indices[-1]
    train_data = open(OUTPUT_DIR + "train", 'w+')
    val_lines = []
    with open(PROCESSED_DATA_PATH, 'r') as data:
        line_counter = 0
        for line in data:
            if line_counter > max_val_idx or line_counter != val_indices[0]:
                train_data.write(line)
            else:
                val_indices.pop(0)
                val_lines.append(line)
            line_counter += 1
    train_data.close()
    with open(OUTPUT_DIR + "test", 'w+') as val_data:
        val_data.writelines(val_lines)

if __name__ == '__main__':
    split_train_validation("E:/Studies/TAU/NLP/")
    # select_validation_indices()




