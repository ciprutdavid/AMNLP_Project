from transformers import AutoTokenizer
import json
import time


DATA_PATH = "/home/david/PycharmProjects/AMNLP_Project/data/splinter_data/squad"
SEED = [42, 43, 44, 45, 46]
EXAMPLES = [16, 32, 64, 128, 256, 512, 1024]
train_file_name = lambda seed, examples: f"squad-train-seed-{seed}-num-examples-{examples}.jsonl"
DEV_FILE_NAME = "dev.jsonl"


def create_squad_train(seed,examples,tokenizer):
    data = []
    labels = []
    if seed not in SEED:
        raise Exception("seed needs to be 42,43,44,45 or 46")
    if examples not in EXAMPLES:
        raise Exception("examples needs to be 16,32,64,128,256,512 or 1024")
    path = DATA_PATH + '/' + train_file_name(seed,examples)
    with open(path,'r') as file:
        for item in file:
            item_dict = json.loads(item)
            if 'context' not in item_dict:
                continue
            else:
                if len(tokenizer(item_dict['context'])['input_ids']) > 512: continue
                data.append(item_dict['context'] + " " + item_dict['qas'][0]['question'] + " </s> " + "<extra_id_0>")
                labels.append("<extra_id_0> " + item_dict['qas'][0]['answers'][0] + " </s>")



def create_squad_val():
    path  = DATA_PATH + '/' + DEV_FILE_NAME
    with open(path,'r') as file:
        print()





if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    s = time.time()
    create_squad_train(42,1024,tokenizer)
    e = time.time()
    print(e-s)