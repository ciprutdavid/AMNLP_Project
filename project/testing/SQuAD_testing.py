import json
from project.testing.testing_utils import *


NUM_OF_EXAMPLES = [0, 32, 128, 1024]
DATA_PATH = "../../data/splinter_data/squad/dev.jsonl"


def load_data(tokenizer=AutoTokenizer.from_pretrained('t5-base'),size=200):  # TODO : finish val data creation
    data = []
    labels = []
    skip_count = 0
    count = 0
    with open(DATA_PATH, 'r') as file:
        for idx, item in enumerate(file):
            if skip_count <= 1000:
                skip_count += 1
                continue
            item_dict = json.loads(item)
            masked_context = item_dict['context'] + " </s> " + item_dict['qas'][0][
                'question'] + " " + "<extra_id_0>"
            if idx == 0 or len(tokenizer(masked_context)['input_ids']) > 512: continue
            data.append(item_dict['context'] + " </s> " + item_dict['qas'][0]['question'] + " " + "<extra_id_0>")
            labels.append(item_dict['qas'][0]['answers'][0])
            count += 1
            if count == size:
                break
    return data, labels


if __name__ == "__main__":
    evaluate_models(load_data,"SQuAD")