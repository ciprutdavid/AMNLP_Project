import json
from project.testing.testing_utils import *

NUM_OF_EXAMPLES = [0, 32, 128, 1024]
DATA_PATH = "data/splinter_data/textbookqa/dev.jsonl"


def load_data(tokenizer=AutoTokenizer.from_pretrained('t5-base')):
    data = []
    labels = []
    with open(DATA_PATH, 'r') as file:
        for item in file:
            item_dict = json.loads(item)
            if 'context' not in item_dict: continue
            masked_context = item_dict['context'] + " </s> " + item_dict['qas'][0]['question'] + " " + "<extra_id_0>"
            if len(tokenizer(masked_context)['input_ids']) > 512: continue
            data.append(masked_context)
            labels.append(item_dict['qas'][0]['answers'][0])
    return data, labels


if __name__ == "__main__":
    evaluate_models(load_data,"Textbook QA")