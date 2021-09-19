import json
from project.testing.testing_utils import *

NUM_OF_EXAMPLES = [0, 32, 128, 1024]
DATA_PATH = "data/splinter_data/naturalquestions/dev.jsonl"


def load_data(tokenizer=AutoTokenizer.from_pretrained('t5-base'), size=200):
    data = []
    labels = []
    if size is not None:
        count = 0
    with open(DATA_PATH, 'r') as file:
        for item in file:
            item_dict = json.loads(item)
            if 'context' not in item_dict: continue
            context = item_dict['context']
            context = context.replace("<P> ", "")
            context = context.replace(" </P>", "")
            masked_context = context + " </s> " + item_dict['qas'][0]['question'] + " " + "<extra_id_0>"
            if len(tokenizer(masked_context)['input_ids']) > 512: continue
            data.append(masked_context)
            labels.append(item_dict['qas'][0]['answers'][0])
            if size is not None:
                count += 1
                if count == size:
                    break
    return data, labels


if __name__ == "__main__":
    evaluate_models(load_data,"Natural QA")