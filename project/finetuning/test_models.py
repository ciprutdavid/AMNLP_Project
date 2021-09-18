import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration
import pretraining.splinter_t5_model as splinter_model
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
    normalize_answer, compute_exact, compute_f1, make_eval_dict, merge_eval)
import json
import torch.nn.functional as F

T5_CHECKPOINT_PATH = ''
SPLINTER_CHECKPOINT_PATH = ''
SQUAD_PATH = ''
NATURAL_Q_PATH = ''
TEXTBOOKQA_PATH = ''
DIM = 512


class EvaluateModel:
    def __init__(self, model = None, path = None):
        if model == None:
            model = splinter_model
            path = '../model_checkpoints/t5_splinter_finetuned/splinter_finetune_42_512/checkpoint-10/pytorch_model.bin'
        self.model = model.from_pretrained(path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained('t5-base')
        self.data = self.natural_qa_data()

    def compute_f1_of_all_dataset(self):
        sum = 0.
        valid_count = 0
        for _ in range(12000):
            res = self.interpretate()
            if res != -1:
                valid_count += 1
                sum += res
        return sum / valid_count, valid_count


    def interpretate(self):
        with torch.no_grad():
            try:
                line = next(self.data)
            except StopIteration:
                return -1
            if 'context' not in line:
                print("Metadata entry of the dataset")
                return -1
            if not(line['context'].startswith("<P>") or line['context'].startswith("<p>")):
                print("bad start")
                return -1

            y = self._find_start_end_indices(self.tokenizer(line['context'].lower()).input_ids,
                                            self.tokenizer(line['qas'][0]['answers'][0].lower()).input_ids)
            num_occ_of_answer = len(line['qas'][0]['detected_answers'][0]['char_spans'])
            prepared_line = line['context'] +  " </s> " + line['qas'][0]['question'] + " " + "<extra_id_0>"
            tokenized = self.tokenizer(prepared_line, padding = 'max_length', truncation = True, max_length = DIM).input_ids
            if len(y) != num_occ_of_answer or len(tokenized) > DIM:
                return -1
            tokenized_2d = torch.tensor(tokenized).view(1, len(tokenized)).to(device='cuda')
            try:
                pred = torch.argmax(self.model(tokenized_2d), dim=1)
            except RuntimeError:
                print(prepared_line)
                return -1
            st, en = pred[0], pred[1]
            if st > en:
                en = st
            pred_text = self.tokenizer.decode(tokenized[st:en+1])
            f1_score =  compute_f1(pred_text, line['qas'][0]['answers'][0])
            if f1_score > 0:
                print(pred_text)
                print(line['qas'][0]['answers'][0])
                print("")
            return f1_score

            # print(prepared_line)
            # print("answer: {}".format(line['qas'][0]['answers'][0]))
            # print("top 5: {}".format(torch.topk(pred, 5).indices))
            # print("labels: {}".format(y))
            # print("")
            # return pred

    def natural_qa_data(self):
        path = './mrqa-few-shot/naturalquestions/dev.jsonl'
        with open(path, 'rb') as f:
            test_set = list(f)
        for sen in test_set:
            yield(json.loads(sen))

    def textbook_qa_data(self):
        path = './mrqa-few-shot/textbookqa/dev.jsonl'
        with open(path, 'rb') as f:
            test_set = list(f)
        for sen in test_set:
            yield(json.loads(sen))


    def _find_start_end_indices(self, context_ids, answer_ids):
        res = []
        context_ids = context_ids[:-1] # remove id 1 from the end which the tokenizer adds
        answer_ids = answer_ids[:-1] # remove id 1 from the end which the tokenizer adds
        for i in range(len(context_ids) - len(answer_ids) + 1):
            if context_ids[i:i+len(answer_ids)] == answer_ids:
                start_index = i
                end_index = i + len(answer_ids) - 1
                res.append([start_index, end_index])
        return res
