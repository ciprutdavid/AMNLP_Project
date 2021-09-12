import torch.nn.functional as F
import splinter_t5_model as model
from transformers import Trainer, T5Config, AutoTokenizer, TrainingArguments, T5ForConditionalGeneration
from splinter_dataset import SplinterDatasetWrapper, SplinterCollate



class SplinterT5Trainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        start_scores, end_scores = model(**inputs)
        start_loss = F.nll_loss(start_scores, labels['start_labels'])
        end_loss = F.nll_loss(end_scores, labels['end_labels'])
        loss = start_loss + end_loss
        return loss

tokenizer = AutoTokenizer.from_pretrained('t5-base', cache_dir='../data/t5_tokenizer_cache/')
splinter_model = model.SplinterT5Model()

args = {
    'output_dir': "t5_splinter_pretrain_output_dir/",
    'do_eval': True,
    'evaluation_strategy': "steps",
    'max_steps': 20000,
    'save_steps': 50,
    'save_total_limit': 10,
    'eval_steps': 32,
    'dataloader_pin_memory': False,
    'per_device_train_batch_size': 8,
    'gradient_accumulation_steps': 32,
    'warmup_ratio': 0.1,
    'logging_steps': 1
}

trainer_config = {
    'model': splinter_model,
    'args': TrainingArguments(**args),
    'data_collator': SplinterCollate(tokenizer, 'cuda'),
    'train_dataset': SplinterDatasetWrapper('train'),
    'eval_dataset': SplinterDatasetWrapper('validation'),
    'tokenizer': tokenizer
}

if __name__ == "__main__":
    trainer = SplinterT5Trainer(**trainer_config)
    trainer.train()
