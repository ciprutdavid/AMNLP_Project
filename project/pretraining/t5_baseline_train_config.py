import torch
from transformers import Trainer, T5Model, T5Config, AutoTokenizer,TrainingArguments, T5ForConditionalGeneration
import t5_baseline_dataset as baseline_data

TRAIN_PATH = "/home/yandex/AMNLP2021/davidciprut/AMNLP_Project/project/data/train"
VAL_PATH = "/home/yandex/AMNLP2021/davidciprut/AMNLP_Project/project/data/test"

torch.cuda.device(0)
train_data = baseline_data.load_data(TRAIN_PATH)
val_data = baseline_data.load_data(VAL_PATH)
tokenizer = AutoTokenizer.from_pretrained('t5-base', cache_dir="t5_baseline_pretrain_output_dir/tokenizer_cache/")
model_config = T5Config(decoder_start_token_id=tokenizer.convert_tokens_to_ids(['<pad>'])[0])
model = T5ForConditionalGeneration(model_config).to('cuda')
args = {
    'output_dir':"t5_baseline_pretrain_output_dir/",
    'do_eval':True,
    'evaluation_strategy':"steps",
    'num_train_epochs':2,
    'max_steps':20000,
    'save_steps':500,
    'save_total_limit':10,
    'eval_steps':1000,
    'dataloader_pin_memory':False,
    'per_device_train_batch_size':8,
    'gradient_accumulation_steps' : 32,
    'warmup_ratio': 0.1,
    'logging_steps': 1
}

trainer_config ={
    'model': model,
    'args':TrainingArguments(**args),
    'data_collator':baseline_data.T5_Collate(tokenizer,'cuda'),
    'train_dataset':baseline_data.WikiDataset(train_data),
    'eval_dataset':baseline_data.WikiDataset(val_data),
    'tokenizer':tokenizer
}

if __name__ == "__main__":
    trainer = Trainer(**trainer_config)
    trainer.train()