import torch
from transformers import Trainer, T5Model, T5Config, AutoTokenizer,TrainingArguments
import t5_baseline_dataset as baseline_data



TRAIN_PATH = "/content/drive/MyDrive/Colab Notebooks/AMNLP_project/data/train"
VAL_PATH = "/content/drive/MyDrive/Colab Notebooks/AMNLP_project/data/val"
train_data = baseline_data.load_data(TRAIN_PATH)
val_data = baseline_data.load_data(VAL_PATH)
torch.cuda.set_device(0)
model_config = T5Config()
model = T5Model(model_config).to('cuda')
tokenizer = AutoTokenizer.from_pretrained('t5-base', cache_dir="t5_baseline_pretrain_output_dir/tokenizer_cache/")
args = {
    'output_dir':"t5_baseline_pretrain_output_dir/",
    'do_eval':True,
    'evaluation_strategy':"steps",
    'num_train_epochs':20,
    'save_steps':1000,
    'save_total_limit':10,
    'eval_steps':1000,
    'dataloader_pin_memory':False,
    'per_device_train_batch_size':8
}

trainer_config ={
    'model': model,
    'args':TrainingArguments(**args),
    'data_collator':baseline_data.T5_Collate(tokenizer,'cuda'),
    'train_dataset':baseline_data.WikiDataset(train_data),
    'eval_dataset':baseline_data.WikiDataset(val_data),
    'tokenizer':tokenizer,
}

if __name__ == "__main__":
    print()