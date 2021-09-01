import transformers
from transformers import Trainer, T5Model, T5Config, AutoTokenizer
import t5_baseline_dataset as baseline_data



TRAIN_PATH = "/content/drive/MyDrive/Colab Notebooks/AMNLP_project/data/train"
VAL_PATH = "/content/drive/MyDrive/Colab Notebooks/AMNLP_project/data/val"
model_config = T5Config()
tokenizer = AutoTokenizer.from_pretrained('t5-base', cache_dir="t5_baseline_pretrain_output_dir/tokenizer_cache/")
args = {
    'output_dir':"t5_baseline_pretrain_output_dir/",
    'do_eval':True,
    'evaluation_strategy':"steps",
    'num_train_epochs':20,
    # 'lr_scheduler_type':[],
    'save_strategy':[],
    'save_steps':1000,
    'save_total_limit':10,
    'eval_steps':1000,
}

trainer_config ={
    'model': T5Model(model_config),
    'args':args,
    'data_collator':baseline_data.T5_Collate(tokenizer),
    'train_dataset':baseline_data.load_data(TRAIN_PATH),
    'eval_dataset':baseline_data.load_data(VAL_PATH),
    'tokenizer':tokenizer,
    # 'compute_metrics':[],
}