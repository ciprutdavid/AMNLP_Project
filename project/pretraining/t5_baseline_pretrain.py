from transformers import Trainer, T5Model, T5Config, AutoTokenizer, TrainingArguments, T5ForConditionalGeneration
import t5_baseline_pretrain_dataset as baseline_data
import pickle

PROCESSED_PATH = "/home/yandex/AMNLP2021/benzeharia/project/AMNLP_Project/project/data/processed"
TRAIN_INDEX_PATH = "../data/all_train_indices.pkl"
VAL_INDEX_PATH = "../data/all_val_indices.pkl"

if __name__ == "__main__":

    print("LOADING INDICES")
    with open(TRAIN_INDEX_PATH, 'rb+') as train_index:
        train_indices = pickle.load(train_index)
    with open(VAL_INDEX_PATH, 'rb+') as val_index:
        val_indices = pickle.load(val_index)

    train_data = []
    val_data = []

    print("PREPARING TRAIN/VAL DATA")
    train_file_idx = 0
    val_file_idx = 0
    len_train = len(train_indices)
    len_val = len(val_indices)
    with open(PROCESSED_PATH, 'r',errors='ignore') as reader:
        for idx, line in enumerate(reader):
            if train_file_idx < len_train and idx == train_indices[train_file_idx]:
                train_data.append(line)
                train_file_idx += 1
            elif val_file_idx < len_val and idx == val_indices[val_file_idx]:
                val_data.append(line)
                val_file_idx += 1
            else:
                continue

    print("CREATING CONFIGURATIONS")
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model_config = T5Config(decoder_start_token_id=tokenizer.convert_tokens_to_ids(['<pad>'])[0])
    model = T5ForConditionalGeneration(model_config).to('cuda')
    args = {
        'output_dir': "t5_baseline_pretrain_output_dir/",
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
        'model': model,
        'args': TrainingArguments(**args),
        'data_collator': baseline_data.T5_Collate(tokenizer, 'cuda'),
        'train_dataset': baseline_data.WikiDataset(train_data),
        'eval_dataset': baseline_data.WikiDataset(val_data),
        'tokenizer': tokenizer
    }

    print("START TRAINING")
    trainer = Trainer(**trainer_config)
    trainer.train()
