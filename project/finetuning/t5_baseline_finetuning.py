import itertools
from transformers import AutoTokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration
import t5_baseline_fintune_dataset as baseline_dataset
MODEL_PATH = "../pretraining/t5_baseline_pretrain_output_dir/checkpoint-3900"
DATA_PATH = "../../data/splinter_data/squad"
SEED = [42, 43, 44]
EXAMPLES = [16, 32, 64, 128, 256, 512, 1024]
train_file_name = lambda seed, examples: f"squad-train-seed-{seed}-num-examples-{examples}.jsonl"
DEV_FILE_NAME = "dev.jsonl"

if __name__ == "__main__":
    settings = itertools.product(SEED,EXAMPLES)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    for seed,examples in settings:
        train_dataset = baseline_dataset.SquaDataset(*baseline_dataset.create_squad_train(seed,examples,tokenizer))
        val_datset = baseline_dataset.SquaDataset(*baseline_dataset.create_squad_val(1000,tokenizer))
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)


        args = { # TODO : BEFORE FINETUNING CHOOSE SETTINGS
            'output_dir': f"output_dir/t5_finetune_{seed}_{examples}/",
            'do_eval': True,
            'evaluation_strategy': "steps",
            'num_train_epochs': 3,
            # 'max_steps': 200*int(examples/8),
            'save_steps': 20*int(examples/8),
            'save_total_limit': 10,
            'eval_steps': int(examples/8),
            'dataloader_pin_memory': False,
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': int(examples/8),
            'warmup_ratio': 0.1,
            'logging_steps': 1,
        }

        trainer_config = {
            'model': model,
            'args': TrainingArguments(**args),
            'data_collator': baseline_dataset.SquaDataColate(tokenizer=tokenizer),
            'train_dataset': train_dataset,
            'eval_dataset': val_datset,
        }

        trainer = Trainer(**trainer_config)
        trainer.train()
