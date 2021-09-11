import itertools
from transformers import AutoTokenizer, Trainer, TrainingArguments
import t5_baseline_fintune_dataset as baseline_dataset


DATA_PATH = "/home/david/PycharmProjects/AMNLP_Project/data/splinter_data/squad"
SEED = [42, 43, 44]
EXAMPLES = [16, 32, 64, 128, 256, 512, 1024]
train_file_name = lambda seed, examples: f"squad-train-seed-{seed}-num-examples-{examples}.jsonl"
DEV_FILE_NAME = "dev.jsonl"

if __name__ == "__main__":
    settings = itertools.product(SEED,EXAMPLES)

    for seed,examples in settings:
        tokenizer = AutoTokenizer.from_pretrained('t5-base')
        train_dataset = baseline_dataset.SquaDataset(*baseline_dataset.create_squad_train(seed,examples,tokenizer))
        val_datset =  baseline_dataset.SquaDataset(*baseline_dataset.create_squad_val(examples))
        model = '' # TODO : Figure out how to load the pretrainer model


        args = { # TODO : BEFORE FINETUNING CHOOSE SETTINGS
            'output_dir': f"t5_finetune_{seed}_{examples}/",
            'do_eval': True,
            'evaluation_strategy': "steps",
            'max_steps': 200,
            'save_steps': 10,
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
            'data_collator': baseline_dataset.SquaDataColate(),
            'train_dataset': train_dataset,
            'eval_dataset': val_datset,
        }

        trainer = Trainer(**trainer_config)
        trainer.train()
