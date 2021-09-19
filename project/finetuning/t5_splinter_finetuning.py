import itertools
from transformers import AutoTokenizer, Trainer, TrainingArguments
import project.pretraining.splinter_t5_model as splinter_model
import project.finetuning.finetuning_utils as utils

MODEL_PATH = "model_checkpoints/t5_splinter_pretrained/checkpoint-2400"
DATA_PATH = "data/splinter_data/squad"
SEED = [42]
EXAMPLES = [32, 128, 512]
train_file_name = lambda seed, examples: f"squad-train-seed-{seed}-num-examples-{examples}.jsonl"
DEV_FILE_NAME = "dev.jsonl"


if __name__ == "__main__":
    settings = itertools.product(SEED, EXAMPLES)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    for seed, examples in settings:
        train_dataset = utils.SquaDataset(*utils.create_squad_train(seed, examples, tokenizer))
        val_datset = utils.SquaDataset(*utils.create_squad_val(1000, tokenizer))
        model = splinter_model.from_pretrained(MODEL_PATH,device='cuda')
        model.reinitialize_qas_weights()

        args = {
            # output setting
            'output_dir': f"project/finetuning/output_dir/splinter_finetune_{seed}_{examples}/",

            # save setting
            'save_strategy': "epoch",
            'save_steps': 1,
            'save_total_limit': 10,

            # evaluation setting
            'do_eval': True,
            'evaluation_strategy': "epoch",
            'eval_steps': 1,
            'per_device_eval_batch_size': 8,

            # train setting
            'num_train_epochs': 10,
            'per_device_train_batch_size': 8,
            'warmup_ratio': 0.1,

            # logging setting
            'logging_strategy': "epoch",
            'logging_steps': 1,

            # dataloader setting
            'dataloader_pin_memory': False
        }

        trainer_config = {
            'model': model,
            'args': TrainingArguments(**args),
            'data_collator': utils.SquaDataColate(tokenizer=tokenizer),
            'train_dataset': train_dataset,
            'eval_dataset': val_datset,
        }

        trainer = utils.QASS_Trainer(**trainer_config)
        trainer.train()
