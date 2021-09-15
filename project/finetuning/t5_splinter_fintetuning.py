import itertools
from transformers import AutoTokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration
import t5_baseline_fintune_dataset as baseline_dataset
import project.pretraining.splinter_t5_model as splinter_model
import torch.nn.functional as F

MODEL_PATH = "../pretraining/t5_splinter_pretrain_output_dir/checkpoint-2800"
DATA_PATH = "../../data/splinter_data/squad"
SEED = [42, 43, 44]
EXAMPLES = [16, 32, 64, 128, 256, 512, 1024]
train_file_name = lambda seed, examples: f"squad-train-seed-{seed}-num-examples-{examples}.jsonl"
DEV_FILE_NAME = "dev.jsonl"


class SplinterT5Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs, labels)
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    settings = itertools.product(SEED, EXAMPLES)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    for seed, examples in settings:
        train_dataset = baseline_dataset.SquaDataset(*baseline_dataset.create_squad_train(seed, examples, tokenizer))
        val_datset = baseline_dataset.SquaDataset(*baseline_dataset.create_squad_val(1000, tokenizer))
        model = splinter_model.from_pretrained(MODEL_PATH)

        args = {  # TODO : BEFORE FINETUNING CHOOSE SETTINGS
            # output setting
            'output_dir': f"output_dir/t5_finetune_{seed}_{examples}/",

            # save setting
            'save_strategy': "epoch",
            'save_steps': 1,
            'save_total_limit': 10,

            # evaluation setting
            'do_eval': True,
            'evaluation_strategy': "epoch",
            'eval_steps': int(examples / 8),
            'per_device_eval_batch_size': 8,

            # train setting
            'num_train_epochs': 10,
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': int(examples / 8),
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
            'data_collator': baseline_dataset.SquaDataColate(tokenizer=tokenizer),
            'train_dataset': train_dataset,
            'eval_dataset': val_datset,
        }

        trainer = SplinterT5Trainer(**trainer_config)
        trainer.train()
