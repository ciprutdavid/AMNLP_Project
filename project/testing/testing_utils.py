from transformers import AutoTokenizer
from transformers.data.metrics.squad_metrics import compute_f1
import project.pretraining.splinter_t5_model as splinter_model
import project.finetuning.t5_baseline_with_qass as basline_qass
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt

t5_models_dict = {
    0: "pretrained",
    32: "model_checkpoints/t5_baseline_finetuned/t5_finetune_42_32/checkpoint-40",
    128: "model_checkpoints/t5_baseline_finetuned/t5_finetune_42_128/checkpoint-150",
    512: "model_checkpoints/t5_baseline_finetuned/t5_finetune_42_512/checkpoint-600"
}

splinter_models_dict = {
    0: "model_checkpoints/t5_splinter_pretrained/checkpoint-2400",
    32: "model_checkpoints/t5_splinter_finetuned/splinter_finetune_42_32/checkpoint-40",
    128: "model_checkpoints/t5_splinter_finetuned/splinter_finetune_42_128/checkpoint-150",
    512: "model_checkpoints/t5_splinter_finetuned/splinter_finetune_42_512/checkpoint-600"
}


def evaluate_f1(model, data, labels, tokenizer=AutoTokenizer.from_pretrained('t5-base')):
    f1_values = []
    for idx, question in tqdm(enumerate(data)):
        question_ids = tokenizer(question, padding='max_length', return_tensors='pt').input_ids
        predicted_ids = model.generate(question_ids)[0]
        if len(predicted_ids) == 0:
            predicted_answer = ""
        else:
            predicted_answer = tokenizer.decode(predicted_ids)
        ground_truth = labels[idx]
        f1_values.append(compute_f1(ground_truth, predicted_answer))
    return statistics.mean(f1_values)


def evaluate_models(dataloader,dataset_name):

    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    data, labels = dataloader(tokenizer)
    baseline_f1_scores = []
    for key in t5_models_dict:
        model = basline_qass.from_pretrained(t5_models_dict[key],'cpu')
        baseline_f1_scores.append(evaluate_f1(model, data, labels, tokenizer))
        print(f'{key} : {baseline_f1_scores[-1]}')

    baseline_info = {
        'model_name': 'T5 Baseline - Ours',
        'f1_scores': baseline_f1_scores
    }

    splinter_f1_scores = []
    for key in splinter_models_dict:
        model = splinter_model.from_pretrained(splinter_models_dict[key], 'cpu')
        splinter_f1_scores.append(evaluate_f1(model, data, labels, tokenizer))
        print(f'{key} : {splinter_f1_scores[-1]}')

    splinter_info = {
        'model_name': 'Splinter - Ours',
        'f1_scores': splinter_f1_scores
    }
    plot_evaluation([baseline_info, splinter_info], dataset_name)


def plot_evaluation(model_info_list,dataset_name):
    x = [0, 32, 128, 1024]
    for model_info in model_info_list:
        plt.plot(x,model_info['f1_scores'],'--x',label=model_info['model_name'])
    plt.legend()
    plt.xticks(x)
    plt.xlabel('Number of Examples Model was fine-tuned on')
    plt.ylabel('F1 Scores')
    plt.title(dataset_name)
    plt.tight_layout()
    plt.savefig('plots/' + dataset_name + 'png',dpi=300)
    plt.show()
