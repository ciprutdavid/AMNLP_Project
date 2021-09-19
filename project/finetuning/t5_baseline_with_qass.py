import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Config, AutoTokenizer, T5ForConditionalGeneration

MASK_ID = 32099  # mask id of <extra_id_0>
DIM = 512  # seq_len
MODEL_PATH = "model_checkpoints/t5_baseline_pretrained/checkpoint-3900"

class BaselineWithQass(torch.nn.Module):
    def __init__(self, t5_model_path=MODEL_PATH):
        super(BaselineWithQass, self).__init__()
        self.t5_encoder = T5ForConditionalGeneration.from_pretrained(t5_model_path).encoder  # From pretrained baseline
        self.S = nn.Parameter(torch.randn(size=(DIM, DIM)), requires_grad=True)
        self.E = nn.Parameter(torch.randn(size=(DIM, DIM)), requires_grad=True)

    def forward(self, input_ids):
        question_indices = (input_ids == MASK_ID).view(-1)  # dim = NUM_OF_BATCH*SEQ_LEN

        X_T = self.t5_encoder(input_ids).last_hidden_state  # dim = NUM_OF_BATCH. x BATCH_SIZE x SEQ_LEN
        X = torch.transpose(X_T, 2, 1)  # dim = NUM_OF_BATCH x SEQ_LEN x BATCH_SIZE

        start_scores = (X_T @ self.S @ X).view(-1, DIM)[question_indices, :]
        end_scores = (X_T @ self.E @ X).view(-1, DIM)[question_indices, :]
        scores = torch.cat((start_scores, end_scores))
        return scores

    def generate(self, input_ids, tokenizer=AutoTokenizer.from_pretrained('t5-base')):
        """
        Assumptions :
            input_ids: Assume batch_size = 1 (input_ids.shape[0] = 1)
            Only one sentinel token is present

        """
        with torch.no_grad():
            output = F.softmax(self.forward(input_ids), dim=-1)
            start_probs = output[:1, :]
            end_probs = output[1:, :]
            start_indices = torch.argmax(start_probs, dim=-1).to('cpu')
            end_indices = torch.argmax(end_probs, dim=-1).to('cpu')
        return input_ids[:, start_indices:end_indices + 1]
