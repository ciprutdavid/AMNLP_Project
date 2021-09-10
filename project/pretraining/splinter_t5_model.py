import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Config

MASK_ID = 32099  # mask id of <extra_id_0>
DIM = 512  # seq_len


class SplinterT5Model(torch.nn.Module):
    def __init__(self):
        super(SplinterT5Model, self).__init__()
        self.t5_encoder = T5EncoderModel(T5Config())
        self.S = nn.Parameter(torch.randn(size=(DIM, DIM)))
        self.E = nn.Parameter(torch.randn(size=(DIM, DIM)))

    def forward(self, input_ids, attention_mask):
        question_indices = (input_ids == MASK_ID).long().view(-1)  # dim = NUM_OF_BATCH*SEQ_LEN
        relevant_attention_mask = attention_mask[question_indices, :]  # dim = BATCH_SIZE x SEQ_LEN
        relevant_attention_mask[relevant_attention_mask == 0] = float('-inf')  # dim BATCH_SIZE x SEQ_LEN

        X_T = self.t5_encoder(input_ids)  # dim = NUM_OF_BATCH. x BATCH_SIZE x SEQ_LEN
        X_T = X_T * relevant_attention_mask.unsqeeeze(-1)
        X = torch.transpose(X_T, 2, 1)  # dim = NUM_OF_BATCH x SEQ_LEN x BATCH_SIZE

        start_scores = (X_T @ self.S @ X).view(-1, DIM)[question_indices, :]
        end_scores = (X_T @ self.S @ X).view(-1, DIM)[question_indices, :]

        return start_scores, end_scores
