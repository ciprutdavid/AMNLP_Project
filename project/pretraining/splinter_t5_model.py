import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Config

MASK_ID = 32099  # mask id of <extra_id_0>
DIM = 512 # seq_len


class SplinterT5Model(torch.nn.Module):
    def __init__(self):
        super(SplinterT5Model, self).__init__()
        self.t5_encoder = T5EncoderModel(T5Config())
        self.S = nn.Parameter(torch.randn(size=(DIM, DIM)))
        self.E = nn.Parameter(torch.randn(size=(DIM, DIM)))

    def forward(self, input_ids, attention_mask):
        question_indices = (input_ids == MASK_ID).long().view(-1)
        relevant_attention_mask = attention_mask[question_indices, :]
        relevant_attention_mask[relevant_attention_mask == 0] = float('-inf')

        X_T = self.t5_encoder(input_ids)
        X = torch.transpose(X_T, 2, 1)

        start_prob = F.softmax((X_T @ self.S @ X).view(-1, DIM)[question_indices,:] * relevant_attention_mask,-1)
        end_prob = F.softmax((X_T @ self.S @ X).view(-1, DIM)[question_indices,:] * relevant_attention_mask,-1)

        return start_prob, end_prob
