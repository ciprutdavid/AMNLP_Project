import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Config

MASK_ID = 32099  # mask id of <extra_id_0>


class SplinterT5Model(torch.nn.Module):
    def __init__(self, S_size, E_size):
        super(SplinterT5Model, self).__init__()
        self.t5_encoder = T5EncoderModel(T5Config())
        self.S = nn.Parameter(torch.randn(size=S_size))
        self.E = nn.Parameter(torch.randn(size=E_size))

    def forward(self, input_ids, attention_mask):
        '''
        :param input_ids: dimension Bx512
        :return softmax per question in each batch: dimension (Q_1+Q_2+...+Q_B)x512
        '''

        mask_indices = (input_ids == MASK_ID).long()  # Bx512 (Batch x Max_Seq_Len)
        sum_Q = torch.count_nonzero(mask_indices,dim=1)
        real_seq_len_by_batch = torch.count_nonzero(attention_mask,dim=1)
        attention_mask = attention_mask.view(-1)

        mask_indices = mask_indices.view(-1)
        X_T = self.t5_encoder(input_ids).view(-1, 512)  # (B*512)x512 (Batch * Max_Seq_Len) x Hidden_dim
        X_Q = X_T[mask_indices, :].T
        start = (X_T @ self.S @ X_Q) # (Batch * Max_Seq_Len) x (Q_1+Q_2+...+Q_B)
        end = (X_T @ self.E @ X_Q) # (Batch * Max_Seq_Len) x (Q_1+Q_2+...+Q_B)

    def gather_mask_token_embeddings(self, embeddings, mask_indices):
        pass
