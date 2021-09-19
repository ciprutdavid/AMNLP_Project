import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Model ,T5EncoderModel, T5Config, AutoTokenizer

MASK_ID = 32099  # mask id of <extra_id_0>
DIM = 512  # seq_len


class HFSplinterT5Model(torch.nn.Module):
    def __init__(self):
        super(HFSplinterT5Model, self).__init__()
        self.t5_encoder = T5Model.from_pretrained('t5-small').encoder
        self.S = nn.Parameter(torch.randn(size=(DIM, DIM)), requires_grad=True)
        self.E = nn.Parameter(torch.randn(size=(DIM, DIM)), requires_grad=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, input_ids):
        question_indices = (input_ids == MASK_ID).view(-1)  # dim = NUM_OF_BATCH*SEQ_LEN

        X_T = self.t5_encoder(input_ids).last_hidden_state  # dim = NUM_OF_BATCH. x BATCH_SIZE x SEQ_LEN
        X = torch.transpose(X_T, 2, 1)  # dim = NUM_OF_BATCH x SEQ_LEN x BATCH_SIZE

        start_scores = (X_T @ self.S @ X).view(-1, DIM)[question_indices, :]
        end_scores = (X_T @ self.E @ X).view(-1, DIM)[question_indices, :]
        scores = torch.cat((start_scores, end_scores))
        return scores

    def compute_loss(self,outputs,labels):
        return self.cross_entropy(outputs,labels)

    def reinitialize_qas_weights(self):
        self.S = nn.Parameter(torch.randn(size=(DIM, DIM)), requires_grad=True)
        self.E = nn.Parameter(torch.randn(size=(DIM, DIM)), requires_grad=True)

    def generate(self, input_ids):
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


def from_pretrained(path='t5-small', device='cuda'):
    model = HFSplinterT5Model().to(device)
    if path == 't5-small':
        return model.to(device)
    else:
        weight_dict = torch.load(path + "/pytorch_model.bin", map_location=device)
        model.load_state_dict(weight_dict)
        return model.to(device)
