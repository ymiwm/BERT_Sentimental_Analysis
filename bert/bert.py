from torch import nn
from .modeling_bert import BertModel
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")

        self.linear = nn.Linear(768, 1)

    def forward(self, bert_ids):
        bert_outputs, _ = self.bert(bert_ids)
        bert_outputs = bert_outputs.mean(1)

        output = torch.sigmoid(self.linear(bert_outputs))

        return output