import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from config import *
from torchcrf import CRF
from transformers.models.bert.modeling_bert import BertAttention, BertPooler

from transformers import logging
logging.set_verbosity_error()

config = BertConfig.from_pretrained(BERT_MODEL_NAME)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        # 冻结Bert参数，只训练下游模型
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False
        self.ent_linear = nn.Linear(BERT_DIM, ENT_SIZE)
        self.crf = CRF(ENT_SIZE, batch_first=True)
        self.pola_linear2 = nn.Linear(BERT_DIM * 2, BERT_DIM)
        self.pola_linear3 = nn.Linear(BERT_DIM * 3, BERT_DIM)
        self.pola_linear = nn.Linear(BERT_DIM, POLA_DIM)
        self.attention = BertAttention(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout()

    def get_text_encoded(self, input_ids, mask):
        return self.bert(input_ids, attention_mask=mask)[0]

    def get_entity_fc(self, text_encoded):
        return self.ent_linear(text_encoded)

    def get_entity_crf(self, entity_fc, mask):
        return self.crf.decode(entity_fc, mask)

    def get_entity(self, input_ids, mask):
        text_encoded = self.get_text_encoded(input_ids, mask)
        entity_fc = self.get_entity_fc(text_encoded)
        pred_ent_label = self.get_entity_crf(entity_fc, mask)
        return pred_ent_label

    def get_pola(self, input_ids, mask, ent_cdm, ent_cdw):
        text_encoded = self.get_text_encoded(input_ids, mask)

        # shape [b, c] -> [b, c, 768]
        ent_cdm_weight = ent_cdm.unsqueeze(-1).repeat(1, 1, BERT_DIM)
        ent_cdw_weight = ent_cdw.unsqueeze(-1).repeat(1, 1, BERT_DIM)
        cdm_feature = torch.mul(text_encoded, ent_cdm_weight)
        cdw_feature = torch.mul(text_encoded, ent_cdw_weight)

        # 根据配置，使用不同的策略，重新组合特征，在降维到768维
        if LCF == 'fusion':
            out = torch.cat([text_encoded, cdm_feature, cdw_feature], dim=-1)
            out = self.pola_linear3(out)
        if LCF == 'fusion2':
            out = torch.cat([text_encoded, cdw_feature], dim=-1)
            out = self.pola_linear2(out)
        elif LCF == 'cdw':
            out = cdw_feature

        # self-attension 结合上下文信息，增强语义
        out = self.attention(out, None)
        # pooler 取[CLS]标记位，作为整个句子的特征
        out = torch.sigmoid(self.pooler(torch.tanh(out[0])))
        return self.pola_linear(out)

    def ent_loss_fn(self, input_ids, ent_label, mask):
        text_encoded = self.get_text_encoded(input_ids, mask)
        entity_fc = self.get_entity_fc(text_encoded)
        return -self.crf.forward(entity_fc, ent_label, mask, reduction='mean')

    def pola_loss_fn(self, pred_pola, pola_label):
        return F.cross_entropy(pred_pola, pola_label)

    def loss_fn(self, input_ids, ent_label, mask, pred_pola, pola_label):
        return self.ent_loss_fn(input_ids, ent_label, mask) + \
        self.pola_loss_fn(pred_pola, pola_label)


if __name__ == '__main__':
    input_ids = torch.randint(0, 3000, (2, 30)).to(DEVICE)
    mask = torch.ones((2, 30)).bool().to(DEVICE)
    model = Model().to(DEVICE)
    # print(model.get_entity(input_ids, mask))
    ent_cdm = torch.rand((2, 30)).to(DEVICE)
    ent_cdw = torch.rand((2, 30)).to(DEVICE)
    print(model.get_pola(input_ids, mask, ent_cdm, ent_cdw))