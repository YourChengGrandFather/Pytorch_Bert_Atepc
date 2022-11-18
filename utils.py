import pandas as pd
import torch.utils.data as data
from config import *
from transformers import BertTokenizer
import torch
import random


def get_ent_pos(lst):
    items = []
    for i in range(len(lst)):
        # B-ASP 开头
        if lst[i] == 1:
            item = [i]
            while True:
                i += 1
                # 到 I-ASP 结束
                if i >= len(lst) or lst[i] != 2:
                    items.append(item)
                    break
                else:
                    item.append(i)
        i += 1
    return items

# [CLS]这个手机外观时尚，美中不足的是拍照像素低。[SEP]
# 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 1 2 2 2 0 0 0
# print(get_ent_pos([0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,1,2,2,2,0,0,0]))
# [[5, 6], [16, 17, 18, 19]]


# def get_ent_weight(max_len, ent_pos):
#     cdm = []
#     cdw = []

#     for i in range(max_len):
#         dst = min(abs(i - ent_pos[0]), abs(i - ent_pos[-1]))
#         if dst <= SRD:
#             cdm.append(1)
#             cdw.append(1)
#         else:
#             cdm.append(0)
#             cdw.append(1 / (dst - SRD + 1))
#     return cdm, cdw


def get_ent_weight(max_len, ent_pos):
    cdm = []
    cdw = []

    for i in range(max_len):
        dst = min(abs(i - ent_pos[0]), abs(i - ent_pos[-1]))
        if dst <= SRD:
            cdm.append(1)
            cdw.append(1)
        else:
            cdm.append(0)
            cdw.append(1 - (dst - SRD + 1) / max_len)

    return cdm, cdw

print(get_ent_weight(23, [5,6]))
exit()



class Dataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        file_path = TRAIN_FILE_PATH if type == 'train' else TEST_FILE_PATH
        self.df = pd.read_csv(file_path)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    def __len__(self):
        return len(self.df) - 1

    def __getitem__(self, index):
        # 相邻两个句子拼接
        text1, bio1, pola1 = self.df.loc[index]
        text2, bio2, pola2 = self.df.loc[index+1]
        text = text1 + ' ; ' + text2
        bio = bio1 + ' O ' + bio2
        pola = pola1 + ' -1 ' + pola2

        # 按自己的规则分词
        tokens = ['[CLS]'] + text.split(' ') + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # BIO标签转id
        bio_arr = ['O'] + bio.split(' ') + ['O']
        bio_label = [BIO_MAP[l] for l in bio_arr]
        
        # 情感值转数字
        pola_arr = ['-1'] + pola.split(' ') + ['-1']
        pola_label = list(map(int, pola_arr))
        
        return input_ids, bio_label, pola_label

    def collate_fn(self, batch):
        # 统计最大句子长度
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        max_len = len(batch[0][0])
        
        # 变量初始化
        batch_input_ids = []
        batch_bio_label = []
        batch_mask = []
        batch_ent_cdm = []
        batch_ent_cdw = []
        batch_pola_label = []
        batch_pairs = [] 

        for input_ids, bio_label, pola_label in batch:
            # 获取实体位置，没有实体跳过
            ent_pos = get_ent_pos(bio_label)
            if len(ent_pos) == 0:
                continue

            # 填充句子长度
            pad_len = max_len - len(input_ids)
            batch_input_ids.append(input_ids + [BERT_PAD_ID] * pad_len)
            batch_mask.append([1] * len(input_ids) + [0] * pad_len)
            batch_bio_label.append(bio_label + [BIO_O_ID] * pad_len)

            # 实体和情感分类对应
            pairs = []
            for pos in ent_pos:
                pola = pola_label[pos[0]]
                # 异常值替换
                pola = 0 if pola == -1 else pola
                pairs.append((pos, pola))
            batch_pairs.append(pairs)

            # 随机取一个实体
            sg_ent_pos = random.choice(ent_pos)
            cdm, cdw = get_ent_weight(max_len, sg_ent_pos)
            
            # 计算加权参数
            batch_ent_cdm.append(cdm)
            batch_ent_cdw.append(cdw)
            # 实体第一个字的情感极性
            pola = pola_label[sg_ent_pos[0]]
            pola = 0 if pola == -1 else pola
            batch_pola_label.append(pola)

        return (
            torch.tensor(batch_input_ids),
            torch.tensor(batch_mask).bool(),
            torch.tensor(batch_bio_label),
            torch.tensor(batch_ent_cdm),
            torch.tensor(batch_ent_cdw),
            torch.tensor(batch_pola_label),
            batch_pairs,
        )


def get_pola(model, input_ids, mask, ent_label):
    # 变量初始化
    b_input_ids = []
    b_mask = []
    b_ent_cdm = []
    b_ent_cdw = []
    b_ent_pos = []

    # 根据label解析实体位置
    ent_pos = get_ent_pos(ent_label)
    n = len(ent_pos)
    if n == 0:
        return None, None

    # n个实体一起预测，同一个句子复制n份，作为一个batch
    b_input_ids.extend([input_ids] * n)
    b_mask.extend([mask] * n)
    b_ent_pos.extend(ent_pos)
    for sg_ent_pos in ent_pos:
        cdm, cdw = get_ent_weight(len(input_ids), sg_ent_pos)
        b_ent_cdm.append(cdm)
        b_ent_cdw.append(cdw)
    
    # 列表转tensor
    b_input_ids = torch.stack(b_input_ids, dim=0).to(DEVICE)
    b_mask = torch.stack(b_mask, dim=0).to(DEVICE)
    b_ent_cdm = torch.tensor(b_ent_cdm).to(DEVICE)
    b_ent_cdw = torch.tensor(b_ent_cdw).to(DEVICE)
    b_ent_pola = model.get_pola(b_input_ids, b_mask, b_ent_cdm, b_ent_cdw)
    return b_ent_pos, b_ent_pola

if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    print(iter(loader).next())