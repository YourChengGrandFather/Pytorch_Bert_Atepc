TRAIN_FILE_PATH = './output/process/atepc.train.csv'
TEST_FILE_PATH = './output/process/atepc.test.csv'

BIO_O_ID = 0
BIO_B_ID = 1
BIO_I_ID = 2
BIO_MAP = {'O': BIO_O_ID, 'B-ASP': BIO_B_ID, 'I-ASP': BIO_I_ID}
ENT_SIZE = 3

POLA_O_ID = -1
POLA_MAP = ['Negative', 'Positive']
POLA_DIM = 2

BERT_PAD_ID = 0
BERT_MODEL_NAME = '../huggingface/bert-base-chinese'
BERT_DIM = 768

SRD = 3  # Semantic-Relative Distance

BATCH_SIZE = 50
EPOCH = 100
LR = 1e-4

MODEL_DIR = './output/models/'

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS = 1e-10
LCF = 'cdw' # cdw cdm fusion