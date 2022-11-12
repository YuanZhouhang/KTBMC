

import os
import torch
import numpy as np
import random
import torch.distributed as dist
import torch.utils.data.distributed
from pytorch_pretrained_bert import BertTokenizer
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
from warnings import simplefilter

from data.vocab import Vocab

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

from utils.args import get_args
from data.ktbmc_dataset import KTBMC_dataset
from models.ktbmc import MultimodalBertClf
from utils.trainer.ktbmc_trainer import KTBMC_Trainer


def seed_torch(seed=16):
    random.seed(seed)
    os.environ['RANK'] = '0'
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch()
    args = get_args()
    vocab = Vocab()
    bert_tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    )
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz

    data_path = args.data_path
    trainset = TBMC_dataset(data_path, args, is_train=True)
    testset = TBMC_dataset(data_path, args, is_train=False)

    model = MultimodalBertClf(args)
    trainner = TBMC_Trainer(model, trainset, testset, args)
    trainner.train(0)