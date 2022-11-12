# author: Yuan Zhouhang
# Bacterial Therapy Prognosis Experimental Dataset
#
# ------------------------------------------------------------

import glob
import cv2
import torch
import os
import json
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_pretrained_bert import BertTokenizer
from data.vocab import Vocab


def collate_fn(batch, args):
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img1_tensor = torch.stack([row[2] for row in batch])
    img2_tensor = torch.stack([row[3] for row in batch])

    tgt_tensor = torch.tensor([row[4] for row in batch]).long()
    iamge_path = [row[5] for row in batch]

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        # if args.with_no_txt is False:
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1

    return text_tensor, segment_tensor, mask_tensor, img1_tensor, img2_tensor, tgt_tensor, iamge_path


def train_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.46777044, 0.44531429, 0.40661017],
            std=[0.12221994, 0.12145835, 0.14380469],
        ),
    ])


def test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.46777044, 0.44531429, 0.40661017],
            std=[0.12221994, 0.12145835, 0.14380469],
        ),
    ])


class TBMC_dataset(Dataset):
    def __init__(self, path, args, is_train=False, is_test=False) -> None:
        print("Init Bact_treatment_prognosis dataset")

        super().__init__()

        self.categories = ['bact', 'fungal']
        self.is_train = is_train
        self.is_test = is_test
        self.path = path
        self.tokenizer = (
            BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        )

        self.vocab = args.vocab
        self.args = args
        self.text_start_token = ["[SEP]"]
        self.max_seq_len = args.max_seq_len - args.num_image_embeds

        if is_train:
            self.transform = train_transform()
            self.sample_files = glob.glob(f"{path}/images/train/*/*")

        else:
            self.transform = test_transform()
            self.sample_files = glob.glob(f"{path}/images/val/*/*")

        if is_test:
            self.transform = test_transform()
            self.sample_files = glob.glob(f"{path}/images/test/*/*")

        self.samples = []

        for f in self.sample_files:
            gt = self.categories.index(f.split(os.sep)[-2])
            sequence_files = sorted(glob.glob(f"{f}/*"))
            for i in range(len(sequence_files) - 1):
                filename = sequence_files[i].split('/')[-1][:-4]
                self.samples.append((gt, filename, sequence_files[i], sequence_files[i + 1]))

    def __getitem__(self, item):
        gt, filename, imagepath_1, imagepath_2 = self.samples[item]

        with open(f'{self.path}texts_txt/{self.categories[gt]}/{filename}.txt', 'r') as file:
            sentence = file.read()

        sentence = (
                self.text_start_token
                + self.tokenizer(sentence)[:(self.args.max_seq_len - 1)]
        )
        segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor([
            self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
            for w in sentence
        ])

        # The first SEP is part of Image Token.
        segment = segment[1:]
        sentence = sentence[1:]
        # The first segment (0) is of images.
        segment += 1

        image1 = self.transform(pil_loader(imagepath_1))
        image2 = self.transform(pil_loader(imagepath_2))

        return sentence, segment, image1, image2, gt, imagepath_1

    def __len__(self):
        return len(self.samples)


def pil_loader(imgname):
    with open(imgname, 'rb') as f:
        return Image.open(imgname).convert('RGB')

