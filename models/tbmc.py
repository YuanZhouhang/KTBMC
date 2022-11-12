#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import torch
import torch.nn as nn
from torch import linalg as LA
from pytorch_pretrained_bert.modeling import BertModel, BertLayerNorm, BertLayer

from models.image import ImageEncoder


class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):

        super(ImageBertEmbeddings, self).__init__()
        self.args = args
        self.img_embeddings = nn.Linear(args.input_img_sz, args.hidden_sz)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.fc = nn.Linear(args.hidden_sz, args.hidden_sz // 2)
        self.LayerNorm = BertLayerNorm(384)  # not safe
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, input_imgs, token_type_ids):
        bsz = input_imgs.size(0)
        seq_length = self.args.num_image_embeds + 2  # +2 for CLS and SEP Token

        cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        cls_token_embeds = self.word_embeddings(cls_id)

        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)

        imgs_embeddings = self.img_embeddings(input_imgs)

        token_embeddings = torch.cat(
            [cls_token_embeds, imgs_embeddings, sep_token_embeds], dim=1
        )

        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.fc(embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEncoder(nn.Module):
    def __init__(self, config, num_hidden_layers):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class MultimodalBertEncoder(nn.Module):
    def __init__(self, args, classes=2):
        super(MultimodalBertEncoder, self).__init__()
        print("Init MultimodelBertEncoder")
        self.args = args
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings

        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        self.img_encoder = ImageEncoder(args)
        
        self.encoder = BertEncoder(bert.config, num_hidden_layers=3) # bert has 12 hidden_layers
        self.pooler = bert.pooler
        self.clf = nn.Linear(args.hidden_sz, classes)

    def forward(self, input_txt, attention_mask, segment, input_img1, input_img2):
        bsz = input_txt.size(0)
        attention_mask = torch.cat(
            [
                torch.ones(bsz, self.args.num_image_embeds + 2).long().cuda(),
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        img_tok = (
            torch.LongTensor(input_txt.size(0), self.args.num_image_embeds + 2)
                .fill_(0)
                .cuda()
        )
        # with torch.no_grad():
        img1 = self.img_encoder(input_img1)  # BxNx3x224x224 -> BxNx2048
        img2 = self.img_encoder(input_img2)  # BxNx3x224x224 -> BxNx2048
        img_dif = img1 - img2

        img_embed_out_1 = self.img_embeddings(img1, img_tok)
        img_embed_out_2 = self.img_embeddings(img_dif, img_tok)
        img_embed_out = torch.cat([img_embed_out_1, img_embed_out_2], 2)

        txt_embed_out = self.txt_embeddings(input_txt, segment)

        if self.args.with_no_text is True:
            txt_embed_out = torch.zeros(txt_embed_out.shape).cuda()

        if self.args.with_no_image is True:
            img_embed_out = torch.zeros(img_embed_out.shape).cuda()

        encoder_input = torch.cat([img_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID
        encoded_layers = self.encoder(
            encoder_input, extended_attention_mask, output_all_encoded_layers=False
        )
        return self.pooler(encoded_layers[-1])


class MultimodalBertClf(nn.Module):
    def __init__(self, args, classes=2):
        super(MultimodalBertClf, self).__init__()
        self.freeze = True
        self.args = args
        self.enc = MultimodalBertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, classes)

    def forward(self, txt, mask, segment, img1, img2):
        if self.freeze:
            with torch.no_grad():
                x = self.enc(txt, mask, segment, img1, img2)
        else:
            x = self.enc(txt, mask, segment, img1, img2)
        return self.clf(x)
