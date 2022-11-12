

import math
import os
import time
import torch
import functools
import torchmetrics
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from data.bact_treatment_prognosis_dataset import collate_fn
from utils.checkpoint import load_checkpoint, save_checkpoint
from torchmetrics.functional import auc


class TBMC_Trainer(object):
    def __init__(self, model, train_dataset, test_dataset, val_dataset,args) -> None:
        super().__init__()
        self.model = model.cuda()

        self.optm = Adam(model.parameters(), lr=args.lr, betas=(args.beta0, args.beta1), weight_decay=args.weight_decay)
        self.optm_drop = torch.optim.lr_scheduler.ExponentialLR(self.optm, gamma=0.98)

        collate = functools.partial(collate_fn, args=args)
        self.trainloader = DataLoader(train_dataset, args.batch_sz, shuffle=True, collate_fn=collate,
                                          num_workers=args.num_workers)
        self.testloader = DataLoader(test_dataset, args.batch_sz, collate_fn=collate, num_workers=args.num_workers)
        self.valloader = DataLoader(val_dataset, args.batch_sz, collate_fn=collate, num_workers=args.num_workers)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.args = args
        self.batch_size = args.batch_sz
        self.end_epoch = args.epochs

    def train(self, start_epoch):
        ckpt_path = f"{self.args.save_model_dir}/tbmc_trainner/{self.args.expid}"
        os.makedirs(ckpt_path, exist_ok=True)
        if self.args.resume:
            checkpoint = load_checkpoint(ckpt_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optm.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"load model success.\tepoch:{start_epoch}\tloss:{loss:.6f}")

        loss = self.valid(start_epoch)
        best_acc,best_epoch =0,0
        
        for epoch in range(start_epoch, self.end_epoch):
            loss = self.train_epoch(epoch)         

            with torch.no_grad():
                total_acc = self.valid(start_epoch)
            if total_acc > best_acc: 
                best_epoch = epoch
                best_acc = total_acc
                torch.save(self.model.state_dict(),'best.mdl') 
        
            if epoch >= self.end_epoch - 20:
                os.makedirs(f"{self.args.save_model_dir}/{self.args.expid}", exist_ok=True)

        print('best acc:',best_acc,'best_epoch:',best_epoch)
        print('detect the test data!')
        self.model.load_state_dict((torch.load('best.mdl')))
        valid_accuracy = torchmetrics.Accuracy().cuda()
        
        for batchid, minibatchdata in enumerate(self.testloader):
            
            img1, img2, text, mask, segment, gt,path = minibatchdata[3], minibatchdata[4], minibatchdata[0], minibatchdata[2], minibatchdata[1], minibatchdata[5],minibatchdata[6]
            img1, img2, text, mask, segment, gt = img1.cuda(), img2.cuda(), text.cuda(), mask.cuda(), segment.cuda(), gt.cuda()
            out = self.model(text, mask, segment, img1, img2)
            pred = out.softmax(dim=-1)
            acc = valid_accuracy(pred,gt)
            
        total_acc = valid_accuracy.compute()
        print(f'test_acc:',{total_acc})

    def overfitting_test(self):
        sampledata = next(iter(self.trainloader))
        for epoch in range(0, self.end_epoch):
            img1, img2, text, gt= sampledata[3], sampledata[4], sampledata[0], sampledata[5]
            img1, img2, text, gt = img1.cuda(), img2.cuda(), text.cuda(), gt.cuda()
            out = self.model(text, None, None, img1, img2)

            print(f"gt:{gt}")
            print(f"out:{F.softmax(out, dim=0)}")

            loss = self.criterion(out, gt)
            self.optm.zero_grad()
            loss.backward()
            self.optm.step()

            print(
                f"epoch:{epoch}:\n loss:{loss}\n ")

            self.optm_drop.step()

        return sampledata

    def train_epoch(self, epoch):
        if epoch <= 30:
            self.model.freeze = True
        else:
            self.model.freeze = False
        self.model.train()
        loss = 0
        for batchid, minibatchdata in enumerate(self.trainloader):
            img1, img2, text, mask, segment, gt = minibatchdata[3], minibatchdata[4], minibatchdata[0], minibatchdata[2], minibatchdata[1], minibatchdata[5]
            img1, img2, text, mask, segment, gt = img1.cuda(), img2.cuda(), text.cuda(), mask.cuda(), segment.cuda(), gt.cuda()

            out = self.model(text, mask, segment, img1, img2)
            loss = self.criterion(out, gt)
            self.optm.zero_grad()
            loss.backward()
            self.optm.step()

            if batchid % 10 == 0:
                print(f"epoch:{epoch} {batchid}/{len(self.trainloader)}:\n loss:{loss}\n")

        self.optm_drop.step()

        return loss

    def valid(self, epoch):
        self.model.eval()
        
        valid_accuracy = torchmetrics.Accuracy().cuda()
        for batchid, minibatchdata in enumerate(self.valloader):
            
            img1, img2, text, mask, segment, gt = minibatchdata[3], minibatchdata[4], minibatchdata[0], minibatchdata[2], minibatchdata[1], minibatchdata[5]
            img1, img2, text, mask, segment, gt = img1.cuda(), img2.cuda(), text.cuda(), mask.cuda(), segment.cuda(), gt.cuda()
            out = self.model(text, mask, segment, img1, img2)
        
            pred = out.softmax(dim=-1)

            acc = valid_accuracy(pred, gt)

        total_acc = valid_accuracy.compute()
        print(f"total acc:{total_acc}")
        return total_acc
        
        


