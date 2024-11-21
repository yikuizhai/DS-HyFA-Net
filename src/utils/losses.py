import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss_novel(nn.Module):
    def __init__(self, critn, coeffs=1.0):
        super().__init__()
        self.critn = critn
        self.coeffs = coeffs

    def forward(self, preds, tar):
        if isinstance(self.coeffs, float):
            coeffs = [self.coeffs]*len(preds)
        else:
            coeffs = self.coeffs
        if len(coeffs) != len(preds):
            raise ValueError
        loss = 0.0
        tar_aux = (tar.type(torch.int) ^ 1).type(torch.float32)
        tars = [tar, tar_aux]
        for coeff, pred, tar in zip(coeffs, preds, tars):
                loss += coeff * self.critn(pred, tar)
        return loss

class BCELoss(nn.Module):
    def __init__(self, critn, coeffs=1.0):
        super().__init__()
        self.critn = critn
        self.coeffs = coeffs

    def forward(self, pred, tar):
        if isinstance(self.coeffs, float):
            coeffs = [self.coeffs]*len(pred)
        else:
            coeffs = self.coeffs
        if len(coeffs) != len(pred):
            raise ValueError
        loss = 0.0
        loss += coeffs * self.critn(pred, tar)
        return loss

class MixedLoss(nn.Module):
    def __init__(self, critns, coeffs=1.0):
        super().__init__()
        self.critns = critns
        if isinstance(coeffs, float):
            coeffs = [coeffs]*len(critns)
        if len(coeffs) != len(critns):
            raise ValueError
        self.coeffs = coeffs

    def forward(self, pred, tar):
        loss = 0.0
        for critn, coeff in zip(self.critns, self.coeffs):
            loss += coeff * critn(pred, tar)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, critn, coeffs=1.0):
        super().__init__()
        self.critn = critn
        self.coeffs = coeffs

    def forward(self, preds, tar):
        if isinstance(self.coeffs, float):
            coeffs = [self.coeffs]*len(preds)
        else:
            coeffs = self.coeffs
        if len(coeffs) != len(preds):
            raise ValueError
        loss = 0.0
        for coeff, pred in zip(coeffs, preds):
            loss += coeff * self.critn(pred, tar)
        return loss

# Refer to https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2):
        super().__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, pred, tar):
        pred, tar = pred.flatten(1), tar.flatten(1)
        prob = F.sigmoid(pred)

        num = (prob*tar).sum(1) + self.smooth
        den = (prob.pow(self.p) + tar.pow(self.p)).sum(1) + self.smooth

        loss = 1 - num/den
        
        return loss.mean()

class BCLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.m = margin
        self.eps = 1e-4

    def forward(self, pred, tar):
        utar = 1-tar

        n_u = utar.sum() + self.eps
        n_c = tar.sum() + self.eps
        loss = 0.5*torch.sum(utar*torch.pow(pred, 2)) / n_u + \
            0.5*torch.sum(tar*torch.pow(torch.clamp(self.m-pred, min=0.0), 2)) / n_c
        return loss
