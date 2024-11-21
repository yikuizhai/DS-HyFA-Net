import torch
import torch.nn as nn
import torch.nn.functional as F
from models._blocks import Conv1x1, Conv3x3, MaxPool2x2
from models._common import CBAM
from src.models.backbones.resnet import resnet18

class E_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True)

    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        return F.relu(x + y)

class Encoder2(nn.Module):
    def __init__(self, in_ch, enc_chs):
        super().__init__()
        self.conv1 = E_Block(2*in_ch, enc_chs[0])
        self.pool1 = MaxPool2x2()

        self.conv2 = E_Block(enc_chs[0], enc_chs[1])
        self.pool2 = MaxPool2x2()

        self.conv3 = E_Block(enc_chs[1], enc_chs[2])
        self.pool3 = MaxPool2x2()

        self.conv4 = E_Block(enc_chs[2], enc_chs[3])
        self.pool4 = MaxPool2x2()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # ch = 6
        Comm_feat1 = self.conv1(x)
        Comm_feat1 = self.pool1(Comm_feat1)
        Comm_feat2 = self.pool2(self.conv2(Comm_feat1))
        Comm_feat3 = self.pool3(self.conv3(Comm_feat2))
        Comm_feat4 = self.pool4(self.conv4(Comm_feat3))
        Comm_feats = [Comm_feat2, Comm_feat3, Comm_feat4]
        return Comm_feats

class A_Block(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch1+in_ch2, out_ch, norm=True, act=True)
        self.cbam = CBAM(out_ch)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[2:])
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        y = self.cbam(x)
        return F.relu(x+y)

class HyFAM(nn.Module):
    def __init__(self, itm_ch, enc_chs, dec_chs):
        super().__init__()
        enc_chs = enc_chs[::-1]
        self.conv = Conv3x3(enc_chs[0]*3, enc_chs[0]*3, norm=True, act=True)
        self.A_Block1 = A_Block(enc_chs[0]*3, enc_chs[0]*3, dec_chs[0])
        self.A_Block2 = A_Block(enc_chs[1]*3, dec_chs[0], dec_chs[1])
        self.A_Block3 = A_Block(enc_chs[2]*3, dec_chs[1], dec_chs[2])
        self.A_Block4 = A_Block(itm_ch, dec_chs[2], dec_chs[3])
    def forward(self, t, feats):
        x = self.conv(feats[2])
        y = self.A_Block1(x, feats[2])  #dec channel 32
        y = self.A_Block2(feats[1], y) # 64
        y = self.A_Block3(feats[0], y) # 128
        y = self.A_Block4(t, y) # 256
        return y

class Preditor(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = Conv1x1(in_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        return x

class Decoder(nn.Module):
    def __init__(self, itm_ch, enc_chs, dec_chs):
        super().__init__()
        self.HyFAM = HyFAM(itm_ch, enc_chs, dec_chs)
        self.preditor = Preditor(dec_chs[3])

    def forward(self, t, feats):
        y = self.HyFAM(t, feats)
        y = self.preditor(y)
        return y

class DSHyFANet(nn.Module):
    def __init__(self, in_ch=3, enc_chs_d=(64, 128, 256), enc_chs_c=(32, 64, 128, 256), dec_chs=(32, 64, 128, 256), AUX=True):
        super().__init__()

        self.Encoder2 = Encoder2(in_ch, enc_chs_c)
        self.Encoder1 = resnet18(pretrained=True)
        self.Encoder1.layer4 = nn.Identity()

        self.Decoder = Decoder(itm_ch=6, enc_chs=enc_chs_d, dec_chs=dec_chs)
        self.AUX = AUX
        if self.AUX:
            self.conv_out = Conv1x1(in_ch=enc_chs_c[3], out_ch=1, norm=True)


    def forward(self, t1, t2):

        dFeat0A =self.Encoder1.conv1(t1)
        dFeat0A =self.Encoder1.bn1(dFeat0A)
        dFeat0A =self.Encoder1.relu(dFeat0A)
        dFeat1A =self.Encoder1.maxpool(dFeat0A)
        dFeat1A =self.Encoder1.layer1(dFeat1A)
        dFeat2A =self.Encoder1.layer2(dFeat1A)
        dFeat3A =self.Encoder1.layer3(dFeat2A)

        dFeat0B =self.Encoder1.conv1(t2)
        dFeat0B =self.Encoder1.bn1(dFeat0B)
        dFeat0B =self.Encoder1.relu(dFeat0B)
        dFeat1B =self.Encoder1.maxpool(dFeat0B)
        dFeat1B =self.Encoder1.layer1(dFeat1B)
        dFeat2B =self.Encoder1.layer2(dFeat1B)
        dFeat3B =self.Encoder1.layer3(dFeat2B)

        cFeats = self.Encoder2(t1, t2)

        feat0 = torch.cat([dFeat1A, dFeat1B, cFeats[0]], dim=1)
        feat1 = torch.cat([dFeat2A, dFeat2B, cFeats[1]], dim=1)
        feat2 = torch.cat([dFeat3A, dFeat3B, cFeats[2]], dim=1)

        pred_main = self.Decoder(torch.cat((t1, t2), dim=1), [feat0, feat1, feat2])

        if self.AUX:
            pred_aux = self.conv_out(cFeats[2])
            pred_aux = F.interpolate(pred_aux, size=pred_main.shape[2:])
            return pred_main, pred_aux
        else:
            return pred_main


if __name__ == '__main__':
    from torchstat import stat

    t1 = torch.randn(1, 3, 256, 256)
    t2 = torch.randn(1, 3, 256, 256)

    net = DSHyFANet(in_ch=3)

    stat(net, (3, 256, 256))














