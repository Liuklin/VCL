import pdb
import copy
from collections import OrderedDict

import numpy

import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import encoders
import random


from thop import profile

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    # self.feat_attention = attention_block[0](512)

    def forward(self, x):
        #   x=self.feat_attention(x)
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True, dim=512, p_detach=0.75,
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.embedding = nn.Embedding(num_classes,hidden_size)
        self.textencoder = encoders.TransformerEncoder(hidden_size=1024, ff_size=2048, num_layers=1,
                                                       num_heads=8,
                                                       dropout=0.1, emb_dropout=0.1, freeze=False)
        self.cross_attention = encoders.VideoTextTransformerEncoder(hidden_size=1024, ff_size=2048, num_layers=1,
                                                                    num_heads=8,
                                                                    dropout=0.1, emb_dropout=0.1, freeze=False)
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        # self.temporal_model2 = encoders.TransformerEncoder(hidden_size=1024, ff_size=2048, num_layers=2, num_heads=8,
        #                                                   dropout=0.1, emb_dropout=0.1, freeze=False)
        self.fc = nn.Linear(hidden_size, self.num_classes)
        self.fc1 = nn.Linear(hidden_size, self.num_classes)
        self.bottleneck = nn.Linear(hidden_size*2,hidden_size//2)
        self.shared_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size//2, hidden_size=hidden_size,
                                        num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.classifier = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        self.register_backward_hook(self.backward_hook)
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.mlm_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(hidden_size, hidden_size)),
                         ('gelu', QuickGELU()),
                         ('ln', LayerNorm(hidden_size)),
                         ('fc', nn.Linear(hidden_size, 1296))]))

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            a = torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
            return a

         # return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])  # 35*3*224*224
        x = self.conv2d(x)  # 35*512
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                        for idx, lgt in enumerate(len_x)])

        return x

    def forward(self, x, len_x, label=None, label_lgt=None, train_flag=True):
        if len(x.shape) == 5:
             # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            # frame-wise features
            framewise = x  # 2*512*180

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        if train_flag:
            if batch == 2:
                label1 = label[: label_lgt[0]]
                label2 = label[label_lgt[0]:]
                text1 = self.embedding(label1)   #14*1024
                text2 = self.embedding(label2)  #10*1024
                # embeddings1 = []
                # for i in label2:
                #     word = self.i2g_dict.get(i.item())
                #     embeddings1.append(word)
                if label_lgt[0] > label_lgt[1]:
                    for i in range(label_lgt[0]-label_lgt[1]):
                        text2 = torch.cat([text2, torch.zeros(1024).unsqueeze(dim=0).cuda()], dim=0)#text2 10*1024->14*1024
                    tlength0 = label_lgt[0]
                    tlength1 = label_lgt[1]
                    tmask1 = torch.zeros(1, 1, tlength0).cuda()
                    tmask2 = torch.zeros(1, 1, tlength0).cuda()
                    tmask2[:, :, tlength1:tlength0] = 1
                    tmask = torch.cat([tmask1, tmask2], dim=0)
                    tmask = tmask.bool()
                else:
                    for i in range(label_lgt[1] - label_lgt[0]):
                        text1 = torch.cat([text1, torch.zeros(1024).unsqueeze(dim=0).cuda()], dim=0)
                    tlength0 = label_lgt[1]
                    tlength1 = label_lgt[0]
                    tmask1 = torch.zeros(1, 1, tlength0).cuda()
                    tmask2 = torch.zeros(1, 1, tlength0).cuda()
                    tmask1[:, :, tlength1:tlength0] = 1
                    tmask = torch.cat([tmask1, tmask2], dim=0)
                    tmask = tmask.bool()
                text = torch.stack([text1, text2], dim=0)
                text = self.textencoder(text, label_lgt, tmask)
                text = text[0]
                # t_en = self.textencoder(text, label_lgt, None)
                # t_en = t_en[0]
                # #x_attention = self.cross_attention(x.reshape(batch, -1, 1024), t_en, tmask)
                x_attention = self.cross_attention(x.reshape(batch, -1, 1024), text, tmask)
                x_cross = x_attention[0]
                # x_en = self.textencoder(x_cross, lgt, None)
                # x_en = x_en[0]
                # x_cross = x_en.transpose(0, 1)
                x_cross = x_cross.transpose(0, 1)

                mask_index1 = random.sample(range(len(label1) - 1), int(0.15*((len(label1) - 1))))
                mlm_mask1 = tmask1
                mask_index2 = random.sample(range(len(label2) - 1), int(0.15 * ((len(label2) - 1))))
                mlm_mask2 = tmask2
                for index in mask_index1:
                    mlm_mask1[..., index] = 1
                for index in mask_index2:
                    mlm_mask2[..., index] = 1
                mlm_mask = torch.cat([mlm_mask1, mlm_mask2], dim=0)
                mlm_mask = mlm_mask.bool()
                mlm_feats = self.textencoder(text, label_lgt, mlm_mask)
                mlm_feats = mlm_feats[0]
                vlength0 = torch.round(lgt[0]).int()
                vlength1 = torch.round(lgt[1]).int()
                vmask1 = torch.zeros(1, 1, vlength0).cuda()
                vmask2 = torch.zeros(1, 1, vlength0).cuda()
                vmask2[:, :, vlength1:vlength0] = 1
                vmask = torch.cat([vmask1, vmask2], dim=0)
                vmask = vmask.bool()
                mlm_attention = self.cross_attention(mlm_feats, x.reshape(batch, -1, 1024), vmask)
                mlm_cross = mlm_attention[0]
                # mlm_en = self.textencoder(mlm_cross, label_lgt, None)
                # mlm_en = mlm_en[0]
                # mlm_out = self.mlm_head(mlm_en)
                mlm_out = self.mlm_head(mlm_cross)
                text1_o = mlm_out[0]
                text1_out = text1_o[: label_lgt[0]]
                text2_o = mlm_out[1]
                text2_out = text2_o[: label_lgt[1]]
                mlm_loss = self.loss['MLM'](text1_out, label1) + self.loss['MLM'](text2_out, label2)


            else:
                text = self.embedding(label)
                tlength0 = label_lgt[0]
                tmask = torch.zeros(1, 1, tlength0).cuda()
                tmask = tmask.bool()
                x_cross = x
                mlm_out = self.mlm_head(text)
                mlm_loss = self.loss['MLM'](mlm_out, label)

        # cross_outputs = self.fc(x)
        cross_outputs = self.fc(x_cross)  # loss
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])  # loss

        # tm_attention = self.cross_attention(tm_outputs['predictions'].reshape(batch, -1, 1024), text, tmask)
        # tm_cross = tm_attention[0]
        # tm_cross = tm_cross.transpose(0, 1)
        # tmcross_outputs = self.fc1(tm_cross)

        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "cross_logits": cross_outputs,
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            "mlm_loss":  mlm_loss,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        ctc_cross = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                ctc_auxi = self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                label_lgt.cpu().int()).mean()
                loss += weight * ctc_auxi
            elif k == 'CroCTC':
                ctc_cross = self.loss['CTCLoss'](ret_dict["cross_logits"].log_softmax(-1),
                                                 label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                 label_lgt.cpu().int()).mean()
                loss += weight * ctc_cross
            elif k == 'SeqCTC':
                ctc_main = self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                label_lgt.cpu().int()).mean()
                loss += weight * ctc_main
            elif k == 'MLM':
                mlm_loss = ret_dict["mlm_loss"]
                loss += weight * mlm_loss
            # elif k == 'Dist':
            #     loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
            #                                                ret_dict["sequence_logits"].detach(),
            #                                                use_blank=False)
        return loss, ctc_auxi, ctc_main, ctc_cross

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['MLM'] = torch.nn.CrossEntropyLoss()
        # self.loss['distillation'] = SeqKD(T=8)
        return self.loss