from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet, resnext
import numpy as np
from core.anchors import generate_default_anchor_maps, hard_nms
from config import CAT_NUM, PROPOSAL_NUM, PART_IMAGE_SIZE, PAD_SIZE, CLASSES, INPUT_SIZE


class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        #print("t1, t2, t3 = ", t1.shape, t2.shape, t3.shape)

        return torch.cat((t1, t2, t3), dim=1)


class attention_net(nn.Module):
    def __init__(self, topN=4):
        super(attention_net, self).__init__()
        # ResNet family
        #self.pretrained_model = resnet.resnet50(pretrained=True)
        #self.pretrained_model = resnet.resnet101(pretrained=True)
        #self.pretrained_model = resnet.resnet152(pretrained=True)

        # ResNext
        self.pretrained_model = resnext.resnext50_32x4d(pretrained=True)
        #self.pretrained_model = resnext.resnext101_32x8d(pretrained=True)

        
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, CLASSES)

        # initiate FPN, Scrutinizer, Teacher net
        self.proposal_net = ProposalNet()
        self.topN = topN
        self.concat_net = nn.Linear(2048 * (CAT_NUM + 1), CLASSES)
        self.partcls_net = nn.Linear(512 * 4, CLASSES)

        # use edge_anchors
        _, edge_anchors, _ = generate_default_anchor_maps()

        self.pad_side = PAD_SIZE
        self.edge_anchors = (edge_anchors + PART_IMAGE_SIZE).astype(np.int)

    def forward(self, x):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)

        """resnet50
        - resnet_out = torch.Size([16, 209]) 
        - rpn_feature = torch.Size([16, 2048, 7, 7])
        - feature = torch.Size([16, 2048])
        """
        """resnet152
        - resnet_out = torch.Size([16, 209]) 
        - rpn_feature = torch.Size([16, 2048, 7, 7])
        - feature = torch.Size([16, 2048])
        """
        """resnext50 with BATCH = 32
        - resnet_out = torch.Size([16, 209]) 
        - rpn_feature = torch.Size([16, 2048, 7, 7])
        - feature = torch.Size([16, 2048])
        """
        """resnext101 with BATCH = 16
        - resnet_out = torch.Size([8, 209]) 
        - rpn_feature = torch.Size([8, 2048, 7, 7])
        - feature = torch.Size([8, 2048])
        """
        #print("resnet_out, rpn_feature, feature =", resnet_out.shape, rpn_feature.shape, feature.shape)
        
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)

        # we will reshape rpn to shape: batch * nb_anchor
        rpn_score = self.proposal_net(rpn_feature.detach())

        """resnet50
        - rpn_score = torch.Size([16, 426]) <class 'torch.Tensor'>
        - edge_anchor = (426, 4) <class 'numpy.ndarray'>
        """
        """resnet152
        - rpn_score = torch.Size([16, 426]) <class 'torch.Tensor'>
        - edge_anchor = (426, 4) <class 'numpy.ndarray'>
        """
        """resnext50
        - rpn_score = torch.Size([16, 426]) <class 'torch.Tensor'>
        - edge_anchor = (426, 4) <class 'numpy.ndarray'>
        """
        """resnext101
        - rpn_score = torch.Size([8, 426]) <class 'torch.Tensor'>
        - edge_anchor = (426, 4) <class 'numpy.ndarray'>
        """
        #print("debug, rpn_score=", rpn_score.size(), type(rpn_score))
        #print("edge_anchor=", self.edge_anchors.shape, type(self.edge_anchors))

        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]

        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).cuda()
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        part_imgs = torch.zeros([batch, self.topN, 3, PART_IMAGE_SIZE, PART_IMAGE_SIZE]).cuda()

        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i : i + 1, :, y0 : y1, x0 : x1],
                        size=(PART_IMAGE_SIZE, PART_IMAGE_SIZE), mode='bilinear', align_corners=True)

        #part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        part_imgs = part_imgs.view(batch * self.topN, 3, PART_IMAGE_SIZE, PART_IMAGE_SIZE)
        _, _, part_features = self.pretrained_model(part_imgs.detach())

        """resnet50
        - part_features = torch.Size([96, 2048])
        """
        """resnet152
        - part_features= torch.Size([96, 2048]) 
        """
        """resnext50
        - part_features= torch.Size([96, 2048]) 
        """
        """resnext101
        - part_features= torch.Size([48, 2048]) 
        """
        #print("part_features=", part_features.size())

        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)

        # concat_logits have the shape: B*200/209
        concat_out = torch.cat([part_feature, feature], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out

        # part_logits have the shape: B*topN*200/209
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob]



"""
loss functions
"""
def list_loss(logits, targets):
    """
    - Args:
        - logit: logits, (batch * num_proposals, classes)
        - targets: tensor of labels, (batch * num_proposals, )
    """
    temp = F.log_softmax(logits, -1)                # (batch * num_proposals, classes)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def list_loss_with_smoothed_targets(logits, smoothed_labels):
    """Teaching loss with smoothed one-hot encoded targets 
    - Args:
        - logit: logits, (batch * num_proposals, classes)
        - smoothed_labels: tensor of labels, (batch * num_proposals, classes)
    """
    # logC(regions -> gt label)
    temp = F.log_softmax(logits, -1)                # (batch * num_proposals, classes)
    targets = torch.argmax(smoothed_labels, axis=-1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    """
    - Args:
        - score:
        - targets:
        - proposal_num:
    """
    loss = Variable(torch.zeros(1).cuda())
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size
