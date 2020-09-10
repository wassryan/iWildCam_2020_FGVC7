from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet, resnext
import numpy as np
from core.anchors import generate_default_anchor_maps, hard_nms
# from config import CAT_NUM, PROPOSAL_NUM
from thop import profile

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
        return torch.cat((t1, t2, t3), dim=1)


class attention_net(nn.Module):
    def __init__(self, cfg, CAT_NUM, topN=4):
        super(attention_net, self).__init__()
        if 'resnet' in cfg.NET.BACKBONE:
            self.pretrained_model = resnet.resnet50(pretrained=True)
        elif 'resnext' in cfg.NET.BACKBONE:
            self.pretrained_model = resnext.resnext50_32x4d(pretrained=True)

        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, cfg.NUM_CLASSES) # ! 6 => num_class
        self.proposal_net = ProposalNet()
        self.CAT_NUM = CAT_NUM
        self.topN = topN # PROPOSAL_NUM=6
        self.concat_net = nn.Linear(2048 * (self.CAT_NUM + 1), cfg.NUM_CLASSES) # ! 6 => num_class
        self.partcls_net = nn.Linear(512 * 4, cfg.NUM_CLASSES) # ! 6 => num_class
        _, edge_anchors, _ = generate_default_anchor_maps(input_shape=(cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1]))
        # self.pad_side = 224
        self.pad_side = 112
        # self.edge_anchors = (edge_anchors + 224).astype(np.int)
        self.edge_anchors = (edge_anchors + 112).astype(np.int)

    def forward(self, x):
        resnet_out, rpn_feature, feature = self.pretrained_model(x) # pool+fc的特征，pool前的特征，pool后的特征
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)
        # we will reshape rpn to shape: batch * nb_anchor
        rpn_score = self.proposal_net(rpn_feature.detach())
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).cuda()
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        part_imgs = torch.zeros([batch, self.topN, 3, 112, 112]).cuda()
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                # part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                #                                       align_corners=True) # 把topN个proposal bilinear到224x224
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(112, 112), mode='bilinear',
                                                      align_corners=True) # 把topN个proposal bilinear到224x224
        # part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        part_imgs = part_imgs.view(batch * self.topN, 3, 112, 112)
        _, _, part_features = self.pretrained_model(part_imgs.detach()) # 把proposal重新输入到resnet50中，提pool后的特征
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :self.CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        # concat_logits have the shape: B*200
        concat_out = torch.cat([part_feature, feature], dim=1) # concat part和global的特征
        concat_logits = self.concat_net(concat_out) # 把part+global的特征经过fc层，得到200类的维度
        raw_logits = resnet_out # 把global的特征经过fc层，得到200类的维度
        # part_logits have the shape: B*N*200
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1) # 把part_feature经过fc层，得到200类的维度
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob]


def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets, proposal_num):
    loss = Variable(torch.zeros(1).cuda())
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size

if __name__ == '__main__':
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model = attention_net().to(device)
    from config import INPUT_SIZE
    input = torch.randn((1, 3, INPUT_SIZE[0],INPUT_SIZE[1])).to(device) # N,C,H,W
    total_ops, total_params = profile(model, inputs=(input, ))
    print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
    print("---|---|---")
    print("%s | %.2f | %.2f" % ("NTS-Net", total_params / (1000 ** 2), total_ops / (1000 ** 3)))# 26.3M, 20.47G
    # resnet140M: 9.4M Flops: 140.5M
