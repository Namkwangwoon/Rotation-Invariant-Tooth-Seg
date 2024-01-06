import torch
import torch.nn.functional as F

from torchvision.ops import sigmoid_focal_loss


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    def forward(self, pred, target):
        #target = F.one_hot(target, num_classes=10)
        #pred = pred.permute(0,2,1)
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def tooth_class_loss(cls_pred, gt_cls, cls_num, weight=None, label_smoothing=None):
    """
    Input
        cls_pred: 1, 17, 16000
        gt_cls: 1, 1, 16000 -> -1 is background, 0~15 is foreground
    """
    # B, _, N = gt_cls.shape
    # gt_cls = gt_cls.view(B, -1)
    gt_cls = gt_cls.type(torch.long)
    gt_cls = gt_cls.squeeze()
    
    
    if label_smoothing is None:
        if weight is None:
            loss = torch.nn.CrossEntropyLoss().type(torch.float).cuda()(cls_pred, gt_cls)
        else:
            loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight).type(torch.float).cuda())(cls_pred, gt_cls)
    else:
        loss = LabelSmoothingLoss(cls_num, smoothing=label_smoothing)(cls_pred, gt_cls)
    return loss


def tooth_class_loss_focal(cls_pred, gt_cls):
    """
    Input
        cls_pred: 1, 17, 16000
        gt_cls: 1, 1, 16000 -> -1 is background, 0~15 is foreground
    """
    # B, _, N = gt_cls.shape
    # gt_cls = gt_cls.view(B, -1)
    gt_cls = gt_cls.type(torch.long)
    gt_cls = gt_cls.squeeze()
    gt_cls = F.one_hot(gt_cls, num_classes=2).type(torch.float32).detach()
    
    loss = sigmoid_focal_loss(cls_pred, gt_cls, reduction='mean')
    
    return loss