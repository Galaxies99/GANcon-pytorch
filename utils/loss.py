import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, with_softmax = True):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.with_softmax = with_softmax
        if self.with_softmax:
            self.cross_entropy = nn.CrossEntropyLoss(reduction = 'none')
        else:
            self.cross_entropy = nn.NLLLoss(reduction = 'none')

    def forward(self, res, gt, mask):
        if self.with_softmax is False:
            res = torch.log(res)
        loss = self.cross_entropy(res, gt.long())
        loss = loss * mask
        sample_loss = loss.sum(dim = [1, 2]) / mask.sum(dim = [1, 2])
        mean_batch_loss = torch.mean(sample_loss)
        return mean_batch_loss


class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma = 2.0, with_softmax = True):
        super(MaskedFocalLoss, self).__init__()
        self.with_softmax = with_softmax
        self.alpha = alpha
        self.gamma = gamma
        if self.with_softmax:
            self.cross_entropy = nn.CrossEntropyLoss(reduction = 'none')
        else:
            self.cross_entropy = nn.NLLLoss(reduction = 'none')
        
    def fowrard(self, res, gt, mask):
        if self.with_softmax is False:
            res = torch.log(res)
        loss = self.cross_entropy(res, gt.long())
        pt = torch.exp(loss)
        loss = loss * mask
        loss = loss * self.alpha[gt] * torch.pow(1 - pt, self.gamma)
        sample_loss = loss.sum(dim = [1, 2]) / mask.sum(dim = [1, 2])
        mean_batch_loss = torch.mean(sample_loss)
        return mean_batch_loss


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, res, gt, mask):
        loss = self.mse(res, F.one_hot(gt, num_classes = 10))
        loss = loss * mask
        sample_loss = loss.sum(dim = [1, 2]) / mask.sum(dim = [1, 2])
        mean_batch_loss = torch.mean(sample_loss)
        return mean_batch_loss


class MaskedSFLoss(nn.Module):
    def __init__(self, alpha, beta = 1.0, gamma = 2.0):
        super(MaskedSFLoss, self).__init__()
        self.beta = beta
        self.focal = MaskedFocalLoss(alpha, gamma)
        self.mse = MaskedMSELoss()
    
    def forward(self, res, gt, mask):
        loss1 = self.focal(res, gt, mask)
        loss2 = self.mse(res, gt, mask)
        return self.beta * loss1 + loss2


class DiscriminateLoss(nn.Module):
    def __init__(self):
        super(DiscriminateLoss, self).__init__()
    
    def forward(self, prediction, mask):
        loss = - torch.log(prediction)
        loss = loss * mask
        sample_loss = loss.sum(dim = [1, 2]) / mask.sum(dim = [1, 2])
        mean_batch_loss = torch.mean(sample_loss)
        return mean_batch_loss


class GeneratorLoss(nn.Module):
    def __init__(self, alpha_, beta_ = 1.0, gamma_ = 2.0, lambda_ = 1.0):
        super(GeneratorLoss, self).__init__()
        self.lambda_ = lambda_
        self.discriminate_loss = DiscriminateLoss()
        self.sf_loss = MaskedSFLoss(alpha_, beta_, gamma_)
    
    def forward(self, prediction, res, gt, mask):
        loss1 = self.discriminate_loss(prediction, mask)
        loss2 = self.sf_loss(res, gt, mask)   
        return loss1 * self.lambda_ + loss2


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.loss = nn.BCELoss(reduction = 'none')
    
    def forward(self, res, gt, mask):
        loss = self.loss(res, gt)
        loss = loss * mask
        sample_loss = loss.sum(dim = [1, 2]) / mask.sum(dim = [1, 2])
        mean_batch_loss = torch.mean(sample_loss)
        return mean_batch_loss
