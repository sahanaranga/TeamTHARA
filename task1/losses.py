import torch
import torch.nn as nn

'''
Other loss functions that were explored during experimentation
'''

class WeightedBCELoss(nn.Module):
    def __init__(self, y_train):
        super(WeightedBCELoss, self).__init__()
        self.y_train = y_train
        self.pos_weight = self.weighted_factors()

    def forward(self, logits, labels):
        loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight, reduction='mean')
        return loss(logits, labels)

    def weighted_factors(self):
        f = torch.sum(self.y_train, dim=0)
        N = len(self.y_train)
        K = torch.full(size=f.size(), fill_value=N)
        pos_weight = torch.div(torch.sub(K, f), f).cuda()
        return pos_weight
    



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        #self.alpha = alpha
        self.gamma = gamma
        self.BCE = torch.nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        BCE_loss = self.BCE(inputs,targets)
        pt = torch.exp(-BCE_loss)
        #F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        F_loss =  (1-pt)**self.gamma * BCE_loss
        

        return F_loss


