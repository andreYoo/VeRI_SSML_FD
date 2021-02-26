import torch
from torch import nn

device_0 = torch.device('cuda:0')
device_1 = torch.device('cuda:1')

class DTL(nn.Module): #Dictionary-based triplet loss
    def __init__(self, delta=0.2, r=0.01,m=0.4, mlp_eval=False):
        super(DTL, self).__init__()
        self.delta = delta # coefficient for mmcl
        self.r = r         # hard negative mining ratio
        self.mlp_eval = mlp_eval
        self.margin = m
        

    def forward(self, inputs, targets,quad=False,is_vec=False):
        m, n = inputs.size()

        if is_vec:
            multiclass = targets
        else:
            targets = torch.unsqueeze(targets, 1)
            multiclass = torch.zeros(inputs.size()).cuda()
            multiclass.scatter_(1, targets, float(1))

        loss = []
        num_pos = 0
        num_hard_neg = 0
        for i in range(m):
            logit = inputs[i].to(device_0)
            label = multiclass[i].to(device_0)

            if quad==True:
                hard_pos_logit = torch.masked_select(logit, label > 1.5)
                soft_pos_logit = torch.masked_select(logit, label >0.5)
                neg_logit = torch.masked_select(logit, label < 0.5)
            else:
                pos_logit = torch.masked_select(logit, label > 0.5)
                neg_logit = torch.masked_select(logit, label < 0.5)

            _, idx = torch.sort(neg_logit.detach().clone(), descending=True)
            num = int(self.r * neg_logit.size(0))
            mask = torch.zeros(neg_logit.size(), dtype=torch.bool).cuda()
            mask[idx[0:num]] = 1
            hard_neg_logit = torch.masked_select(neg_logit, mask)
            if quad==True:
                num_pos += len(hard_pos_logit)
                num_pos += len(soft_pos_logit)
            else:
                num_pos += len(pos_logit)
            num_hard_neg += num
            #print('portion of [total length - %d] pos:neg = %d/%d'%(len(inputs[i]),len(pos_logit),num))
            if quad==True:
                l1 = self.delta * torch.mean((1-hard_pos_logit).pow(2)) + torch.mean((1+hard_neg_logit).pow(2))
                l2 = self.delta * torch.mean((1-soft_pos_logit).pow(2))
                l = l1+l2
            else:
                l = torch.mean((1-pos_logit).pow(2)) + self.delta*torch.mean((1+hard_neg_logit).pow(2))
            loss.append(l)

        loss = torch.mean(torch.stack(loss))
        #print('portion of pos:neg = %.3f / %.3f'%(num_pos/m,num_hard_neg/m))
        if self.mlp_eval==True:
            return loss, num_pos/m # to eval MLP methods
        else:
            return loss
