import torch
import pdb
import time
import numpy as np
device_1 = torch.device('cuda:1')
device_0 = torch.device('cuda:0')
EVAL_MLP = True

_ADJ_SCALE = 100

class DPLM(object): #Dictionary based sample mining
    def __init__(self, t=0.5,l=10,EVAL_MLP=False) :
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.t = t
        self.topk = l
        self.eval_mlp = EVAL_MLP


    def predict(self,memory, targets):
        memory = memory
        mem_vec = memory[targets]
        #Condition - all features are normalised to |x|=1
        node_sim = mem_vec.mm(memory.t()) #similarity matrix
        node_sim[node_sim<self.t]=-1.0
        
        m, n = node_sim.size() #m is scale of batch, n is the number of images on memory.
        node_sim_sorted, index_sorted = torch.sort(node_sim, dim=1, descending=True)

        nodeclass = torch.zeros(node_sim.size()).to(self.device)
        multilabel = torch.zeros(node_sim.size()).to(self.device)
        mask_num = torch.sum(node_sim_sorted != -1.0, dim=1) #listing candiate using node similiarity.
        _pos_list = []
        for i in range(m):
            topk = int(mask_num[i].item())
            topk = max(topk,self.topk)
            topk_idx = index_sorted[i,:topk]
            g_top_index = index_sorted[i,:_ADJ_SCALE]
            
            s_memory = memory[g_top_index]
            all_sim = s_memory.mm(s_memory.t()).fill_diagonal_(0.0) #matrix for neightborhood similiarty
            n_sim_batch = torch.cdist(all_sim[0].reshape(1,_ADJ_SCALE),all_sim,p=2.0)
            n_sim_sorted, index_n_sim_sorted = torch.sort(n_sim_batch,dim=1,descending=False)
            adj_index = g_top_index[index_n_sim_sorted]
            
            topk_idx_nbased = adj_index[0,0:topk] # adjacenty node similarity based list

            if mask_num[i].item()==1:
                nodeclass[i,targets[i]] = float(1)
                g_part = targets[i]
                g_part = torch.unsqueeze(g_part,0)
            else:
                nodeclass[i,targets[i]] = float(1)
                for j in range(topk):
                    if topk_idx[j] in topk_idx_nbased:continue
                    else:
                        topk_idx[j]=-1
                #print('[%d] similarity %d/%d (%.3f)'%(i,len(tmp),topk,len(tmp)/topk))
                nodeclass[i, topk_idx[topk_idx!=-1]] = float(1)
                g_part = topk_idx[topk_idx!=-1]
                nodeclass[i, targets[i]] = float(1)
            vec = memory[topk_idx]
            sim = vec.mm(memory.t())
            _, idx_sorted = torch.sort(sim.detach().clone(), dim=1, descending=True)
            step = 1
            for j in range(topk):
                pos = torch.nonzero(idx_sorted[j] == index_sorted[i, 0]).item()
                if pos > topk: break
                step = max(step, j)
            step = step + 1
            step = min(step, mask_num[i].item())
            if step <= 0: continue
            multilabel[i, index_sorted[i, 0:step]] = float(1)
            #_pos_list.append(list(set(index_sorted[i,0:step]).intersection(g_part)))
            
            _pos_list.append(torch.unique(torch.cat((index_sorted[i,0:step],g_part),0)))
        targets = torch.unsqueeze(targets, 1)

        mining_results = torch.mul(nodeclass,multilabel)
        #mining_results = nodeclass + multilabel
        #mining_results.scatter_(1, targets, float(1))
        if EVAL_MLP==True:
            return mining_results, _pos_list
        else:
            return mining_results



class KNN(object):
    def __init__(self,k=8,l=10,EVAL_MLP =False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.topk = l
        self.eval_mlp = EVAL_MLP

    def predict(self, memory, targets):
        mem_vec = memory[targets]
        # Condition - all features are normalised to |x|=1
        node_sim = mem_vec.mm(memory.t())  # similarity matrix
        m, n = node_sim.size()  # m is scale of batch, n is the number of images on memory.
        node_sim_sorted, index_sorted = torch.sort(node_sim, dim=1, descending=True)
        nodeclass = torch.zeros(node_sim.size()).to(self.device)
        _pos_list = []
        for i in range(m):
            topk = self.k
            topk_idx = index_sorted[i,:topk]
            nodeclass[i, topk_idx] = float(1)
            _pos_list.append(topk_idx)
        targets = torch.unsqueeze(targets, 1)
        nodeclass.scatter_(1, targets, float(1))
        if self.eval_mlp==True:
            return nodeclass, _pos_list
        else:
            return nodeclass

class SS(object):
    def __init__(self,t=0.6, l=10,EVAL_MLP =False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.t = t
        self.topk = l
        self.eval_mlp = EVAL_MLP

    def predict(self, memory, targets):
        mem_vec = memory[targets]
        # Condition - all features are normalised to |x|=1
        node_sim = mem_vec.mm(memory.t())  # similarity matrix
        node_sim[node_sim<self.t]=-1.0
        m, n = node_sim.size()  # m is scale of batch, n is the number of images on memory.
        node_sim_sorted, index_sorted = torch.sort(node_sim, dim=1, descending=True)
        nodeclass = torch.zeros(node_sim.size()).to(self.device)
        _pos_list = []
        mask_num = torch.sum(node_sim_sorted != -1.0, dim=1)  # listing candiate using node similiarity.
        for i in range(m):
            topk = int(mask_num[i])
            topk = max(topk, self.topk)
            topk_idx = index_sorted[i,:topk]
            _pos_list.append(topk_idx)
            nodeclass[i, topk_idx] = float(1)
        targets = torch.unsqueeze(targets, 1)
        nodeclass.scatter_(1, targets, float(1))
        if self.eval_mlp==True:
            return nodeclass, _pos_list
        else:
            return nodeclass

