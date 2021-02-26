import time
import torch
import numpy as np
import pdb

def precision_recall(gt,infer,tpi_memory):
    plist = []
    rlist = []
    for _gt,_inf in zip(gt,infer): #infer has been predicted based on file index
        _inf = _inf.cpu().numpy()
        _replaced_prediction = tpi_memory[_inf]
        #print(_gt)
        #print(_inf)
        _tp = np.sum(_replaced_prediction==_gt)
        _total = np.sum(tpi_memory==_gt)
        plist.append(_tp/len(np.atleast_1d(_inf)))
        rlist.append(_tp/_total)
    return np.mean(np.array(plist)),np.mean(np.array(rlist))

