# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 16:56:22 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
import tensorflow as tf
import numpy as np


def cos_lr(max_lr, min_lr, warm, end):
    def lr(step):
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(step / end * np.pi))
    
    def warmup(step):
        return (max_lr - lr(end))/(warm)*step + lr(end)
    
    l1 = [warmup(i) for i in range(warm)]
    l2 = [lr(i) for i in range(warm, end)]
    l3 = [l2[-1] for _ in range(warm)] # Maintain a constant learning rate at the end as in original
    return l1 + l2 + l3

    
    
# This function is designed for hierarchical prediction (superclass mapping).
# For HADNN1 with SLE data, we are only predicting coordinates, so this function's
# primary use case (mapping to building/floor based on coordinate prediction) 
# is not directly applicable in its full form.
# However, if the `train.py` calls it, it needs to be adapted.
# Given that `n_hierarchy` will be 1, `mapper_superclass` might be simplified or bypassed.
# In `train.py`, `logits` is passed as a list, and `mapper_superclass` is used when `n_hierarchy > 1`.
# Since `n_hierarchy` is now 1, `mapper_superclass` will not be called in a way that
# requires it to infer building/floor from the coordinate prediction.
# The `train_step` functions in `train.py` now directly return `[logit]` where `logit`
# is the coordinate prediction, ensuring compatibility.
# I'll keep the function signature for `mapper_superclass` as is for consistency,
# but its full functionality for hierarchical mapping won't be utilized with `n_hierarchy=1`.
def mapper_superclass(true, logit, h_info):
    all_predict = []
    # The first element is always the main prediction (coordinates in this case)
    all_predict.append(logit) 
    
    # The rest of the original function's logic for inferring hierarchical labels 
    # (building, floor) based on the predicted coordinate is not applicable when
    # n_hierarchy is 1 and we are only doing coordinate prediction.
    # The loop 'for i in range(len(true)-1)' will not run if len(true) is 1.
    # Therefore, this function effectively just passes through the `logit` for coordinate prediction.
    
    return all_predict