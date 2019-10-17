#!/usr/bin/env python3
"""logistic_regression
James Gardner 2019

performs logistic regression on feature vectors
against positional matching labels
"""

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn.functional as F

def load_catalogue():
    catalogue = pd.read_csv('patch_catalogue.csv')
    catalogue.set_index(['name_TGSS','name_NVSS'],inplace=True)

    scores = catalogue['score']
    del (catalogue['ra_TGSS'],catalogue['dec_TGSS'],
         catalogue['ra_NVSS'],catalogue['dec_NVSS'],
         catalogue['score'])

    catalogue['log_flux_TGSS']       = np.log10(catalogue['peak_TGSS'])
    catalogue['log_integrated_TGSS'] = np.log10(catalogue['integrated_TGSS'])
    catalogue['log_ratio_flux_TGSS'] = np.log10(catalogue['peak_TGSS']/
                                                catalogue['integrated_TGSS'])
    catalogue['log_flux_NVSS']       = np.log10(catalogue['peak_NVSS'])

    labels = (scores.values > 0.1)
    features = catalogue.values

    labels = Variable(torch.from_numpy(labels).float())
    features = Variable(torch.Tensor(features))
    
    return labels, features, catalogue.columns, catalogue

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        
    def forward(self, x):
        outputs = F.sigmoid(self.linear(x))
        return outputs

def logistic_regress(labels, features, input_dim, learning_rate, num_epochs):
    model = LogisticRegression(input_dim)
    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in tqdm(range(num_epochs)):
        # reset gradient accumulation
        optimizer.zero_grad()
        # forward step
        predictions = model(features)
        loss = criterion(predictions, labels)
        losses.append(loss.item())
        # backwards step
        loss.backward()
        optimizer.step()    

    return model, losses
        
def plot_regression(model,losses,ticks):
    plt.figure(figsize=(14,7))
    plt.rcParams.update({'font.size': 18})
    plt.plot(losses)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('torch_lr - losses showing stabilisation')
    plt.savefig('torch_lr_losses.pdf',bbox_inches='tight') 
    plt.clf()

    weights = list(model.parameters())
    weights = weights[0].detach().numpy().ravel()
    
    plt.figure(figsize=(14,7))
    plt.rcParams.update({'font.size': 18})
    plt.bar(range(len(weights)),weights)
    plt.xlabel('weights')
    plt.xticks(range(len(weights)),ticks,rotation='vertical')
    plt.ylabel('co-eff')
    plt.title('torch_lr - weights')
    plt.savefig('torch_lr.pdf',bbox_inches='tight') 
    plt.clf()

def multi_object_maker(catalogue):
    # partition the sky
    cat_pairs = set(catalogue.index.values)
    obj_pairs = []

    for pair in tqdm(cat_pairs):
        if catalogue.loc[pair]['pred_labels'] == 1:
            obj_pairs.append(pair)   

    objects = {}
    tnames = {}
    nnames = {}

    index = 0
    for pair in tqdm(obj_pairs):
        tname, nname = pair[0], pair[1]

        if not tname in tnames and not nname in nnames:
            i = index
            objects[i] = [tname,nname]
            tnames[tname] = i
            nnames[nname] = i
        elif tname in tnames and not nname in nnames:
            i = tnames[tname]
            objects[i].append(nname)
            nnames[nname] = i
        elif not tname in tnames and nname in nnames:
            i = nnames[nname]
            objects[i].append(tname)
            tnames[tname] = i
        elif tname in tnames and nname in nnames:
            # must merge objects, zig-zag problem
            i = tnames[tname]
            j = nnames[nname]
            if i == j:
                continue
            else:
                obj_i = objects[i]
                obj_j = objects[j]
                merged_obj = list(set(obj_i+obj_j))
                objects[index] = merged_obj
                del objects[i], objects[j] 
                for name in merged_obj:
                    if   name[0] == 'T':
                        tnames[name] = index
                    elif name[0] == 'N':
                        nnames[name] = index

        index += 1

    multi_objects = {}
    for key, val in objects.items():
        if len(val) > 2:
            multi_objects[key] = val

    return multi_objects
    
def main():
    labels, features, ticks, catalogue = load_catalogue()

    # hyper-parameters
    input_dim = features.shape[1]
    learning_rate = 0.001
    num_epochs = 1000

    model, losses = logistic_regress(labels, features, input_dim, learning_rate, num_epochs)
    
    predictions = model(features).detach().numpy()
    pred_labels = (predictions > 0.5).astype(float)
    accuracy = (pred_labels == labels.numpy()).mean()    
    print(accuracy)
        
    plot_regression(model,losses,ticks)
    
    catalogue['pred_labels'] = pred_labels
    multi_object_maker(catalogue)
    
if __name__ == "__main__":
    main()
