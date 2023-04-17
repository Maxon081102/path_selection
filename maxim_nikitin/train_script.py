import os
import re
import torch
import warnings
import pandas as pd
import numpy as np

from torch import nn
from tqdm.notebook import tqdm
from transformers import get_scheduler
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

class networkDataset(Dataset):
    def __init__(self, dt, max_len=161):
        super(networkDataset).__init__()
        self.max_len = max_len
        self.targets = dt.target.to_numpy()
        self.dt = dt
    
    def __getitem__(self, index):
        value = torch.Tensor(self.dt.to_numpy()[index][:805]).float().view(self.max_len, 805 // self.max_len)
        return torch.Tensor(value), self.targets[index]
    
    def __len__(self):
        return len(self.targets)
    
class block(nn.Module):
    def __init__(self, input_size, output_size, dropout_p=0):
        super(block, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ('lin', nn.Linear(input_size, output_size)),
            ('act', nn.ReLU()),
            ('drop', nn.Dropout(dropout_p)),
            ('lnorm', nn.LayerNorm([output_size]))
        ]))
    
    def forward(self, x):
        return self.block(x)
    
class network(nn.Module):
    def __init__(self, labels_count):
        super(network, self).__init__()
        
        self.model = nn.Sequential(OrderedDict([
            ('block1', block(805, 805)),
            ('block2', block(805, 256)),
            ('block3', block(256, 256)),
            ('block4', block(256, 64)),
            ('last_lin', nn.Linear(64, labels_count)),
            ('logsoftmax', nn.LogSoftmax(dim=1))
        ]))
            
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        output = self.model(x)
        return output
    
def collate_fn(batch):
    x_batch = torch.stack([elem[0] for elem in batch])
    target_batch = torch.Tensor([elem[1] for elem in batch]).long()
    return x_batch, target_batch

def split_dataset(d_train, p, k):
    id_test = []
    id_train = []
    for i in range(2):
        ids = np.arange(len(d_train[d_train.target == i]))
        if i == 0:
            ids = np.random.choice(np.arange(len(d_train[d_train.target == i])), size=int(len(ids) / k))
        else:
            ids = ids.tolist() + np.random.choice(ids, size=int(len(ids) * k) - len(ids)).tolist()
            ids = np.array(ids)
        if i == 0:
            mask = np.random.choice([0, 1], size=len(ids) , p=[p, 1 - p])
        else:
            mask = np.random.choice([0, 1], size=len(ids), p=[p, 1 - p])
        id_test.extend(list(set(np.arange(len(d_train[d_train.target == i]))) - set(ids[mask == 0])))
        id_train.extend(ids[mask == 0])
    
    return id_train, id_test

def create_model_and_optimizer(model_class, model_params, device, lr=1e-3, beta1=0.9, beta2=0.999):
    model = model_class(**model_params)
    model = model.to(device)
    
    optimized_params = []
    for param in model.parameters():
        if param.requires_grad:
            optimized_params.append(param)
    optimizer = torch.optim.Adam(optimized_params, lr, [beta1, beta2])
    return model, optimizer

def train(model, opt, loader, criterion):
    model.train()
    losses_tr = []
    for x_batch, target_batch in loader:
        opt.zero_grad()
        x_batch = x_batch.cuda()
        target_batch = target_batch.cuda()
        pred = model(x_batch)
        loss = criterion(pred, target_batch)
        
        loss.backward()
        opt.step()
        losses_tr.append(loss.item())
    
    return model, opt, np.mean(losses_tr)


def val(model, loader, criterion, metric_names=None):
    model.eval()
    losses_val = []
    if metric_names is not None:
        metrics = defaultdict(list)
    with torch.no_grad():
        for x_batch, target_batch in loader:
            x_batch = x_batch.cuda()
            target_batch = target_batch.cuda()
            pred = model(x_batch)
            loss = criterion(pred, target_batch)

            losses_val.append(loss.item())
            
            if metric_names is not None:
                if 'roc-auc' in metric_names:
                    pred_labels = torch.argmax(pred.cpu(), dim=1)
                    metrics['roc-auc'].append(roc_auc_score(target_batch.cpu().numpy(), pred_labels.numpy()))
                if 'precision' in metric_names:
                    pred_labels = torch.argmax(pred.cpu(), dim=1)
                    metrics['precision'].append(precision_score(target_batch.cpu().numpy(), pred_labels.numpy()))
                if 'accuracy' in metric_names:
                    preds = torch.argsort(pred, dim=1, descending=True)
                    for k in metric_names["accuracy"]["top"]:
                        metrics[f'accuracy ~ top#{k}'].append(
                            np.mean([libs_batch[i].item() in preds[i, :k] for i in range(target_batch.shape[0])])
                        )

        if metric_names is not None:
            for name in metrics:
                metrics[name] = np.mean(metrics[name])
    
    return np.mean(losses_val), metrics if metric_names else None

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def learning_loop(
    model,
    optimizer,
    train_loader,
    val_loader,
    criterion,
    scheduler=None,
    min_lr=None,
    epochs=10,
    val_every=1,
    metric_names=None
):  
    for epoch in range(1, epochs+1):
        model, optimizer, loss = train(model, optimizer, train_loader, criterion)

        if not (epoch % val_every):
            loss, metrics_ = val(model, val_loader, criterion, metric_names=metric_names)
            if scheduler:
                try:
                    scheduler.step()
                except:
                    scheduler.step(loss)
        
        if min_lr and get_lr(optimizer) <= min_lr:
            break
    
    return model, optimizer

def get_best_model(d_train, dataloader_train, dataloader_val, device, id_test_network):
    w1 = [1, 0.9, 0.8, 0.7]
    w2 = [1, 0.1, 0.2, 0.3]
    results = []
    w = []
    val_data = torch.Tensor(d_train.iloc[id_test_network].drop(["target"], axis=1).to_numpy()).float()
    val_data = val_data.to(device)
    val_targets = d_train.iloc[id_test_network]['target']
    for i in tqdm(range(len(w1))):
        for j in tqdm(range(len(w2))):
            model, optimizer = create_model_and_optimizer(
                model_class = network,
                model_params = {
                    "labels_count": 2
                },
                lr = 1e-3,
                device = device
            )
            criterion = nn.NLLLoss(weight=torch.Tensor([w2[j], w1[i]]).float().to(device))
            scheduler = get_scheduler("cosine_with_restarts", optimizer, 10, 50)
            model, optimizer = learning_loop(
                model = model,
                optimizer = optimizer,
                train_loader = dataloader_train,
                val_loader = dataloader_val,
                criterion = criterion,
                scheduler = scheduler,
                epochs = 50,
                min_lr = 1e-6,
                val_every = 1
            )
            pred = model(val_data)
            pred = torch.argmax(pred.cpu(), dim=1)
            results.append(roc_auc_score(val_targets, pred.numpy()))
            w.append([w2[j], w1[i]])
    results = torch.Tensor(results)
    best = torch.argmax(results).tolist()
    return w[best]

if __name__ == "__main__":
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    d_train = pd.read_csv("dataset.csv")
    d_test = pd.read_csv("test.csv")
    count = d_train[d_train.target == 1].target.sum()
    w1, w2 = count / len(d_train), (len(d_train) - count) / len(d_train)
    k = np.sqrt(w2 / w1)

    id_train_network, id_test_network = split_dataset(d_train, 0.7, k)

    max_len = 805
    batch_size = 256
    ds_train = networkDataset(d_train.iloc[id_train_network], max_len)
    ds_val = networkDataset(d_train.iloc[id_test_network],  max_len)

    dataloader_train = DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    dataloader_val = DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
    )

    best_network = get_best_model(d_train, dataloader_train, dataloader_val, device, id_test_network)

    model, optimizer = create_model_and_optimizer(
        model_class = network,
        model_params = {
            "labels_count": 2
        },
        lr = 1e-3,
        device = device
    )
    criterion = nn.NLLLoss(weight=torch.Tensor([best_network[0], best_network[1]]).float().to(device))
    scheduler = get_scheduler("cosine_with_restarts", optimizer, 10, 50)
    model, optimizer = learning_loop(
        model = model,
        optimizer = optimizer,
        train_loader = dataloader_train,
        val_loader = dataloader_val,
        criterion = criterion,
        scheduler = scheduler,
        epochs = 50,
        min_lr = 1e-6,
        val_every = 1
    )