import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm as tq
import tqdm.notebook as tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from .learner import Learner
from .optimizers import get_params_grad

__all__ = ['Learner']

class DelayerScheduler(_LRScheduler):
    """
    CREDIT: https://github.com/pabloppp/pytorch-tools
    Start with a flat lr schedule until it reaches N epochs then applies a scheduler
    """
    def __init__(self, optimizer, delay_epochs, after_scheduler):
        self.delay_epochs = delay_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_epoch >= self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_last_lr()
        
        return self.base_lrs
    
    def step(self, epoch=None):
        if self.finished:
            if epoch is None: 
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.delay_epochs)
        else:
            return super(DelayerScheduler, self).step(epoch)
        
def FlatCosAnnealScheduler(optimizer, delay_epochs, cosine_annealing_epochs):
    base_scheduler = CosineAnnealingLR(optimizer, cosine_annealing_epochs)
    return DelayerScheduler(optimizer, delay_epochs, base_scheduler)

def delayer(epochs, pct_start=0.8): return int(epochs * pct_start)

class Evaluate:
    """
    Keeping track of stats when training a model. Evolution will be all training stats for the entire training while summary is epoch summary per epoch
    """
    def __init__(self):
        self.evolution = {'train_loss':[], 'train_acc':[], 'valid_loss':[], 'valid_acc':[]}
        self.summary = {'train_loss_summary':[], 'train_acc_summary':[], 'valid_loss_summary':[], 'valid_acc_summary':[]}
        self.best_stats = {}
        
    def update_summary(self, train_loss, train_acc, valid_loss, valid_acc):
        """Call after each epoch"""
        self.summary['train_loss_summary'].append(train_loss)
        self.summary['train_acc_summary'].append(train_acc)
        self.summary['valid_loss_summary'].append(valid_loss)
        self.summary['valid_acc_summary'].append(valid_acc)
    
    def update_evolution(self, ds_type, loss, acc):
        """Call after each batch"""
        if ds_type=='train':
            self.evolution['train_loss'].append(loss)
            self.evolution['train_acc'].append(acc)
        elif ds_type=='valid':
            self.evolution['valid_loss'].append(loss)
            self.evolution['valid_acc'].append(acc)
        else: return
        
    def update_best_stats(self, iteration, train_loss, train_acc, valid_loss, valid_acc):
        self.best_stats['iteration'] = iteration
        self.best_stats['train_loss'] = train_loss
        self.best_stats['train_acc'] = train_acc
        self.best_stats['valid_losss'] = valid_loss
        self.best_stats['valid_acc'] = valid_acc
        
    def report(self):
        tq.write(f"""Train loss: {self.summary['train_loss_summary'][-1]:0f}, Train acc: {self.summary['train_acc_summary'][-1]:0f}, Valid loss: {self.summary['valid_loss_summary'][-1]:0f}, Valid acc: {self.summary['valid_acc_summary'][-1]:0f}""")
        
    def summarize(self): self.summary = pd.DataFrame(self.summary)
    
    def report_best(self):
        print(f"""
        Best Model Stats:
        ---------------------------------
        Iteration  : {self.best_stats['iteration']}
        Train Loss : {self.best_stats['train_loss']}
        Train Acc  : {self.best_stats['train_acc']}
        Valid Loss : {self.best_stats['valid_loss']}
        Valid Acc  : {self.best_stats['valid_acc']}
        """)
        
    def plot_summary(self):
        if isinstance(self.summary, pd.DataFrame): self.summary.plot()
            
    def plot_evolution(self):
        fig, axs = plt.subplots(2,2, figsize=(15,10))
        
        axs[0,0].plot(self.evolution['train_acc'])
        axs[0,0].set_xlabel('Iterations')
        axs[0,0].set_ylabel('Accuracy')
        axs[0,0].set_title('Train Accuracy')
        
        axs[0,1].plot(self.evolution['valid_acc'])
        axs[0,1].set_xlabel('Iterations')
        axs[0,1].set_ylabel('Accuracy')
        axs[0,1].set_title('Valid Accuracy')
        
        axs[1,0].plot(self.evolution['train_loss'])
        axs[1,0].set_xlabel('Iterations')
        axs[1,0].set_ylabel('Loss')
        axs[1,0].set_title('Train Loss')
        
        axs[1,1].plot(self.evolution['valid_loss'])
        axs[1,1].set_xlabel('Iterations')
        axs[1,1].set_ylabel('Loss')
        axs[1,1].set_title('Valid Loss')
        
        plt.show()
        
def metrics_batch(output, target, metric_fn=None):
    if metric_fn: return metric_fn(output, target)
    
def loss_batch(loss_fn, output, target, metric_fn=None, opt_fn=None, scheduler=None):
    """
    Calculate loss and metric for batch
    """
    loss = loss_fn(output, target)
    metric = metrics_batch(output, target, metric_fn)
    
    if opt_fn:
        opt_fn.zero_grad()
        loss.backward()
        opt_fn.step()
        if scheduler: scheduler.step()
            
    return loss.data.cpu().item(), metric

def loss_epoch(model, dataloader, ds_type, evaluator, loss_fn, metric_fn=None, opt_fn=None, scheduler=None, device=None, inner_bar=None, inner_loop=None):
    """
    Calculates loss per batch
    """
    running_loss, running_metric = 0,0
    n = len(dataloader)
    nb_ = 0
    device = torch.device('cpu') if device is None else device
    
    for i,(xb,yb) in enumerate(dataloader):
        inner_bar.update(1)
        nb = yb.size(0)
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        
        loss_b, metric_b = loss_batch(loss_fn, output, yb, metric_fn, opt_fn, scheduler)
        running_loss+=loss_b
        running_metric+=metric_b
        nb_+=nb
        
        evaluator.update_evolution(ds_type, running_loss/float(i+1), running_metric/float(nb_))
        
    metric_e = running_metric/float(nb_)
    loss_e = running_loss/float(n)
    return metric_e, loss_e


def loss_batch_ada(model, loss_fn, output, target, opt_fn=None, scheduler=None, metric_fn=None):
    """
    Calculate the loss per batch
    will perform backprop if opt_fn is passed, otherwise evalation 
    """
    loss = loss_fn(output, target)
    metric = metrics_batch(output, target, metric_fn)
    
    # perform backprop if provided
    if opt_fn:
        opt_fn.zero_grad()
        loss.backward(create_graph=True, retain_graph=True)
        _, gradsH = get_params_grad(model)
        opt_fn.step(gradsH)
        
        # perform scheduling if provided
        if scheduler: scheduler.step()
            
    return loss.data.cpu().item(), metric

def loss_epoch_ada(model, dataloader, ds_type, evaluator, loss_fn, metric_fn=None, opt_fn=None, scheduler=None, device=None, inner_bar=None, inner_loop=None):
    """
    Calculate loss per epoch with given dataloader. 
    """
    running_loss, running_metric = 0., 0
    n = len(dataloader)
    nb_ = 0
    device = torch.device(device) if device is None else device
    
    for i,(xb,yb) in enumerate(dataloader):
        inner_bar.update(1)
        nb = yb.size(0)
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        
        loss_b, metric_b = loss_batch_ada(model, loss_fn, output, yb, opt_fn, scheduler, metric_fn)
        running_loss+=loss_b
        running_metric+=metric_b
        nb_+=nb
        
        # NOTE: Check to see if this is printing correctly
        # this is updating the average loss for that batch -> per epoch
        evaluator.update_evolution(ds_type, running_loss/float(i+1), running_metric/float(nb_))
        
    # returning the total average loss over entire dataset
    loss_e = running_loss/float(n)
    metric_e = running_metric/float(nb_)
    return metric_e, loss_e

def fit(learn, epochs=10, device=None, keep_best_state=True, ada=False, **kwargs):
    """
    train model with one cycle policy
    """
    evaluator = Evaluate()
    
    loss_e = loss_epoch_ada if ada==True else loss_epoch
    
    # update opt
    opt_fn = learn.opt_fn
    
    # base
    loss_fn = learn.loss_fn
    train_dl = learn.data.train_dl
    valid_dl = learn.data.valid_dl
    metric_fn = learn.metric_fn
    device = learn.device if device is None else torch.device(device)
    model = learn.model.to(device)
    
    # display settings
    outter_bar = tqdm.tqdm(range(epochs))
    outter_loop = range(epochs)
    train_n = len(train_dl)
    train_inner_bar = tqdm.tqdm(range(train_n), leave=False)
    train_inner_loop = range(train_n)
    valid_n = len(valid_dl)
    valid_inner_bar = tqdm.tqdm(range(valid_n), leave=False)
    valid_inner_loop = range(valid_n)
    
    # best model weights
    if keep_best_state:
        best_model_wts = deepcopy(model.state_dict())
        best_loss = float('inf')
        
    # Training
    for epoch in outter_loop:
        train_inner_bar.reset()
        valid_inner_bar.reset()
        
        # train
        model.train()
        train_metric, train_loss = loss_e(model, train_dl, 'train', evaluator, loss_fn, metric_fn, opt_fn, None, device, train_inner_bar, train_inner_loop)
        
        # eval
        model.eval()
        with torch.no_grad():
            valid_metric, valid_loss = loss_e(model, valid_dl, 'valid', evaluator, loss_fn, metric_fn, device=device, inner_bar=valid_inner_bar, inner_loop=valid_inner_loop)
            
        # update evaluator
        evaluator.update_summary(train_loss, train_metric, valid_loss, valid_metric)
        
        # keeping best
        if keep_best_state:
            if valid_loss<best_loss:
                best_loss = valid_loss
                best_model_wts = deepcopy(model.state_dict())
                evaluator.update_best_stats(epoch+1, train_loss, train_metric, valid_loss, valid_metric)
                
        # report
        outter_bar.update(1)
        evaluator.report()
        
    # summurize training
    evaluator.summarize()
    learn.evaluator = evaluator
    
    # keep best weights
    if keep_best_state: learn.model.load_state_dict(best_model_wts)
        
def fit_one_cycle(learn, epochs=10, pct_start=0.8, div_factor=10., moms=(0.85, 0.95), device=None, keep_best_state=True, ada=False, **kwargs):
    """
    train model with one cycle policy
    """
    # setting training params
    evaluator = Evaluate()
    
    loss_e = loss_epoch_ada if ada==True else loss_epoch
    
    # scheduler params
    steps_per_epoch = len(learn.data.train_dl)
    b1, b2 = moms
    opt_fn = learn.opt_fn
    scheduler = lr_scheduler.OneCycleLR(opt_fn, max_lr=learn.lr, steps_per_epoch=steps_per_epoch, epochs=epochs, pct_start=pct_start, div_factor=div_factor, base_momentum=b1, max_momentum=b2)
    
    # base
    loss_fn = learn.loss_fn
    train_dl = learn.data.train_dl
    valid_dl = learn.data.valid_dl
    metric_fn = learn.metric_fn
    device = learn.device if device is None else torch.device(device)
    model = learn.model.to(device)
    
    # display settings
    outter_bar = tqdm.tqdm(range(epochs))
    outter_loop = range(epochs)
    train_n = len(train_dl)
    train_inner_bar = tqdm.tqdm(range(train_n), leave=False)
    train_inner_loop = range(train_n)
    valid_n = len(valid_dl)
    valid_inner_bar = tqdm.tqdm(range(valid_n), leave=False)
    valid_inner_loop = range(valid_n)
    
    # best model weights
    if keep_best_state:
        best_model_wts = deepcopy(model.state_dict())
        best_loss = float('inf')
        
    # Training
    for epoch in outter_loop:
        train_inner_bar.reset()
        valid_inner_bar.reset()
        
        # train
        model.train()
        train_metric, train_loss = loss_e(model, train_dl, 'train', evaluator, loss_fn, metric_fn, opt_fn, scheduler, device, train_inner_bar, train_inner_loop)
        
        # eval
        model.eval()
        with torch.no_grad():
            valid_metric, valid_loss = loss_e(model, valid_dl, 'valid', evaluator, loss_fn, metric_fn, device=device, inner_bar=valid_inner_bar, inner_loop=valid_inner_loop)
            
        # update evaluator
        evaluator.update_summary(train_loss, train_metric, valid_loss, valid_metric)
        
        # keeping best
        if keep_best_state:
            if valid_loss<best_loss:
                best_loss = valid_loss
                best_model_wts = deepcopy(model.state_dict())
                evaluator.update_best_stats(epoch+1, train_loss, train_metric, valid_loss, valid_metric)
                
        # report
        outter_bar.update(1)
        evaluator.report()
        
    # summurize training
    evaluator.summarize()
    learn.evaluator = evaluator
    
    # keep best weights
    if keep_best_state: learn.model.load_state_dict(best_model_wts)
        
def fit_flat_anneal(learn, epochs=10, pct_start=0.8, div_factor=10., moms=(0.85, 0.95), device=None, keep_best_state=True, ada=False, **kwargs):
    """
    train model with one cycle policy
    """
    evaluator = Evaluate()
    
    loss_e = loss_epoch_ada if ada==True else loss_epoch
    
    # scheduler params
    steps_per_epoch = len(learn.data.train_dl)
    b1, b2 = moms
    delay_epochs = delayer(epochs, pct_start)
    opt_fn = learn.opt_fn
    base_scheduler = CosineAnnealingLR(opt_fn, delay_epochs)
    delayed_scheduler = DelayerScheduler(opt_fn, epochs-delay_epochs, base_scheduler)
    
    # base
    loss_fn = learn.loss_fn
    train_dl = learn.data.train_dl
    valid_dl = learn.data.valid_dl
    metric_fn = learn.metric_fn
    device = learn.device if device is None else torch.device(device)
    model = learn.model.to(device)
    
    # display settings
    outter_bar = tqdm.tqdm(range(epochs))
    outter_loop = range(epochs)
    train_n = len(train_dl)
    train_inner_bar = tqdm.tqdm(range(train_n), leave=False)
    train_inner_loop = range(train_n)
    valid_n = len(valid_dl)
    valid_inner_bar = tqdm.tqdm(range(valid_n), leave=False)
    valid_inner_loop = range(valid_n)
    
    # best model weights
    if keep_best_state:
        best_model_wts = deepcopy(model.state_dict())
        best_loss = float('inf')
        
    # Training
    for epoch in outter_loop:
        train_inner_bar.reset()
        valid_inner_bar.reset()
        
        # train
        model.train()
        train_metric, train_loss = loss_e(model, train_dl, 'train', evaluator, loss_fn, metric_fn, opt_fn, None, device, train_inner_bar, train_inner_loop)
        
        # eval
        model.eval()
        with torch.no_grad():
            valid_metric, valid_loss = loss_e(model, valid_dl, 'valid', evaluator, loss_fn, metric_fn, device=device, inner_bar=valid_inner_bar, inner_loop=valid_inner_loop)
            
        # update evaluator
        evaluator.update_summary(train_loss, train_metric, valid_loss, valid_metric)
        
        # keeping best
        if keep_best_state:
            if valid_loss<best_loss:
                best_loss = valid_loss
                best_model_wts = deepcopy(model.state_dict())
                evaluator.update_best_stats(epoch+1, train_loss, train_metric, valid_loss, valid_metric)
                
        # update scheduler
        delayed_scheduler.step()
        
        # report
        outter_bar.update(1)
        evaluator.report()
        
    # summurize training
    evaluator.summarize()
    learn.evaluator = evaluator
    
    # keep best weights
    if keep_best_state: learn.model.load_state_dict(best_model_wts)
        
Learner.fit = fit
Learner.fit_one_cycle = fit_one_cycle
Learner.fit_flat_anneal = fit_flat_anneal