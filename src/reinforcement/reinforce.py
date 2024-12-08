import os
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from ..utils.general_utils import CfgNode as CN

from .rewards import *
from ..evaluate.evaluate import generate_smiles


class Reinforcer():

    @staticmethod
    def get_default_config(properties=None):
        C = CN()
        # device to train on
        C.device = 'auto'
        # training parameters
        C.epochs = 1
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0

        C.reward_fn = get_reward_fn(properties)
        C.target_property = properties

        return C
    
    def __init__(self, config, model, dataset, tokenizer):
        self.config = config
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = model.configure_optimizers(config)
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("Reinforce", self.device)

        self.n_examples = 0
        self.n_iter = 0
        self.n_epoch = 0
     

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)
    

    def run(self):
        model = self.model
        config = self.config
        
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 shuffle=True,
                                                 batch_size=config.batch_size,
                                                 num_workers=config.num_workers,
                                                 pin_memory=True)
        
        
        model.train()

        for epoch in range(config.epochs):
            for batch, encodings in enumerate(dataloader):

                for k, v in encodings.items():
                    encodings[k] = v.to(self.device)
                
                    assert "labels" in encodings.keys()
                    
                    self.loss, logits, *_ = self.model(**encodings) 
                    generated_smiles = generate_smiles(model, self.tokenizer, batch_size=dataloader.batch_size)
                    rewards = config.reward_fn(generated_smiles, device=config.device)
                    self.adj_loss = self.loss - rewards.mean()
                    model.zero_grad(set_to_none=True)
                    self.adj_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    self.n_examples += dataloader.batch_size
                    self.reward = rewards.mean()

                    # Print progress
                    if batch % 200 == 0:
                        valid_smiles = sum(rewards).item()
                        print(f'epoch: {epoch + 1}, batch: {batch + 1}, loss: {self.loss.item()}, valid SMILES: {valid_smiles}/{len(rewards)}')
                    
                    self.n_iter += 1

                self.n_epoch += 1