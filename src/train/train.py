import os
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from ..utils.general_utils import CfgNode as CN


class Trainer():

    @staticmethod
    def get_default_config(model_type):
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

        if model_type == 'gpt':
            C.tokenizer_path = 'data/tokenizers/gdb13CharTokenizer.json'
            C.dataset_path = 'data/gdb13/gdb13_rand1m.smi'
            C.dataname = 'gdb13'
            C.sample = False

        return C
    
    def __init__(self, config, model, dataset):
        self.config = config
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.model = model
        self.optimizer = model.configure_optimizers(config)
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        self.n_examples = 0
        self.n_iter = 0
        self.n_epoch = 0
        self.best_loss = np.inf
        self.loss_improved = False

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

        """dataloader = DataLoader(self.dataset,
                                sampler=torch.utils.data.RandomSampler(self.dataset, replacement=True, num_samples=int(1e10)), 
                                        # TODO : is this sampling strategy appropriate?
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                batch_size=config.batch_size)"""
        
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 shuffle=True,
                                                 batch_size=config.batch_size,
                                                 num_workers=config.num_workers,
                                                 pin_memory=True)
        
        
        model.train()

        for epoch in range(config.epochs):

            for batch, encodings in enumerate(dataloader):
                self.optimizer.zero_grad()

                for k, v in encodings.items():
                    encodings[k] = v.to(self.device)
                
                assert "labels" in encodings.keys()
                
                self.loss, logits, *_ = self.model(**encodings) 
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()
                
                self.n_examples += dataloader.batch_size

                """if self.loss.item() < self.best_loss:
                    self.loss_improved = True
                    self.best_loss = self.loss.item()
                
                else: 
                    self.loss_improved = False"""

                self.trigger_callbacks('on_batch_end')
                
                
                if batch % 200 == 0:
                    print(f'epoch: {epoch + 1}, batch: {batch + 1}, loss: {self.loss.item()}')
                
                self.n_iter += 1

            self.n_epoch += 1
            
              


