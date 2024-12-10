import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from ..utils.general_utils import CfgNode as CN

from .rewards import *
from ..evaluate.evaluate import generate_smiles


class Reinforcer():

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # training parameters
        C.epochs = 1
        C.steps_per_epoch = 512
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.discount_factor = 0.99  # Discount factor for policy gradients

        C.target_property = 'Validity'

        return C
    
    def __init__(self, config, model, dataset, tokenizer):
        self.config = config
        self.reward_fn = get_reward_fn(config.target_property)
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
        print("Target Property", config.target_property)

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


    def compute_discounted_returns(rewards, discount_factor, device='cuda'):
        T = rewards.size(1)
        discounts = torch.pow(discount_factor, torch.arange(T, device=device))
        discounted_returns = torch.flip(
            torch.cumsum(torch.flip(rewards * discounts, dims=[1]), dim=1),
            dims=[1]
        )
        return discounted_returns


    def run(self):
      model = self.model
      config = self.config
      reward_fn = self.reward_fn

      

      model.train()
      batch_size = config.batch_size

      for epoch in range(config.epochs):
        
        for batch in range(config.steps_per_epoch):

            # Generate SMILES and compute rewards
            generated_smiles, generated_tokens = generate_smiles(
                model, self.tokenizer, batch_size=batch_size, size=100, 
                device=self.device, verb=False
            )

            rewards = reward_fn(generated_smiles, device=self.device) # [batch_size]
            rewards.unsqueeze(1) # [batch_size, 1]
            self.reward = rewards.mean().item()

            # Convert generated tokens to tensors
            input_ids = torch.tensor(
                [tokens[:-1] for tokens in generated_tokens], dtype=torch.long, device=self.device
            )
            target_ids = torch.tensor(
                [tokens[1:] for tokens in generated_tokens], dtype=torch.long, device=self.device
            )
            
            # Compute logits and log probabilities
            logits = model(input_ids)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Gather log probabilities of the generated tokens
            idxs = target_ids.unsqueeze(-1)
            action_values = log_probs.gather(dim=2, index=idxs).squeeze(-1)

            # Compute discounted returns for the entire batch
            discounted_returns = self.compute_discounted_returns(rewards, config.discount_factor, device=self.device)

            # Mask padding tokens if needed (optional, for padding tokens in fixed length)
            if self.tokenizer.pad_token_id is not None:
                mask = target_ids != self.tokenizer.pad_token_id
                action_values = action_values * mask

            # Compute loss
            self.loss = -torch.sum(action_values * discounted_returns) / config.batch_size

            # Backpropagation
            self.optimizer.zero_grad()
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            # Logging
            if batch % 200 == 0:
                print(f'epoch: {epoch + 1}, batch: {batch + 1}, loss: {self.loss.item():.4f}, avg reward: {avg_reward:.2f}')

        self.n_epoch += 1