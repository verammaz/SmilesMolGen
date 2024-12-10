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
    
    def compute_discounted_returns(self, rewards, discount_factor):
        """Compute discounted rewards for the batch."""
        discounted_returns = []
        for reward_seq in rewards:
            seq_len = len(reward_seq)
            discounted = torch.zeros(seq_len, device=self.device)
            cumulative = 0.0
            for t in reversed(range(seq_len)):
                cumulative = reward_seq[t] + discount_factor * cumulative
                discounted[t] = cumulative
            discounted_returns.append(discounted)
        return discounted_returns


    def run(self):
      model = self.model
      config = self.config
      reward_fn = self.reward_fn

      

      model.train()
      batch_size = config.batch_size // 50

      for epoch in range(config.epochs):
        for batch in range(config.steps_per_epoch):
            loss = 0
            total_rewards = 0

            # Generate SMILES and compute rewards
            generated_smiles, generated_tokens = generate_smiles(
                model, self.tokenizer, batch_size=batch_size, size=100, 
                device=self.device, verb=False
            )
            rewards = reward_fn(generated_smiles, device=self.device)

            self.n_iter += 1

            # Inline computation of policy gradient loss
            for tokens, reward in zip(generated_tokens, rewards):
                # Compute discounted returns
                discounted_returns = (
                    torch.pow(config.discount_factor, torch.arange(len(tokens[:-1]), 0, -1, device=self.device)) 
                    * reward
                )

                # Prepare input and target tokens
                input_ids = torch.tensor([tokens[:-1]], dtype=torch.long).to(self.device)
                target_ids = torch.tensor([tokens[1:]], dtype=torch.long).to(self.device)

                # Compute model logits and log probabilities
                logits = model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]

                log_preds = torch.nn.functional.log_softmax(logits, dim=-1)
                idxs = target_ids.unsqueeze(-1)
                action_values = log_preds.gather(dim=2, index=idxs).squeeze(-1)

                # Compute policy gradient loss for this sequence
                sequence_loss = -torch.sum(action_values * discounted_returns)
                loss += sequence_loss

                # Track rewards
                total_rewards += reward.item()

              

            # Backpropagation
            self.loss = loss / batch_size
            self.reward = total_rewards / batch_size

            self.optimizer.zero_grad()
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()

            # Logging
            if batch % 200 == 0:
                print(f'epoch: {epoch + 1}, batch: {batch + 1}, loss: {self.loss.item():.4f}, avg reward: {self.reward:.2f}')

        self.n_epoch += 1
