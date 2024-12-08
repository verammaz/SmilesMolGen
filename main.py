import wandb
import os
import sys
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from src.utils.general_utils import set_seed, setup_logging, CfgNode as CN

from src.tokenizers.CharTokenizer import CharTokenizer
from src.datasets.get_dataset import get_dataset
from src.utils.data_utils import get_max_smiles_len

from src.models.gpt import GPT
from src.train.train import Trainer

import pickle

wandb.login()

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out'

    # model
    C.model = GPT.get_default_config()

    # trainers
    C.gpt_trainer = Trainer.get_default_config('gpt')
    C.predictor_trainer = Trainer.get_default_config('predictor')



    return C


# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    set_seed(config.system.seed)

    config.model.name = f'{config.model.model_type}_{config.model.n_layer}_{config.model.n_query_head}_{config.model.n_kv_head}'

    if config.model.rope : config.model.name += '_rope.pt'
    else : config.model.name += '.pt'

    # get training dataset (hardcoded for now)
    tokenizer = CharTokenizer(tokenizer_path=config.gpt_trainer.tokenizer_path)
    max_smiles_len = get_max_smiles_len(config.gpt_trainer.dataset_path) + 50
    
    train_dataset = get_dataset(data_path=config.gpt_trainer.dataset_path,
                          tokenizer=tokenizer,
                          max_len=max_smiles_len)


    # construct the model
    config.model.vocab_size = tokenizer.vocab_size
    config.model.block_size = train_dataset.get_block_size()
    #config.model.block_size = 512
    
    print(config)
    model = GPT(config.model)
    
    if config.model.pretrained_folder!=None:
        assert os.path.normpath(os.path.abspath(config.model.pretrained_folder)) != os.path.normpath(os.path.abspath(config.system.work_dir)), "pretrained folder cannot be same as current folder. Change the folder name of your pretrained model or current directory using flags"
        model.load_pretrained(config.model.pretrained_folder)
    
    #setup_logging(config)

    # construct the trainer object
    trainer = Trainer(config.gpt_trainer, model, train_dataset)
    
    wandb.init(project="MolGen", config=config)

    # iteration callback
    def batch_end_callback(trainer):
        
        wandb.log({"n_examples" : trainer.n_examples, "train_loss": trainer.loss})
        
        out_dir = os.path.join(config.system.work_dir, config.gpt_trainer.dataname)
        ckpt_path = os.path.join(out_dir, config.model.name)
        os.makedirs(out_dir, exist_ok=True)

        best_loss = np.inf
        
        if (trainer.n_iter + 1) % 200 == 0:
            model.eval()
            with torch.no_grad():
                if config.gpt_trainer.sample: 
                    # sample from the model...
                    tokens = model.sample([tokenizer.bos_token_id], 1, device=trainer.device)
                    tokens = tokens.tolist()
                    mol = tokens[0]
                    try:   
                        end_idx = mol.index(tokenizer.eos_token_id)
                    except ValueError:
                        end_idx = len(mol)
                    mol = mol[:end_idx+1]
                    smiles = tokenizer.decode(mol[1:-1])
                    print(f'\tSampled SMILES:  {smiles}')

                    #wandb.log({"SMILES String": smiles})
            
            # save the best model
            if trainer.loss.item() < best_loss:
                best_loss = trainer.loss.item()
                print("Loss decreased. Saving model...\n")
                torch.save(model.state_dict(), ckpt_path)
        
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
