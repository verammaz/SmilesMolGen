import wandb
import os
import sys
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from src.utils.general_utils import set_seed, setup_logging, CfgNode as CN

from src.tokenizers.CharTokenizer import CharTokenizer
from src.datasets.get_dataset import get_dataset
from src.utils.data_utils import get_max_smiles_len

from src.models.gpt import GPT
from src.train.train import Trainer
from src.reinforcement.reinforce import Reinforcer

from src.evaluate.evaluate import generate_smiles, get_statistics

import pickle

wandb.login()

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out'

    # pipeline
    C.pipeline = CN()
    C.pipeline.train_gpt = True
    C.pipeline.evaluate = True
    C.pipeline.reinforce = True

    # model
    C.model = GPT.get_default_config()

    # trainers
    C.gpt_trainer = Trainer.get_default_config()
    C.reinforcer = Reinforcer.get_default_config()

    return C


# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    set_seed(config.system.seed)

    config.model.name = f'{config.model.model_type}_{config.model.n_layer}_{config.model.n_query_head}_{config.model.n_kv_head}'

    if config.model.rope : config.model.name += '_rope'

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
    
    if config.model.pretrained != None:
        pretrained_state_dict = torch.load(config.model.pretrained, weights_only=True)
        model.load_state_dict(pretrained_state_dict, strict=False) 
    
    #setup_logging(config)
    out_dir = os.path.join(config.system.work_dir, config.gpt_trainer.dataname)
    wandb.init(project="MolGen", config=config)

    if config.pipeline.train_gpt:
        # construct the trainer object
        trainer = Trainer(config.gpt_trainer, model, train_dataset)

        # iteration callback
        def batch_end_trainer_callback(trainer):
            
            wandb.log({"n_examples" : trainer.n_examples, "train_loss": trainer.loss})
            
            ckpt_path = os.path.join(out_dir, f'{config.model.name}_preRL.pt')
            os.makedirs(out_dir, exist_ok=True)

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

                torch.save(model.state_dict(), ckpt_path)
            
                # revert model to training mode
                model.train()

        trainer.set_callback('on_batch_end', batch_end_trainer_callback)

        # run the optimization
        trainer.run()

    # evaluate
    if config.pipeline.evaluate:
        generated_smiles, _ = generate_smiles(model, tokenizer)
        stats_filename = config.model.name + '_stats_preRL.json'
        stats = get_statistics(generated_smiles, train_dataset._molecules, save_path=os.path.join(out_dir, stats_filename))

    if config.pipeline.reinforce:
        reinforcer = Reinforcer(config.reinforcer, model, train_dataset, tokenizer)
        # iteration callback
        def batch_end_rl_callback(reinforcer):
            
            wandb.log({"n_iter" : reinforcer.n_iter,   
                       "reward": reinforcer.reward,
                       "rl_loss": reinforcer.loss})
            
            ckpt_path = os.path.join(out_dir, f'{config.model.name}_RL_{reinforcer.config.target_property}.pt')
            os.makedirs(out_dir, exist_ok=True)

            if (reinforcer.n_iter + 1) % 200 == 0:
                model.eval()
                torch.save(model.state_dict(), ckpt_path)
                model.train()

        reinforcer.set_callback('on_batch_end', batch_end_rl_callback)

        # run the optimization
        reinforcer.run()

        # evaluate after rL
        if config.pipeline.evaluate:
            generated_smiles, _ = generate_smiles(model, tokenizer)
            stats_filename = config.model.name + f'_stats_RL_{reinforcer.config.target_property}.json'
            stats = get_statistics(generated_smiles, train_dataset._molecules, save_path=os.path.join(out_dir, stats_filename))
