
import os
import sys
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from src.models.gpt import GPT
from src.train.train import Trainer
from src.utils.utils import set_seed, setup_logging, CfgNode as CN

from src.tokenizers.CharTokenizer import CharTokenizer

from src.datasets.get_dataset import get_dataset

import pickle

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out'

    # model
    C.model = GPT.get_default_config()

    # trainer
    C.trainer = Trainer.get_default_config()

    return C


# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    set_seed(config.system.seed)

    # get training dataset (hardcoded for now)
    tokenizer = CharTokenizer(tokenizer_path='data/tokenizers/gdb13FullCharTokenizer.json')
    train_dataset = get_dataset(data_path='data/gdb13/gdb13_rand1m.smi',
                          tokenizer=tokenizer,
                          max_len=50)


    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    print(config)
    model = GPT(config.model)
    
    if config.model.pretrained_folder!=None:
        assert os.path.normpath(os.path.abspath(config.model.pretrained_folder)) != os.path.normpath(os.path.abspath(config.system.work_dir)), "pretrained folder cannot be same as current folder. Change the folder name of your pretrained model or current directory using flags"
        model.load_pretrained(config.model.pretrained_folder)
    
    setup_logging(config)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    train_losses = []
   
    # iteration callback
    """def batch_end_callback(trainer):
        if trainer.iter_num % 1 == 0:
            train_losses.append(trainer.loss.item())
            if trainer.device=="cuda":
                print(f"iter_dt {trainer.iter_dt:.2f}s; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f};attn_times {trainer.attn_times*1000:.2f}ms;mem_consumed {trainer.memory_consumed/(1024*1024):.2f}MB")
            else:
                print(f"iter_dt {trainer.iter_dt:.2f}s; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f};attn_times {trainer.attn_times*1000:.2f}ms;mem_consumed - not available on CPU")

        if (trainer.iter_num + 1) % 200 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                #context = "O God, O God!"
                context = "(CCO)"
                encoded_context = train_dataset.tokenizer(context, max_length=40)
                x = torch.tensor(encoded_context, dtype=torch.long)[None,...].to(trainer.device)
                y, attn_time = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)
                y = y[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
                print(f"Attention computation took {attn_time*1000:.2f}ms to run for {config.data.block_size} seq length")
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print("saving loss and attention logs")
            with open(os.path.join(config.system.work_dir, 'train_losses.json'), 'w') as f:
                json.dump(train_losses, f, ensure_ascii=False, indent=4)
            # revert model to training mode
            model.train()"""

    #trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
