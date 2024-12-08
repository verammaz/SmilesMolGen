import json
import sys

# Source : https://github.com/eyalmazuz/MolGen/blob/master/MolGen/src/tokenizers/CharTokenizer.py

class CharTokenizer():

    def __init__(self, tokenizer_path):
        try:
            with open(tokenizer_path, 'r') as f:
                print('Loading existing tokenizer...')
                self.id2token = json.load(f)
                self.id2token = {int(k): v for k, v in self.id2token.items()}
        except:
            print(f'Error: tokenizer dict {tokenizer_path} does not exist. Please build the tokenizer first.')
            sys.exit(1)
        
        # Add special tokens if necessary
        len_tokens = len(self.id2token)
        tokens = self.id2token.values()
        if '[PAD]' not in tokens:
            self.id2token[len_tokens] = '[PAD]'
            len_tokens += 1
        if '[BOS]' not in tokens:
            self.id2token[len_tokens] = '[BOS]'
            len_tokens += 1
        if '[EOS]' not in tokens:
            self.id2token[len_tokens] = '[EOS]'
            len_tokens += 1
        if '[SEP]' not in tokens:
            self.id2token[len_tokens] = '[SEP]'
            len_tokens += 1
        if '[UNK]' not in tokens:
            self.id2token[len_tokens] = '[UNK]'
            len_tokens += 1
        if '[CLS]' not in tokens:
            self.id2token[len_tokens] = '[CLS]'
            len_tokens += 1
        
        # Save (updated) tokenizer
        with open(tokenizer_path, 'w') as f:
            json.dump(self.id2token, f)
        
        self.token2id = {v: k for k, v in self.id2token.items()}
        

    @property
    def vocab_size(self):
        return len(self.id2token)
    
    @property
    def bos_token(self):
        return '[BOS]'

    @property
    def bos_token_id(self):
        return self.token2id['[BOS]']

    @property
    def eos_token(self):
        return '[EOS]'

    @property
    def eos_token_id(self):
        return self.token2id['[EOS]']

    @property
    def pad_token(self):
        return '[PAD]'

    @property
    def pad_token_id(self):
        return self.token2id['[PAD]']

    @property
    def sep_token(self):
        return '[SEP]'
    
    @property
    def unk_token(self):
        return '[UNK]'

    @property
    def unk_token_id(self):
        return self.token2id['[UNK]']

    @property
    def cls_token(self):
        return '[CLS]'

    @property
    def cls_token_id(self):
        return self.token2id['[CLS]']


    def tokenize(self, smiles, padding=False, max_len=-1):
        encodings = []
        bos, eos, sep, cls, sca = [], [], [], [], []

        if smiles.startswith('[CLS]'):
            smiles = smiles[5:]
            cls.append('[CLS]')   
        
        if smiles.startswith('[BOS]'):
            smiles = smiles[5:]
            bos.append('[BOS]')   
                 
        if smiles.endswith('[EOS]'):
            eos.append('[EOS]') 
            smiles = smiles[:-5]

        if '[SEP]' in smiles:
            idx = smiles.find('[SEP]')
            sca += smiles[:idx]
            sep.append(smiles[idx:idx+5])
            smiles = smiles[idx+5:]


        encodings = self.convert_tokens_to_ids(bos + sca + sep + list(smiles) + eos)
        # encodings = self.convert_tokens_to_ids(bos + cls + list(smiles) + eos)
        
        padding_mask = [0] * len(encodings)
        
        if padding and max_len != -1 and len(encodings) < max_len:
            pad_len = (max_len - len(encodings))
            encodings += [self.token2id['[PAD]']] * pad_len
            padding_mask += [1] * pad_len 

        elif max_len != -1 and len(encodings) > max_len:
            encodings = encodings[:max_len]

        return {"input_ids": encodings,
                "padding_mask": padding_mask}



    def __call__(self, smiles, padding=False, max_length=-1):
        return self.tokenize(smiles, padding, max_length)

    def convert_tokens_to_ids(self, tokens):
        encodings = []
        for char in tokens:
            encodings.append(self.token2id[char])
        return encodings

    def convert_ids_to_tokens(self, encodings):
        tokens = []
        for id_ in encodings:
            tokens.append(self.id2token[id_])
        return tokens


    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


    def decode(self, tokens):
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(tokens))
