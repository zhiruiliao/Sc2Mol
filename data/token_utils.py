import argparse
import os
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


class Tokenizer(object):
    def __init__(self, max_len, init_raw_txt=None, init_vocab_txt=None):
        self._max_len = max_len
        if init_raw_txt is not None and init_vocab_txt is None:
            self._vocab = self._init_from_raw_txt(init_raw_txt)
        elif init_raw_txt is None and init_vocab_txt is not None:
            self._vocab = self._init_from_vocab_txt(init_vocab_txt)
        else:
            raise ValueError("Only one  type of initial text file is supported.")
        self.idx_to_token = {idx: token for token, idx in self._vocab.items()}
    
    @property
    def max_len(self):
        return self._max_len
    
    @property
    def vocab(self):
        return self._vocab
    
    @property
    def vocab_size(self):
        return len(self._vocab)
    
    def _init_from_raw_txt(self, raw_txt):
        vocab = {}
        with open(raw_txt) as fin:
            text = ''
            for row in fin.readlines():
                text += row.strip()
            tokens = ['[PAD]', '[SOS]', '[EOS]', '[UNK]'] + sorted(set(text))
        for i, token in enumerate(tokens):
            vocab[token] = i
        return vocab
        
    def _init_from_vocab_txt(self, vocab_txt):
        vocab = {}
        with open(vocab_txt) as fin:
            for row in fin.readlines():
                token, idx = row.split()
                vocab[token] = int(idx)
        return vocab
    
    def save_vocab_txt(self, vocab_txt):
        with open(vocab_txt, 'w') as fout:
            for token, idx in self._vocab.items():
                print(token, idx, file=fout)
        print("The vocabulary text has been saved in %s" % vocab_txt)

    def chars_to_ids(self, chars):
        ids = np.zeros(self._max_len)
        n = min(len(chars), self._max_len - 2)
        ids[0] = self._vocab['[SOS]']
        ids[n + 1] = self._vocab['[EOS]']
        for i in range(n):
            try:
                ids[i + 1] = self._vocab[chars[i]]
            except KeyError:
                ids[i + 1] = self._vocab['[UNK]']
        return ids
    
    def ids_to_chars(self, ids, join=True, clean=True):
        chars = []
        for i in range(len(ids)):
            char = self.idx_to_token[ids[i]]
            if clean:
                if char not in ['[PAD]', '[SOS]', '[EOS]', '[UNK]']:
                    chars += [char]
            else:
                chars += [char]
        if join:
            return "".join(chars)
        else:
            return chars
    
    def tokenize(self, input_txt, save_path, split=1):
        with open(os.path.join(save_path, input_txt)) as fin:
            text = fin.readlines()
        n = len(text)
        print(f"{input_txt} has total {n} SMILES")
        pack_size = n // split + 1
        array = np.zeros((pack_size, self._max_len))
        current_num = 0
        pack_no = 0
        for i, row in enumerate(text):
            clean_row = row.strip()
            ids = self.chars_to_ids(clean_row)
            array[current_num, :] = ids
            current_num += 1
            if current_num == pack_size:
                np.save(os.path.join(save_path, input_txt[:-4] + '_%03d' % pack_no), array)
                print(array.shape)
                current_num = 0
                array = np.zeros((pack_size, self._max_len))
                pack_no += 1
            if i == n - 1:
                array = array[: n % pack_size]
                np.save(os.path.join(save_path, input_txt[:-4] + '_%03d' % pack_no), array)
                print(array.shape)
        assert pack_no == split - 1
        print("[%d] packs, total [%d] samples." % (split, n))
    
    def __str__(self):
        return "Tokenizer [max_len: %d] [vocab_size: %d]" % (self._max_len, len(self._vocab))


def get_mol_core_aroma(mol):
    core_mol = MurckoScaffold.GetScaffoldForMol(mol)
    core = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core_mol),
                            isomericSmiles=False,
                            kekuleSmiles=False,
                            doRandom=True)
    return core


def get_mol_chain_aroma(mol):
    chain = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(mol),
                             isomericSmiles=False,
                             kekuleSmiles=False,
                             doRandom=True)
    return chain


def multi_to_single(smiles):
    new_smiles = smiles.replace('[H]', 'D')
    new_smiles = new_smiles.replace('[nH]', 'M')
    new_smiles = new_smiles.replace('Br', 'R')
    new_smiles = new_smiles.replace('Cl', 'L')
    return new_smiles


def single_to_multi(smiles):
    new_smiles = smiles.replace('D', '[H]')
    new_smiles = new_smiles.replace('M', '[nH]')
    new_smiles = new_smiles.replace('R', 'Br')
    new_smiles = new_smiles.replace('L', 'Cl')
    return new_smiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--max_len', type=int, required=True)
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--save_path', type=str, required=True)

    args = parser.parse_args()
    print(args)
    
    fin = open(args.input, 'r')

    max_patience = 100
    
    core_set = set()
    chain_set = set()
    
    smiles_num = 0
    fout_target = open(os.path.join(args.save_path, args.input[:-4] + '_target.txt'), 'w')
    fout_chain = open(os.path.join(args.save_path, args.input[:-4] + '_chain.txt'), 'w')
    for row in fin.readlines():
        smiles = row.strip()
        mol = Chem.MolFromSmiles(smiles)
        
        for pat in range(max_patience):
            chain_smiles = get_mol_chain_aroma(mol)
            if chain_smiles not in chain_set:
                chain_set.add(chain_smiles)
                break
        
        new_smiles = multi_to_single(smiles)
        new_chain_smiles = multi_to_single(chain_smiles)
        
        print(new_smiles, file=fout_target)
        print(new_chain_smiles, file=fout_chain)
        
        smiles_num += 1
        if smiles_num % 1000 == 0:
            print("%d smiles have been translated" % smiles_num, end='\r')
            
    fin.close()
    fout_target.close()
    fout_chain.close()
    print(f"core set size: {len(core_set)}, chain_set size: {len(chain_set)}")
    
    
    tokenizer = Tokenizer(max_len=args.max_len, init_vocab_txt='vocab.txt'))
    print(tokenizer)
    
    tokenizer.tokenize(args.input[:-4] + '_target.txt', args.save_path, args.split)
    tokenizer.tokenize(args.input[:-4] + '_chain.txt', args.save_path, args.split)
