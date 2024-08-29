import argparse
import os
import sys

import numpy as np
import requests

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer_path", type=str, required=True)

args = parser.parse_args()

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if os.path.exists(input_file_path):
    os.remove(input_file_path)

data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
with open(input_file_path, 'w', encoding='utf-8') as f:
    f.write(requests.get(data_url, timeout=10).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

enc = Tokenizer(args.tokenizer_path)
train_ids = enc.encode(train_data, bos=True, eos=True)
val_ids = enc.encode(val_data, bos=True, eos=True)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint32)
val_ids = np.array(val_ids, dtype=np.uint32)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train has 270,307 tokens
# val has 31,466 tokens