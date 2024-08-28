import os
import argparse 
import random

argparser = argparse.ArgumentParser()
argparser.add_argument('--train_split', type=float, default=0.8, help='Train split. Default: 0.8')
args = argparser.parse_args()

ORIGINAL_DATASET_PATH = 'train.txt'
TRAIN_DATASET_PATH = f'train_{args.train_split}.txt'
VAL_DATASET_PATH = f'val_{1 - args.train_split}.txt'

with open(ORIGINAL_DATASET_PATH, 'r', encoding='ISO-8859-1') as f:
    data = f.readlines()

random.shuffle(data)

train_split = int(len(data) * args.train_split)
train_data = data[:train_split]
val_data = data[train_split:]

with open(TRAIN_DATASET_PATH, 'w', encoding='ISO-8859-1') as f:
    f.writelines(train_data)

with open(VAL_DATASET_PATH, 'w', encoding='ISO-8859-1') as f:
    f.writelines(val_data)

print(f'Split the dataset into {TRAIN_DATASET_PATH} and {VAL_DATASET_PATH}.')