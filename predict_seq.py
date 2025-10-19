""" Predict MRLs of 5' UTRs"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

os.chdir('path/to/root')
sys.path.append(os.getcwd())

from utils import setup_seed, add_dict_to_argparser, chunk, dev


def main():
    setup_seed(seed=666)
    args = create_argparser().parse_args()

    print('>>>Loading model...')
    if args.dimension == 2:
        from models.TwoStage1D import Predictor
    else:
        from models.TwoStage2D import Predictor
    predictor = load_predictor_for_test(model=Predictor, model_path=args.model_path)

    print('>>>Loading data...')
    df = pd.read_excel(args.data_path)
    seqs = np.array(df['Sequence'])
    seqs_onehot = []
    for seq in seqs:
        seq_padded = seq_padding(seq, args.max_len)
        seq_onehot = one_hot_encode(seq_padded, args.max_len, dimension=args.dimension)
        seqs_onehot.append(np.array(seq_onehot))

    print('>>>Predicting...')
    rl_pred = []
    predictor.eval()
    with torch.no_grad():
        for mini_batch in chunk(seqs_onehot, limit=args.batch_size):
            mini_batch = np.array(mini_batch)[:, :, :args.max_len]
            mini_batch = torch.Tensor(mini_batch).to(dev())
            mini_rl_pred = predictor(mini_batch)
            rl_pred.append(mini_rl_pred.cpu().data.numpy())
    rl_pred = np.concatenate(rl_pred, axis=0).reshape(1, -1)[0]

    print('>>>Writing...')
    rl_pred_de = denormalize(rl_pred, args.save_path_mean_std)
    subdf = pd.DataFrame({'new_rl_adjusted': rl_pred_de})
    df = pd.concat([df, subdf], axis=1)
    df.to_excel(args.data_path)
    print('Finished!')


def create_argparser():
    defaults = dict(
        model_path='./root/to/ckpt_path',
        data_path='./root/to/data_path',
        save_path_mean_std='./root/to/temp_data_path',
        max_len=100,
        batch_size=3,
        dimension=3,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_predictor_for_test(model, model_path=None):
    predictor = model(dropout_rate=0.2).to(dev())

    if model_path is not None:
        print(f'loading from checkpoint: {model_path}...')
        path_checkpoint = model_path
        checkpoint = torch.load(path_checkpoint)
        predictor.load_state_dict(checkpoint['model'], strict=True)
    else:
        print(f'creating an initial model...')

    return predictor


def denormalize(norm_rl, save_path_mean_std):
    with open(save_path_mean_std, 'rb') as temp:
        temp_dict = pickle.load(temp)
    scaler = {'mean': temp_dict['mean'], 'std': temp_dict['std']}
    rl = norm_rl * scaler['std'] + scaler['mean']

    out_rl = []
    for rl_0 in rl:
        out_rl.append(rl_0.item())

    return out_rl


def seq_padding(seq, max_len):
    assert isinstance(seq, str), 'the number of sequence must be one'

    ori_len = len(seq)  # ori_len <= max_len
    if ori_len < max_len:
        padding = max_len - ori_len
        seq_padded = seq + padding * 'N'
        return seq_padded
    else:
        return seq


def one_hot_encode(seq, max_len, dimension):
    assert isinstance(seq, str), 'the number of sequence must be one'

    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}

    seq = seq.lower()
    if 'u' in seq:
        seq = seq.replace('u', 't')

    if dimension == 2:
        onehotmtx = torch.zeros([4, max_len])  # the first dimension is channel dimension
        for n, x in enumerate(seq):
            for channel in range(4):
                onehotmtx[channel][n] = nuc_d[x][channel]
    elif dimension == 3:
        onehotmtx = torch.zeros([1, max_len, 4])
        for n, x in enumerate(seq):
            onehotmtx[0][n] = torch.Tensor(nuc_d[x])
    else:
        raise ValueError("dimension must be 2 or 3, please check it again")
    return torch.Tensor(onehotmtx)


if __name__ == '__main__':
    main()