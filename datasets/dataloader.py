import pdb
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def load_data(
        data_path: dict,
        num_single_path,
        test_num,
        val_index,
        logger=None,
        max_len: int = 128,
        batch_size: int = 128,
        num_workers: int = 12,
        dimension: int = 2,
        output_data=False
      ):
    """
    Load data from specified path.

    :param data_path: path to rawdata
    :param save_path_mean_std:
    :param num_classes:
    :param split_threshold:
    :param num_single_path:
    :param test_num:
    :param val_index:
    :param rl_interval_needed:
    :param max_len: the maximum length of sequences in rawdata (default: 100)
    :param num_samples_cls:
    :param batch_size: (default: 128)
    :param shuffle: (default: True)
    :param num_workers: (default: 12)
    :param dimension: (default: 2)
    """

    data_dict = {}
    for name, path in data_path.items():
        if 'varying' not in name:
            if logger is not None:
                logger.info(f'Process dataset: {name}')
            sub_data_dict = read_process_fixed_length(
                data_path=path,
                data_name=name,
                num_single_path=num_single_path,
                test_num=test_num,
                val_index=val_index)
            data_dict.update(sub_data_dict)
        else:  # varying length
            if logger is not None:
                logger.info(f'Process dataset: varying length')
            sub_data_dict = read_process_varying_length_rawdata(data_path=path)
            data_dict.update(sub_data_dict)

    if output_data:
        return data_dict

    labels = []
    datasets_df = {}
    trainset_df = pd.DataFrame(data=None, columns=['seqs', 'labels', 'ori_len'])
    for name, data in data_dict.items():
        sub_df = pd.DataFrame(data=zip(*data), columns=['seqs', 'labels', 'ori_len'])
        if 'train' in name:
            labels.append(data[1])
            trainset_df = pd.concat([trainset_df, sub_df], ignore_index=True)
        else:
            datasets_df[name] = sub_df
    datasets_df['train'] = trainset_df

    mean, std = get_mean_std(np.concatenate(labels, axis=0))
    with open('./checkpoints/mean_std/mean_std_dict.pkl', 'wb') as temp:
        pickle.dump({'mean': mean, 'std': std}, temp)

    dataloader_dict = {}
    for name, df in datasets_df.items():
        sub_dataset = UTRDataset(
            dataframe=df, mean=mean, std=std, max_len=max_len, dimension=dimension)
        sub_dataloader = DataLoader(dataset=sub_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)
        dataloader_dict[name] = sub_dataloader

    return dataloader_dict


def read_process_fixed_length(
        data_path,
        data_name,
        num_single_path: int = 220000,
        test_num: int = 20000,
        val_index: int = 200000,
):
    seqs, labels = read_fixed_length_rawdata(data_path, num_single_path)
    lengths = np.array([len(seq) for seq in seqs])

    train_seqs, train_labels, train_lengths = (
        seqs[test_num:val_index], labels[test_num:val_index], lengths[test_num:val_index])
    val_seqs, val_labels, val_lengths = (
        seqs[val_index:], labels[val_index:], lengths[val_index:])
    test_seqs, test_labels, test_lengths = (
        seqs[:test_num], labels[:test_num], lengths[:test_num])

    if 'egfp' in data_name:
        return {
            f'{data_name}_train': (train_seqs, train_labels, train_lengths),
            f'{data_name}_val': (val_seqs, val_labels, val_lengths),
            f'{data_name}_test': (test_seqs, test_labels, test_lengths)}
    else:  # mcherry
        return {f'{data_name}_test': (test_seqs, test_labels, test_lengths)}


def read_fixed_length_rawdata(data_path, num):
    rawdata = pd.read_excel(data_path)
    rawdata = rawdata.sort_values(by=['total'], ascending=False).reset_index(drop=True)
    rawdata = rawdata.loc[:num - 1, ['utr', 'total', 'rl']]

    # get seqs and labels
    seqs = rawdata['utr']
    labels = rawdata['rl']

    return np.array(seqs), np.array(labels)


def read_process_varying_length_rawdata(data_path):
    df = pd.read_csv(data_path)

    random = df[df['set'] == 'random']
    # Filter out UTRs with too few less reads
    random = random[random['total_reads'] >= 10]
    random.sort_values('total_reads', inplace=True, ascending=False)
    random.reset_index(inplace=True, drop=True)  # shape: (83919, 34)

    human = df[df['set'] == 'human']
    # Filter out UTRs with too few less reads
    human = human[human['total_reads'] >= 10]
    human.sort_values('total_reads', inplace=True, ascending=False)
    human.reset_index(inplace=True, drop=True)  # shape: (15555, 34)

    # Reference:
    # To ensure that 5′ UTRs of every length would be represented equally, we took the
    # 100 5′ UTRs with the deepest read coverage at every length (~10% of the library) as the test
    # set, rather than using the top 10% of the entire population. The remaining 90% was used for
    # training.

    # random test dataset
    subrandom = pd.DataFrame(columns=random.columns)
    for i in range(25, 101):
        tmp = random[random['len'] == i].copy()
        tmp.sort_values('total_reads', inplace=True, ascending=False)
        tmp.reset_index(inplace=True, drop=True)
        subrandom = pd.concat([subrandom, tmp.iloc[:100]], ignore_index=True)  # shape: (7600, 34)

    # human test dataset
    subhuman = pd.DataFrame(columns=human.columns)
    for i in range(25, 101):
        tmp = human[human['len'] == i].copy()
        tmp.sort_values('total_reads', inplace=True, ascending=False)
        tmp.reset_index(inplace=True, drop=True)
        subhuman = pd.concat([subhuman, tmp.iloc[:100]], ignore_index=True)  # shape: (7600, 34)

    # train dataset
    trainset = pd.concat([random, human, subrandom, subhuman], ignore_index=True).drop_duplicates(keep=False)  # shape: (84274, 34)

    seqs_random_testset, labels_random_testset, len_random_testset = subrandom['utr'], subrandom['rl'], subrandom['len']
    seqs_human_testset, labels_human_testset, len_human_testset = subhuman['utr'], subhuman['rl'], subhuman['len']
    seqs_trainset, labels_trainset, len_trainset = trainset['utr'], trainset['rl'], trainset['len']

    return {
        'varying_length_random_test':
            (np.array(seqs_random_testset), np.array(labels_random_testset), np.array(len_random_testset)),
        'varying_length_human_test':
            (np.array(seqs_human_testset), np.array(labels_human_testset), np.array(len_human_testset)),
        'varying_length_train':
            (np.array(seqs_trainset), np.array(labels_trainset), np.array(len_trainset))}


def get_mean_std(data):
    mean = np.mean(data)
    std = np.std(data)
    return mean, std


def denormalize(data, mean, std):
    data_denorm = data * std + mean
    return data_denorm


class UTRDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 mean,
                 std,
                 max_len: int = 100,
                 dimension: int = 2):
        self.seqs = np.array(dataframe['seqs'])
        self.labels = np.array(dataframe['labels'])
        self.lengths = np.array(dataframe['ori_len'])
        self.max_len = max_len
        self.mean = mean
        self.std = std
        self.dimension = dimension

    def __getitem__(self, index):
        out_dict ={}

        out_dict["ori_len"] = self.lengths[index]
        seq_mask = get_mask(self.seqs[index], self.max_len, self.dimension)
        seq_padded = seq_padding(self.seqs[index], self.max_len)
        seq_onehot = one_hot_encode(seq_padded, self.max_len, self.dimension)
        label_normalized = (self.labels[index] - self.mean) / self.std
        out_dict["label"] = torch.Tensor([label_normalized])
        return seq_onehot, seq_mask, out_dict

    def __len__(self):
        return len(self.seqs)


def get_mask(seq, max_len, dimension):
    assert isinstance(seq, str), 'the number of sequence must be one'

    ori_len = len(seq)
    if dimension == 2:
        mask_1 = torch.ones([4, ori_len])  # valid
        mask_0 = torch.zeros([4, max_len - ori_len])  # invalid
        mask = torch.cat((mask_1, mask_0), dim=1)
        assert mask.shape == (4, max_len), 'the mask shape is wrong, it should be (4, {})'.format(max_len)
    elif dimension == 3:
        mask_1 = torch.ones([1, ori_len, 4])  # valid
        mask_0 = torch.zeros([1, max_len - ori_len, 4])  # invalid
        mask = torch.cat((mask_1, mask_0), dim=1)
        assert mask.shape == (1, max_len, 4), 'the mask shape is wrong, it should be (1, {}, 4)'.format(max_len)
    else:
        raise ValueError("dimension must be 2 or 3, please check it again")
    return torch.Tensor(mask)


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


def binary_mtx(mtx, max_len, ori_len):
    b_mtx = torch.zeros([max_len, 4])
    for k in range(max_len):
        if k < ori_len:
            b_mtx[k][torch.argmax(mtx[k])] = 1
    return b_mtx


def decode_seq(mtx, max_len):  # single matrix as input
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}
    seq = []
    for i in range(max_len):
        for x in ['a', 'c', 'g', 't', 'n']:
            if (mtx[i] == torch.Tensor(nuc_d[x])).all():
                seq.append(x)
                break
    return "".join(seq)


def mtx2seq(mtxs, max_len, ori_len):
    seqs = []
    for i in range(len(mtxs)):
        mtx_binaried = binary_mtx(torch.reshape(mtxs[i], (max_len, 4)),
                                  max_len,
                                  ori_len[i])
        seq = decode_seq(mtx_binaried, max_len)
        seqs.append(seq[:ori_len[i]])

    return seqs


def getseq_input_single(seq, max_len, dimension):
    seq_padded = seq_padding(seq, max_len)
    seq_onehot = one_hot_encode(seq_padded, max_len, dimension)
    if dimension == 2:
        return seq_onehot.reshape(1, 4, max_len)
    else:
        return seq_onehot.reshape(1, 1, max_len, 4)