import os
import pdb
import random
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from scipy.stats import linregress
from tqdm import tqdm


# set random seed
def setup_seed(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  #


# define loss function: weighted squared error
class WSE(nn.Module):
    def __init__(self):
        super(WSE, self).__init__()

    def forward(self, y_pred, y_true):
        weighted_mse = (1 + torch.exp(y_true)) * torch.square(y_pred - y_true)
        return torch.mean(weighted_mse)


# define metrics
def r2(x, y):
    r = linregress(x, y).rvalue
    return r ** 2


def accuracy(x, y):
    comp = torch.eq(x, y).sum().item()
    return comp / (len(x) * x.shape[-1])


def rm_module(path):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def train_predictor(model, train_loader, epoch, epochs, optimizer, loss_fn, b, logger, flood=False):
    model.train()  # turn on train mode
    train_rl_loss = 0
    train_loop = tqdm(train_loader, position=0)
    for onehot, _, out_dict in train_loop:
        onehot, label = onehot.cuda(), out_dict["label"].cuda()
        optimizer.zero_grad()
        rl = model(onehot)
        rl_loss = loss_fn['predictor'](rl, label)
        if flood == True:
            rl_loss = torch.abs(rl_loss - b) + b
        rl_loss.backward()
        optimizer.step()
        train_rl_loss = train_rl_loss + rl_loss.item() * onehot.size(0)

        # show training information
        train_loop.set_description(f'Epoch [{epoch}/{epochs}]')
        train_loop.set_postfix({'rl_loss': '{:.6f}'.format(rl_loss)})
    train_rl_loss /= len(train_loader.dataset)
    logger.info('>>Epoch: {} \tMode: Train \trl loss: {:.6f}'.format(epoch, train_rl_loss))

    return train_rl_loss


def train_autoencoder(model, train_loader, epoch, epochs, optimizer, loss_fn, b, logger, flood=False):
    autoencoder, predictor = model['autoencoder'], model['predictor']
    encoder_auto = autoencoder.getencoder()
    decoder_auto = autoencoder.getdecoder()
    encoder_pred = predictor.getencoder()
    predhead_pred = predictor.getpredhead()

    encoder_auto.train()
    decoder_auto.train()
    encoder_pred.eval()
    predhead_pred.eval()

    train_seqs_loss = 0
    train_loop = tqdm(train_loader, position=0)
    for onehot, mask, _ in train_loop:
        onehot, mask = onehot.cuda(), mask.cuda()
        optimizer.zero_grad()

        with torch.no_grad():
            vector_base = encoder_pred(onehot)
            rl = predhead_pred(vector_base)
        rl_repeat = rl.repeat(1, 25)
        rl_repeat = torch.reshape(rl_repeat, (-1, 1, 25, 1))
        vector_delta = encoder_auto(onehot)
        vector_combined = torch.cat([vector_base + vector_delta, rl_repeat], dim=1)
        seqs_pred = decoder_auto(vector_combined)
        seqs_pred = seqs_pred * mask

        seqs_loss = loss_fn['autoencoder'](seqs_pred, onehot)
        if flood == True:
            seqs_loss = torch.abs(seqs_loss - b) + b
        seqs_loss.backward()
        optimizer.step()
        train_seqs_loss = train_seqs_loss + seqs_loss.item() * onehot.size(0)

        # show training information
        train_loop.set_description(f'Epoch [{epoch}/{epochs}]')
        train_loop.set_postfix({'seqs_loss': '{:.6f}'.format(seqs_loss)})
    train_seqs_loss /= len(train_loader.dataset)
    logger.info('>>Epoch: {} \tMode: Train \tseqs loss: {:.6f}'.format(epoch, train_seqs_loss))

    return train_seqs_loss


def evaluate_predictor(model, eval_loader, epoch, loss_fn, mode, dataset, logger):
    model.eval()  # turn on evaluation mode
    eval_loss = 0
    all_rl = []
    all_labels = []
    with torch.no_grad():
        for onehot, _, out_dict in eval_loader:
            onehot, label = onehot.cuda(), out_dict["label"].cuda()
            rl = model(onehot)
            # sum loss
            rl_loss = loss_fn['predictor'](rl, label)
            eval_loss = eval_loss + rl_loss.item() * onehot.size(0)
            # save all rl and labels
            all_rl.append(rl.cpu().data.numpy())
            all_labels.append(label.cpu().data.numpy())
    # compute loss
    eval_loss /= len(eval_loader.dataset)
    # compute r-square
    all_rl, all_labels = np.concatenate(all_rl).reshape(1, -1), np.concatenate(all_labels).reshape(1, -1)
    r_square = r2(all_rl, all_labels)
    logger.info('>>Epoch: {} \tMode: {} \tDataset: {} \tLoss: {:.6f} \tR-square: {:6f}'
                .format(epoch, mode, dataset, eval_loss, r_square))

    return dict(
        loss=eval_loss,
        r_square=r_square,
        rl=all_rl,
        labels=all_labels,
    )


def evaluate_autoencoder(model, eval_loader, epoch, loss_fn, mode, dataset, logger):
    autoencoder, predictor = model['autoencoder'], model['predictor']
    encoder_auto = autoencoder.getencoder()
    decoder_auto = autoencoder.getdecoder()
    encoder_pred = predictor.getencoder()
    predhead_pred = predictor.getpredhead()

    encoder_auto.eval()
    decoder_auto.eval()
    encoder_pred.eval()
    predhead_pred.eval()

    eval_loss = 0
    all_index_true = []
    all_index_pred = []
    with torch.no_grad():
        for onehot, mask, _ in eval_loader:
            onehot, mask = onehot.cuda(), mask.cuda()

            vector_base = encoder_pred(onehot)
            rl = predhead_pred(vector_base)
            rl_repeat = rl.repeat(1, 25)
            rl_repeat = torch.reshape(rl_repeat, (-1, 1, 25, 1))
            vector_delta = encoder_auto(onehot)
            vector_combined = torch.concat([vector_base + vector_delta, rl_repeat], dim=1)
            seqs_pred = decoder_auto(vector_combined)
            seqs_pred = seqs_pred * mask

            # sum loss
            seqs_loss = loss_fn['autoencoder'](seqs_pred, onehot)
            eval_loss = eval_loss + seqs_loss.item() * onehot.size(0)
            # save the indexes to compute the accuracy at the end of an epoch
            index_true = torch.argmax(onehot, dim=1)
            index_pred = torch.argmax(seqs_pred, dim=1)
            all_index_true.append(index_true.cpu().data.numpy())
            all_index_pred.append(index_pred.cpu().data.numpy())
    # compute loss
    eval_loss /= len(eval_loader.dataset)
    # compute accuracy
    all_index_true, all_index_pred = np.concatenate(all_index_true), np.concatenate(all_index_pred)
    seqs_acc = accuracy(torch.Tensor(all_index_true), torch.Tensor(all_index_pred))
    logger.info('>>Epoch: {} \tMode: {} \tDataset: {} \tLoss: {:.6f} \tAccuracy: {:6f}'
                .format(epoch, mode, dataset, eval_loss, seqs_acc))

    return eval_loss, seqs_acc