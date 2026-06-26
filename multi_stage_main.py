import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils import add_dict_to_argparser, current_time, Logger, return_model_best
from datasets import load_data
from utils.multi_stage_train import (
    setup_seed, WSE, train_predictor, train_autoencoder, evaluate_predictor, evaluate_autoencoder)


def main():
    args = create_argparser().parse_args()
    setup_seed(seed=args.seed)

    if not os.path.isdir(args.root_path):
        os.mkdir(args.root_path)

    logger = Logger(os.path.join(args.root_path, 'run.log')).create_logger()
    logger.info(f'>>>Loading {args.dimension}D data...')
    data_loader = load_data(
        data_path=args.data_path,
        num_single_path=args.num_single_path,
        test_num=args.test_num,
        val_index=args.val_index,
        logger=logger,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dimension=args.dimension)

    if args.train:
        run_train(data_loader, logger, args)
    if args.test:
        if not args.train:
            args.root_path = args.checkpoint_root_path
        run_test(data_loader, logger, args)


def run_train(data_loader, logger, args):
    from models.TwoStage2D import Predictor, Autoencoder

    logger.info('>>>Training Predictor..')
    predictor = Predictor(dropout_rate=args.dropout_rate).to(args.device)
    optim_pred = optim.Adam(predictor.parameters(), lr=args.train_lr['predictor'])
    scheduler_pred = optim.lr_scheduler.StepLR(optim_pred, step_size=args.step_size)

    start_epoch = 0
    if args.checkpoint_root_path is not None:
        checkpoint_path = os.path.join(args.checkpoint_root_path, 'predictor', 'last.pkl')
        logger.info(f'loading from checkpoint: {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path)
        predictor.load_state_dict(checkpoint['model'], strict=True)
        optim_pred.load_state_dict(checkpoint['train_optim'])
        start_epoch = checkpoint['epoch']
        scheduler_pred.load_state_dict(checkpoint['scheduler'])

    from collections import defaultdict
    r2_record = defaultdict(list)

    predictor_dir_path = os.path.join(args.root_path, 'predictor')
    if not os.path.isdir(predictor_dir_path):
        os.mkdir(predictor_dir_path)

    for epoch in range(start_epoch + 1, args.predictor_epochs + 1):
        train_predictor(
            model=predictor,
            train_loader=data_loader['train'],
            epoch=epoch,
            epochs=args.predictor_epochs,
            optimizer=optim_pred,
            loss_fn=args.loss_function,
            b=args.b,
            logger=logger,
            flood=False)
        for name, data in data_loader.items():
            if 'val' in name:
                result = evaluate_predictor(
                    model=predictor,
                    eval_loader=data,
                    epoch=epoch,
                    loss_fn=args.loss_function,
                    mode='Validation',
                    dataset=name,
                    logger=logger)
                r2_record[name].append(result['r_square'])
        scheduler_pred.step()

        checkpoint = {
            "model": predictor.state_dict(),
            'train_optim': optim_pred.state_dict(),
            "epoch": epoch,
            'scheduler': scheduler_pred.state_dict()
        }
        torch.save(checkpoint, os.path.join(predictor_dir_path, 'last.pkl'))
        if r2_record[f'{args.observe_set}_val'][-1] == max(r2_record[f'{args.observe_set}_val']):
            logger.info('saving best at epoch {} ...'.format(epoch))
            torch.save(
                checkpoint,
                os.path.join(
                    predictor_dir_path,
                    'best_{}_{:.4f}.pkl'.format(epoch, r2_record[f'{args.observe_set}_val'][-1])))

    logger.info('>>>Training Autoencoder...')
    logger.info('Loading predictor from {}'.format(return_model_best(predictor_dir_path)['path']))
    predictor_trained = Predictor(dropout_rate=args.dropout_rate).to(args.device)
    predictor_state_dict = torch.load(return_model_best(predictor_dir_path)["path"])
    predictor_trained.load_state_dict(predictor_state_dict["model"], strict=True)

    autoencoder = Autoencoder(dropout_rate=args.dropout_rate, encoder=None).to(args.device)
    optim_ae = optim.Adam(autoencoder.parameters(), lr=args.train_lr['autoencoder'])
    scheduler_ae = optim.lr_scheduler.StepLR(optim_ae, step_size=args.step_size)

    autoencoder_dir_path = os.path.join(args.root_path, 'autoencoder')
    if not os.path.isdir(autoencoder_dir_path):
        os.mkdir(autoencoder_dir_path)

    acc_record = defaultdict(list)
    for epoch in range(1, args.autoencoder_epochs + 1):
        train_autoencoder(
            model={'autoencoder': autoencoder, 'predictor': predictor_trained},
            train_loader=data_loader['train'],
            epoch=epoch,
            epochs=args.autoencoder_epochs,
            optimizer=optim_ae,
            loss_fn=args.loss_function,
            b=args.b,
            logger=logger,
            flood=False
        )
        for name, data in data_loader.items():
            if 'val' in name:
                _, seqs_acc_epoch = evaluate_autoencoder(
                    model={'autoencoder': autoencoder, 'predictor': predictor_trained},
                    eval_loader=data,
                    epoch=epoch,
                    loss_fn=args.loss_function,
                    mode='Validation',
                    dataset=name,
                    logger=logger
                )
                acc_record[name].append(seqs_acc_epoch)
        scheduler_ae.step()
        torch.save(
            autoencoder.state_dict(), os.path.join(autoencoder_dir_path, 'last.pkl'))
    logger.info('Training completed successfully!')


def run_test(data_loader, logger, args):
    from models.TwoStage2D import Predictor, Autoencoder
    predictor_model_path = return_model_best(os.path.join(args.root_path, 'predictor'))["path"]
    logger.info('>>>Testing Predictor...')
    logger.info('Loading predictor from {}'.format(predictor_model_path))
    predictor_trained = Predictor(dropout_rate=args.dropout_rate).to(args.device)
    predictor_state_dict = torch.load(predictor_model_path)
    predictor_trained.load_state_dict(predictor_state_dict["model"], strict=True)
    for name, data in data_loader.items():
        if 'test' in name:
            evaluate_predictor(
                model=predictor_trained,
                eval_loader=data,
                epoch=1,
                loss_fn=args.loss_function,
                mode='Test',
                dataset=name,
                logger=logger
            )

    autoencoder_model_path = os.path.join(args.root_path, 'autoencoder', 'last.pkl')
    logger.info('>>>Testing Autoencoder...')
    logger.info('Loading autoencoder from {}'.format(autoencoder_model_path))
    autoencoder_trained = Autoencoder(dropout_rate=args.dropout_rate).to(args.device)
    autoencoder_state_dict = torch.load(autoencoder_model_path)
    autoencoder_trained.load_state_dict(autoencoder_state_dict, strict=True)
    for name, data in data_loader.items():
        if 'test' in name:
            evaluate_autoencoder(
                model={'autoencoder': autoencoder_trained, 'predictor': predictor_trained},
                eval_loader=data,
                epoch=1,
                loss_fn=args.loss_function,
                mode='Test',
                dataset=name,
                logger=logger
            )
    logger.info('Testing completed successfully!')


def create_argparser():
    defaults = dict(
        train=True,
        test=True,
        data_path=dict(
            egfp_m1pseudo_2='./data/GSM3130440_egfp_m1pseudo_2.xlsx',
            varying_length='./data/GSM4084997_varying_length_25to100.csv',
        ),
        seed=666,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dimension=2,
        num_single_path=220000,
        test_num=20000,
        val_index=200000,
        batch_size=256,
        predictor_epochs=50,  #
        autoencoder_epochs=50,
        num_workers=12,
        max_len=100,
        dropout_rate=0.2,
        step_size=10,
        save_interval=2,
        b=0.2,  # flood level
        train_lr={'predictor': 2e-04, 'autoencoder': 2e-04},
        loss_function={'predictor': WSE(), 'autoencoder': nn.BCELoss()},
        root_path=f'./logs/run_logs/multi_stage/{current_time()}',
        checkpoint_root_path=None,
        observe_set='egfp_m1pseudo_2',
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == '__main__':
    main()