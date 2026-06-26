import argparse
import logging
import os
import random
import time
from collections import OrderedDict
from itertools import islice

import numpy as np
import torch


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def setup_seed(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  #


def current_time():
    """
    return a str about current date
    """

    return time.strftime('%Y-%m-%d-%H-%M')


class Logger(object):
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self):
        return self.create_logger()

    def create_logger(self):
        logger = logging.getLogger()  # 设定日志对象
        logger.setLevel(logging.INFO)  # 设定日志等级

        file_handler = logging.FileHandler(self.log_path)  # 文件输出
        console_handler = logging.StreamHandler()  # 控制台输出

        # 输出格式
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s "
        )

        file_handler.setFormatter(formatter)  # 设置文件输出格式
        console_handler.setFormatter(formatter)  # 设施控制台输出格式
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def return_model_best(dir_path):
    """
    model name: {xxx}_{epoch/step}_{performance}.{pkl/pt/pth} or last.pkl
    """

    file_list = os.listdir(dir_path)
    if 'last.pkl' in file_list:
        file_list.remove('last.pkl')
    info_list = []
    for file_name in file_list:
        file_name, file_extension = os.path.splitext(file_name)  # 去掉后缀
        extra, epoch, performance = file_name.split('_')
        info_list.append((extra, epoch, performance, file_extension))
    sorted_list = sorted(info_list, key=lambda x: x[2], reverse=True)  # 降序
    extra_best, epoch_best, performance_best, file_extension_best = sorted_list[0]
    model_best = f'{extra_best}_{epoch_best}_{performance_best}{file_extension_best}'
    model_best_path = os.path.join(dir_path, model_best)

    return {'epoch': epoch_best, 'performance': performance_best,
            'model_name': model_best, 'path': model_best_path}


def chunk(it, limit):
    it = iter(it)
    return iter(lambda: list(islice(it, limit)), [])


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda")
    return torch.device("cpu")


def rm_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict