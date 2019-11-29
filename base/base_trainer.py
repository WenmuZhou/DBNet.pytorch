# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:50
# @Author  : zhoujun

import os
import shutil
import pathlib
from pprint import pformat
import torch
from torch import nn

from utils import setup_logger


class BaseTrainer:
    def __init__(self, config, model, criterion, weights_init):
        config['trainer']['output_dir'] = os.path.join(str(pathlib.Path(os.path.abspath(__name__)).parent),
                                                       config['trainer']['output_dir'])
        config['name'] = config['name'] + '_' + model.name
        self.save_dir = os.path.join(config['trainer']['output_dir'], config['name'])
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')

        if config['trainer']['resume_checkpoint'] == '' and config['trainer']['finetune_checkpoint'] == '':
            shutil.rmtree(self.save_dir, ignore_errors=True)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.global_step = 0
        self.start_epoch = 1
        self.config = config

        self.model = model
        self.criterion = criterion
        # logger and tensorboard
        self.tensorboard_enable = self.config['trainer']['tensorboard']
        self.epochs = self.config['trainer']['epochs']
        self.display_interval = self.config['trainer']['display_interval']
        if self.tensorboard_enable:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.save_dir)

        self.logger = setup_logger(os.path.join(self.save_dir, 'train_log'))
        self.logger.info(pformat(self.config))

        # device
        torch.manual_seed(self.config['trainer']['seed'])  # 为CPU设置随机种子
        if len(self.config['trainer']['gpus']) > 0 and torch.cuda.is_available():
            self.with_cuda = True
            torch.backends.cudnn.benchmark = True
            self.logger.info(
                'train with gpu {} and pytorch {}'.format(self.config['trainer']['gpus'], torch.__version__))
            self.gpus = {i: item for i, item in enumerate(self.config['trainer']['gpus'])}
            self.device = torch.device("cuda:0")
            torch.cuda.manual_seed(self.config['trainer']['seed'])  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(self.config['trainer']['seed'])  # 为所有GPU设置随机种子
        else:
            self.with_cuda = False
            self.logger.info('train with cpu and pytorch {}'.format(torch.__version__))
            self.device = torch.device("cpu")
        self.logger.info('device {}'.format(self.device))
        self.metrics = {'recall': 0, 'precision': 0, 'hmean': 0, 'train_loss': float('inf'), 'best_model': ''}

        self.optimizer = self._initialize('optimizer', torch.optim, model.parameters())

        if self.config['trainer']['resume_checkpoint'] != '':
            self._laod_checkpoint(self.config['trainer']['resume_checkpoint'], resume=True)
        elif self.config['trainer']['finetune_checkpoint'] != '':
            self._laod_checkpoint(self.config['trainer']['finetune_checkpoint'], resume=False)
        else:
            if weights_init is not None:
                model.apply(weights_init)
        if self.config['lr_scheduler']['type'] != 'PolynomialLR':
            self.scheduler = self._initialize('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)

        # 单机多卡
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

        if self.tensorboard_enable:
            try:
                # add graph
                dummy_input = torch.zeros(1, self.config['data_loader']['args']['dataset']['img_channel'],
                                          self.config['data_loader']['args']['dataset']['input_size'],
                                          self.config['data_loader']['args']['dataset']['input_size']).to(self.device)
                self.writer.add_graph(model, dummy_input)
            except:
                import traceback
                # self.logger.error(traceback.format_exc())
                self.logger.warn('add graph to tensorboard failed')

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            try:
                self.epoch_result = self._train_epoch(epoch)
                if self.config['lr_scheduler']['type'] != 'PolynomialLR':
                    self.scheduler.step()
                self._on_epoch_finish()
            except torch.cuda.CudaError:
                self._log_memory_usage()
        if self.tensorboard_enable:
            self.writer.close()
        self._on_train_finish()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _eval(self):
        """
        eval logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _on_epoch_finish(self):
        raise NotImplementedError

    def _on_train_finish(self):
        raise NotImplementedError

    def _log_memory_usage(self):
        if not self.with_cuda:
            return

        template = """Memory Usage: \n{}"""
        usage = []
        for deviceID, device in self.gpus.items():
            deviceID = int(deviceID)
            allocated = torch.cuda.memory_allocated(deviceID) / (1024 * 1024)
            cached = torch.cuda.memory_cached(deviceID) / (1024 * 1024)

            usage.append('    CUDA: {}  Allocated: {} MB Cached: {} MB \n'.format(device, allocated, cached))

        content = ''.join(usage)
        content = template.format(content)

        self.logger.debug(content)

    def _save_checkpoint(self, epoch, file_name, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        filename = os.path.join(self.checkpoint_dir, file_name)
        torch.save(state, filename)
        if save_best:
            shutil.copy(filename, os.path.join(self.checkpoint_dir, 'model_best.pth'))
            self.logger.info("Saving current best: {}".format(file_name))
        else:
            self.logger.info("Saving checkpoint: {}".format(filename))

    def _laod_checkpoint(self, checkpoint_path, resume):
        """
        Resume from saved checkpoints
        :param checkpoint_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if resume:
            self.global_step = checkpoint['global_step']
            self.start_epoch = checkpoint['epoch'] + 1
            self.config['lr_scheduler']['args']['last_epoch'] = self.start_epoch
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            if self.with_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            self.logger.info("resume from checkpoint {} (epoch {})".format(checkpoint_path, self.start_epoch))
        else:
            self.logger.info("finetune from checkpoint {}".format(checkpoint_path))

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.config[name]['type']
        module_args = self.config[name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)
