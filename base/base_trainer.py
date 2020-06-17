# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:50
# @Author  : zhoujun

import os
import pathlib
import shutil
from pprint import pformat

import anyconfig
import torch

from utils import setup_logger


class BaseTrainer:
    def __init__(self, config, model, criterion):
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
        self.start_epoch = 0
        self.config = config
        self.model = model
        self.criterion = criterion
        # logger and tensorboard
        self.tensorboard_enable = self.config['trainer']['tensorboard']
        self.epochs = self.config['trainer']['epochs']
        self.log_iter = self.config['trainer']['log_iter']
        if config['local_rank'] == 0:
            anyconfig.dump(config, os.path.join(self.save_dir, 'config.yaml'))
            self.logger = setup_logger(os.path.join(self.save_dir, 'train.log'))
            self.logger_info(pformat(self.config))

        # device
        torch.manual_seed(self.config['trainer']['seed'])  # 为CPU设置随机种子
        if torch.cuda.device_count() > 0 and torch.cuda.is_available():
            self.with_cuda = True
            torch.backends.cudnn.benchmark = True
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.config['trainer']['seed'])  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(self.config['trainer']['seed'])  # 为所有GPU设置随机种子
        else:
            self.with_cuda = False
            self.device = torch.device("cpu")
        self.logger_info('train with device {} and pytorch {}'.format(self.device, torch.__version__))
        # metrics
        self.metrics = {'recall': 0, 'precision': 0, 'hmean': 0, 'train_loss': float('inf'),'best_model_epoch':0}

        self.optimizer = self._initialize('optimizer', torch.optim, model.parameters())

        # resume or finetune
        if self.config['trainer']['resume_checkpoint'] != '':
            self._load_checkpoint(self.config['trainer']['resume_checkpoint'], resume=True)
        elif self.config['trainer']['finetune_checkpoint'] != '':
            self._load_checkpoint(self.config['trainer']['finetune_checkpoint'], resume=False)

        if self.config['lr_scheduler']['type'] != 'WarmupPolyLR':
            self.scheduler = self._initialize('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)

        self.model.to(self.device)

        if self.tensorboard_enable and config['local_rank'] == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.save_dir)
            try:
                # add graph
                in_channels = 3 if config['dataset']['train']['dataset']['args']['img_mode'] != 'GRAY' else 1
                dummy_input = torch.zeros(1, in_channels, 640, 640).to(self.device)
                self.writer.add_graph(self.model, dummy_input)
                torch.cuda.empty_cache()
            except:
                import traceback
                self.logger.error(traceback.format_exc())
                self.logger.warn('add graph to tensorboard failed')
        # 分布式训练
        if torch.cuda.device_count() > 1:
            local_rank = config['local_rank']
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False,
                                                                   find_unused_parameters=True)
        # make inverse Normalize
        self.UN_Normalize = False
        for t in self.config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] == 'Normalize':
                self.normalize_mean = t['args']['mean']
                self.normalize_std = t['args']['std']
                self.UN_Normalize = True

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            if self.config['distributed']:
                self.train_loader.sampler.set_epoch(epoch)
            self.epoch_result = self._train_epoch(epoch)
            if self.config['lr_scheduler']['type'] != 'WarmupPolyLR':
                self.scheduler.step()
            self._on_epoch_finish()
        if self.config['local_rank'] == 0 and self.tensorboard_enable:
            self.writer.close()
        self._on_train_finish()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _eval(self, epoch):
        """
        eval logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _on_epoch_finish(self):
        raise NotImplementedError

    def _on_train_finish(self):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, file_name):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        state_dict = self.model.module.state_dict() if self.config['distributed'] else self.model.state_dict()
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        filename = os.path.join(self.checkpoint_dir, file_name)
        torch.save(state, filename)

    def _load_checkpoint(self, checkpoint_path, resume):
        """
        Resume from saved checkpoints
        :param checkpoint_path: Checkpoint path to be resumed
        """
        self.logger_info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'], strict=resume)
        if resume:
            self.global_step = checkpoint['global_step']
            self.start_epoch = checkpoint['epoch']
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
            self.logger_info("resume from checkpoint {} (epoch {})".format(checkpoint_path, self.start_epoch))
        else:
            self.logger_info("finetune from checkpoint {}".format(checkpoint_path))

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.config[name]['type']
        module_args = self.config[name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def inverse_normalize(self, batch_img):
        if self.UN_Normalize:
            batch_img[:, 0, :, :] = batch_img[:, 0, :, :] * self.normalize_std[0] + self.normalize_mean[0]
            batch_img[:, 1, :, :] = batch_img[:, 1, :, :] * self.normalize_std[1] + self.normalize_mean[1]
            batch_img[:, 2, :, :] = batch_img[:, 2, :, :] * self.normalize_std[2] + self.normalize_mean[2]

    def logger_info(self, s):
        if self.config['local_rank'] == 0:
            self.logger.info(s)
