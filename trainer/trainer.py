# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:58
# @Author  : zhoujun
import time

import torch
import torchvision.utils as vutils
from tqdm import tqdm

from base import BaseTrainer
from utils import WarmupPolyLR, runningScore, cal_text_score


class Trainer(BaseTrainer):
    def __init__(self, config, model, criterion, train_loader, validate_loader, metric_cls, post_process=None):
        super(Trainer, self).__init__(config, model, criterion)
        self.show_images_iter = self.config['trainer']['show_images_iter']
        self.train_loader = train_loader
        if validate_loader is not None:
            assert post_process is not None and metric_cls is not None
        self.validate_loader = validate_loader
        self.post_process = post_process
        self.metric_cls = metric_cls
        self.train_loader_len = len(train_loader)
        if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
            warmup_iters = config['lr_scheduler']['args']['warmup_epoch'] * self.train_loader_len
            if self.start_epoch > 1:
                self.config['lr_scheduler']['args']['last_epoch'] = (self.start_epoch - 1) * self.train_loader_len
            self.scheduler = WarmupPolyLR(self.optimizer, max_iters=self.epochs * self.train_loader_len,
                                          warmup_iters=warmup_iters, **config['lr_scheduler']['args'])
        if self.validate_loader is not None:
            self.logger_info(
                'train dataset has {} samples,{} in dataloader, validate dataset has {} samples,{} in dataloader'.format(
                    len(self.train_loader.dataset), self.train_loader_len, len(self.validate_loader.dataset), len(self.validate_loader)))
        else:
            self.logger_info('train dataset has {} samples,{} in dataloader'.format(len(self.train_loader.dataset), self.train_loader_len))

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        running_metric_text = runningScore(2)
        lr = self.optimizer.param_groups[0]['lr']

        for i, batch in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            # 数据进行转换和丢到gpu
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
            cur_batch_size = batch['img'].size()[0]

            preds = self.model(batch['img'])
            loss_dict = self.criterion(preds, batch)
            # backward
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()
            if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
                self.scheduler.step()
            # acc iou
            score_shrink_map = cal_text_score(preds[:, 0, :, :], batch['shrink_map'], batch['shrink_mask'], running_metric_text,
                                              thred=self.config['post_processing']['args']['thresh'])

            # loss 和 acc 记录到日志
            loss_str = 'loss: {:.4f}, '.format(loss_dict['loss'].item())
            for idx, (key, value) in enumerate(loss_dict.items()):
                loss_dict[key] = value.item()
                if key == 'loss':
                    continue
                loss_str += '{}: {:.4f}'.format(key, loss_dict[key])
                if idx < len(loss_dict) - 1:
                    loss_str += ', '

            train_loss += loss_dict['loss']
            acc = score_shrink_map['Mean Acc']
            iou_shrink_map = score_shrink_map['Mean IoU']

            if self.global_step % self.log_iter == 0:
                batch_time = time.time() - batch_start
                self.logger_info(
                    '[{}/{}], [{}/{}], global_step: {}, speed: {:.1f} samples/sec, acc: {:.4f}, iou_shrink_map: {:.4f}, {}, lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step, self.log_iter * cur_batch_size / batch_time, acc,
                        iou_shrink_map, loss_str, lr, batch_time))
                batch_start = time.time()

            if self.tensorboard_enable and self.config['local_rank'] == 0:
                # write tensorboard
                for key, value in loss_dict.items():
                    self.writer.add_scalar('TRAIN/LOSS/{}'.format(key), value, self.global_step)
                self.writer.add_scalar('TRAIN/ACC_IOU/acc', acc, self.global_step)
                self.writer.add_scalar('TRAIN/ACC_IOU/iou_shrink_map', iou_shrink_map, self.global_step)
                self.writer.add_scalar('TRAIN/lr', lr, self.global_step)
                if self.global_step % self.show_images_iter == 0:
                    # show images on tensorboard
                    self.inverse_normalize(batch['img'])
                    self.writer.add_images('TRAIN/imgs', batch['img'], self.global_step)
                    # shrink_labels and threshold_labels
                    shrink_labels = batch['shrink_map']
                    threshold_labels = batch['threshold_map']
                    shrink_labels[shrink_labels <= 0.5] = 0
                    shrink_labels[shrink_labels > 0.5] = 1
                    show_label = torch.cat([shrink_labels, threshold_labels])
                    show_label = vutils.make_grid(show_label.unsqueeze(1), nrow=cur_batch_size, normalize=False, padding=20, pad_value=1)
                    self.writer.add_image('TRAIN/gt', show_label, self.global_step)
                    # model output
                    show_pred = []
                    for kk in range(preds.shape[1]):
                        show_pred.append(preds[:, kk, :, :])
                    show_pred = torch.cat(show_pred)
                    show_pred = vutils.make_grid(show_pred.unsqueeze(1), nrow=cur_batch_size, normalize=False, padding=20, pad_value=1)
                    self.writer.add_image('TRAIN/preds', show_pred, self.global_step)
        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr, 'time': time.time() - epoch_start,
                'epoch': epoch}

    def _eval(self, epoch):
        self.model.eval()
        # torch.cuda.empty_cache()  # speed up evaluating after training finished
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                # 数据进行转换和丢到gpu
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start = time.time()
                preds = self.model(batch['img'])
                boxes, scores = self.post_process(batch, preds,is_output_polygon=self.metric_cls.is_output_polygon)
                total_frame += batch['img'].size()[0]
                total_time += time.time() - start
                raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
                raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        self.logger_info('FPS:{}'.format(total_frame / total_time))
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg

    def _on_epoch_finish(self):
        self.logger_info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'],
            self.epoch_result['lr']))
        net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)
        net_save_path_best = '{}/model_best.pth'.format(self.checkpoint_dir)

        if self.config['local_rank'] == 0:
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path)
            save_best = False
            if self.validate_loader is not None and self.metric_cls is not None:  # 使用f1作为最优模型指标
                recall, precision, hmean = self._eval(self.epoch_result['epoch'])

                if self.tensorboard_enable:
                    self.writer.add_scalar('EVAL/recall', recall, self.global_step)
                    self.writer.add_scalar('EVAL/precision', precision, self.global_step)
                    self.writer.add_scalar('EVAL/hmean', hmean, self.global_step)
                self.logger_info('test: recall: {:.6f}, precision: {:.6f}, f1: {:.6f}'.format(recall, precision, hmean))

                if hmean >= self.metrics['hmean']:
                    save_best = True
                    self.metrics['train_loss'] = self.epoch_result['train_loss']
                    self.metrics['hmean'] = hmean
                    self.metrics['precision'] = precision
                    self.metrics['recall'] = recall
                    self.metrics['best_model_epoch'] = self.epoch_result['epoch']
            else:
                if self.epoch_result['train_loss'] <= self.metrics['train_loss']:
                    save_best = True
                    self.metrics['train_loss'] = self.epoch_result['train_loss']
                    self.metrics['best_model_epoch'] = self.epoch_result['epoch']
            best_str = 'current best, '
            for k, v in self.metrics.items():
                best_str += '{}: {:.6f}, '.format(k, v)
            self.logger_info(best_str)
            if save_best:
                import shutil
                shutil.copy(net_save_path, net_save_path_best)
                self.logger_info("Saving current best: {}".format(net_save_path_best))
            else:
                self.logger_info("Saving checkpoint: {}".format(net_save_path))


    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger_info('{}:{}'.format(k, v))
        self.logger_info('finish train')
