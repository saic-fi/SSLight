import time
from datetime import datetime, date

import torch
import torch.nn as nn
import torch.distributed as dist

from sslight.engine.trainer import Trainer
from sslight.utils.param_utils import clip_gradients, cancel_gradients_last_layer
import sslight.utils.log_utils as logging


logger = logging.get_logger(__name__)


class MOCOTrainer(Trainer):
    def __init__(self, gpu, ngpus_per_node, cfg):
        super().__init__(gpu, ngpus_per_node, cfg)

    def train_epoch(self, epoch):
        metrics, times = {}, {}
        for t in ['io', 'forward', 'backward', 'batch']:
            times[t] = logging.AverageMeter()
        for t in ['loss', 'ssl_loss']:
            metrics[t] = logging.AverageMeter()
        if self.cfg.TRAIN.JOINT_LINEAR_PROBE:
            for t in ['sup_loss', 'sup_accu']:
                metrics[t] = logging.AverageMeter()
        
        if self.cfg.DISTRIBUTED:
            self.data_ins.set_epoch(epoch)

        self.model.train()
        end = time.time()
        data_tflag = time.time()
        num_param_groups = len(self.optimizer.param_groups)
        for i, (_, images, labels) in enumerate(self.loader):
            images = [im.cuda(self.gpu, non_blocking=True) for im in images]
            labels = labels.cuda(self.gpu, non_blocking=True).contiguous()
            times['io'].update(time.time() - data_tflag)

            for j, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.lr_schedule[self.steps]
                if j == 0:
                    param_group["weight_decay"] = self.wd_schedule[self.steps]
                else:
                    param_group["weight_decay"] = self.wd_schedule[self.steps] * param_group["wd_factor"]
            
            # forward
            with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                tflag = time.time()
                src_feats, sup_logits, _, _, moco_logits, moco_labels  = self.model(images)
                times['forward'].update(time.time() - tflag)
                ssl_loss = self.loss(moco_logits, moco_labels)
                if self.cfg.TRAIN.JOINT_LINEAR_PROBE:
                    sup_loss = nn.functional.cross_entropy(sup_logits, labels)
                    loss = ssl_loss + sup_loss
                    sup_pred = torch.max(sup_logits, dim=-1)[1]
                    sup_accu = torch.eq(sup_pred.long(), labels.long()).float().mean()
                else:
                    loss = ssl_loss


            tflag = time.time()
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            param_norms = None
            # loss.backward()
            # self.model.module.student_clip_grad(epoch)
            # self.optimizer.step()
            
            if self.fp16_scaler is None:
                loss.backward()
                if self.cfg.SOLVER.CLIP_GRAD > 0:
                    param_norms = clip_gradients(self.model.module.encoder_q, self.cfg.SOLVER.CLIP_GRAD)
                cancel_gradients_last_layer(epoch, self.model.module.encoder_q, self.cfg.SOLVER.FREEZE_LAST_LAYER)
                self.optimizer.step()
            else:
                self.fp16_scaler.scale(loss).backward()
                if self.cfg.SOLVER.CLIP_GRAD > 0:
                    self.fp16_scaler.unscale_(self.optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = clip_gradients(self.model.module.encoder_q, self.cfg.SOLVER.CLIP_GRAD)
                cancel_gradients_last_layer(epoch, self.model.module.encoder_q, self.cfg.SOLVER.FREEZE_LAST_LAYER)
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()
            times['backward'].update(time.time() - tflag)
            
            torch.cuda.synchronize()
            ############################################################################
            for k, meter in metrics.items():
                meter.update(locals()[k].item(), n=int(images[0].size(0)))
            ############################################################################
            
            if self.cfg.DISTRIBUTED:
                dist.barrier()
                global_loss = torch.tensor([metrics['loss'].val, metrics['loss'].avg], device=self.gpu)
                # Get the sum of results from all GPUs
                dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
                global_loss /= self.cfg.WORLD_SIZE

            if self.steps % self.log_step == 0 and self.rank == 0:
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.steps)
                self.writer.add_scalar('wd', self.optimizer.param_groups[0]['weight_decay'], self.steps)
                self.writer.add_scalar('momentum', self.momentum_schedule[self.steps], self.steps)
                self.writer.add_scalar('loss', metrics['loss'].val, self.steps)
                self.writer.add_scalar('ssl_loss', metrics['ssl_loss'].val, self.steps)
                if self.cfg.TRAIN.JOINT_LINEAR_PROBE:
                    self.writer.add_scalar('sup_loss', metrics['sup_loss'].val, self.steps)
                    self.writer.add_scalar('sup_accu', metrics['sup_accu'].val, self.steps)
                self.writer.add_scalar('loss_g_val', global_loss[0], self.steps)
                self.writer.add_scalar('loss_g_avg', global_loss[1], self.steps)
                
                if self.cfg.LOG_GRAD:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            self.writer.add_scalar(name+'.data', torch.mean(torch.abs(param.data)).item(), self.steps)
                            if param.grad is None:
                                self.writer.add_scalar(name+'.grad', 0.0, self.steps)
                            else:
                                self.writer.add_scalar(name+'.grad', torch.mean(torch.abs(param.grad)).item(), self.steps)

            # Print log info
            if self.rank == 0 and self.steps % self.log_step == 0:
                log_time = date.today().strftime("%b-%d-%Y") + "_" + datetime.now().strftime("%H-%M-%S")
                mm = self.momentum_schedule[self.steps]
                metric_info = \
                    f'Epoch: [{epoch}][{i}/{len(self.loader)}]\t' + \
                    f'Step {self.steps}\t' + \
                    f'lr {round(self.optimizer.param_groups[0]["lr"], 5)}\t' + \
                    f'wd {self.optimizer.param_groups[0]["weight_decay"]:.5f}\t' + \
                    f'mm {mm:.5f}\t' + \
                    f'Loss {metrics["loss"].val:.4f} ({metrics["loss"].avg:.4f})\t' + \
                    f'SSL loss {metrics["ssl_loss"].val:.4f} ({metrics["ssl_loss"].avg:.4f})\t' + \
                    f'Global Loss {global_loss[0]:.4f} ({global_loss[1]:.4f})\t'
                if self.cfg.TRAIN.JOINT_LINEAR_PROBE:
                    metric_info = metric_info +  f'Sup loss {metrics["sup_loss"].val:.4f} ({metrics["sup_loss"].avg:.4f})\t' + f'Sup accu {metrics["sup_accu"].val:.4f} ({metrics["sup_accu"].avg:.4f})\t'
                time_info = \
                    f'IO time {times["io"].val:.4f} ({times["io"].avg:.4f})\t' + \
                    f'Forward time {times["forward"].val:.4f} ({times["forward"].avg:.4f})\t' + \
                    f'Backward time {times["backward"].val:.4f} ({times["backward"].avg:.4f})\t' + \
                    f'Batch time {times["batch"].val:.4f} ({times["batch"].avg:.4f})\t' + \
                    f'Current time {log_time}\t'
                logger.info(metric_info+time_info)
            
            self.steps += 1
            times['batch'].update(time.time() - end)
            end = time.time()
            data_tflag = time.time()