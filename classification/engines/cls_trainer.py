import os, pprint
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import SGD
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from utils.log_utils import get_sha
import utils.log_utils as logging
from utils.optimizer import LARS
from utils import eval_util
from utils.scheduler import cosine_scheduler, multistep_scheduler
from utils.param_utils import get_params, has_batchnorms
from utils.param_utils import num_of_trainable_params
from utils.load_pretrained_models import *
from data.image_dataset import ImageDatasetLoader

import models


logger = logging.get_logger(__name__)


class CLSTrainer():
    def __init__(self, gpu, ngpus_per_node, cfg):
        self.cfg = cfg
        self.rank = 0
        self.node_rank = 0
        if cfg.DISTRIBUTED:            
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            self.node_rank = cfg.NODE_RANK
            self.rank = cfg.NODE_RANK * ngpus_per_node + gpu
            dist.init_process_group(
                backend=cfg.DIST_BACKEND, init_method=cfg.DIST_URL, 
                world_size=cfg.WORLD_SIZE, rank=self.rank,
            )        
        
        # Setup logging format.
        self.log_step     = cfg.LOG_STEP
        self.ckpt_path    = osp.join(cfg.OUTPUT_DIR, "checkpoints")
        self.tfboard_path = osp.join(cfg.OUTPUT_DIR, 'tensorboard')
        if self.rank == 0:
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            os.makedirs(self.ckpt_path, exist_ok=True)
            self.writer = SummaryWriter(self.tfboard_path)

        logging.setup_logging(self.rank, cfg.OUTPUT_DIR)
        logger.info(f"git:\n  {get_sha()}\n")
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

        self.gpu = gpu
        self.steps = 0
        self.pretrained_path = cfg.MODEL.PRE_TRAINED_PATH
        self.resume_path = cfg.TRAIN.CHECKPOINT_FILE_PATH
        self.train_batch_size = cfg.TRAIN.BATCH_SIZE # batch size per gpu
        self.val_batch_size = cfg.VAL.BATCH_SIZE
        self.global_batch_size = self.train_batch_size * cfg.WORLD_SIZE

        """get dataloader"""
        self.get_data()

        """get the train parameters"""
        self.total_epochs = cfg.SOLVER.TOTAL_EPOCHS
        self.warmup_steps = cfg.SOLVER.WARMUP_EPOCHS * self.num_examples // self.global_batch_size
        self.total_steps = self.total_epochs * self.num_examples // self.global_batch_size
        self.start_epoch = 0
        self.saved_epoch = cfg.TRAIN.CHECKPOINT_PERIOD
        self.start_warmup = cfg.SOLVER.START_WARMUP

        self.base_lr = cfg.SOLVER.BASE_LR * self.global_batch_size / 256.
        self.min_lr = cfg.SOLVER.MIN_LR
        
        """create the model"""
        self.create_model()
        if self.rank == 0:
            logger.info(f"Number of trainable parameters: {num_of_trainable_params(self.model)}")
        
        """get Loss class"""
        self.loss = nn.CrossEntropyLoss().cuda(self.gpu)

        """get optimizer"""
        params_groups = get_params(self.model, cfg.SOLVER.LR_FACTOR)
        if cfg.SOLVER.OPTIMIZING_METHOD == "AdamW":
            self.optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        elif cfg.SOLVER.OPTIMIZING_METHOD == "SGD":
            self.optimizer = torch.optim.SGD(params_groups, lr=self.base_lr, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV) 
        elif cfg.SOLVER.OPTIMIZING_METHOD == "LARS":
            self.optimizer = LARS(params_groups, lr=self.base_lr, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        elif cfg.SOLVER.OPTIMIZING_METHOD == "LBFGS":
            self.optimizer = torch.optim.LBFGS(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.base_lr, 
                max_iter=20, 
                max_eval=None, 
                tolerance_grad=1e-07, 
                tolerance_change=1e-09, 
                history_size=10, 
                line_search_fn=None
            )

        """init schedulers"""
        if cfg.SOLVER.SCHEDULER == "cos":
            self.lr_schedule = cosine_scheduler(
                self.base_lr,
                self.min_lr,
                self.total_epochs, len(self.train_loader),
                warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
                start_warmup_value=self.start_warmup
            )
        else:
            self.lr_schedule = multistep_scheduler(
                self.base_lr,
                cfg.SOLVER.MILESTONES,
                self.total_epochs, len(self.train_loader),
                gamma=0.1,
                warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
                start_warmup_value=self.start_warmup
            )
        
    def get_data(self):
        self.data_ins = ImageDatasetLoader(self.cfg, self.rank)
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.DATA.MEAN, std=self.cfg.DATA.STD)]
        )
        self.val_transforms = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.DATA.MEAN, std=self.cfg.DATA.STD)]
        )
        self.train_loader, self.num_examples   = self.data_ins.get_loader('TRAIN', self.train_batch_size, self.train_transforms)
        self.val_loader, self.val_num_examples = self.data_ins.get_loader('TEST',  self.val_batch_size,   self.val_transforms)

    def create_model(self):
        backbone_arch = self.cfg.MODEL.BACKBONE_ARCH
        eval_method = self.cfg.EVAL_METHOD
        logger.info("=> creating model '{}'".format(backbone_arch))
        if 'vit' in backbone_arch:
            self.model = models.__dict__[backbone_arch](patch_size=self.cfg.VIT.PATCH_SIZE)
        else:
            self.model = models.__dict__[backbone_arch](pretrained=False)

        num_ps_1 = num_of_trainable_params(self.model)

        if eval_method.lower() == 'semi':
            if 'mobilenet' in backbone_arch:
                enc_dim = self.model.classifier.in_features
                self.model.classifier = nn.Sequential(
                    OrderedDict([
                        ('fc1', nn.Linear(enc_dim, 2048)),
                        ('bn1', nn.BatchNorm1d(2048)),
                        ('act1', nn.GELU()),
                        ('fc2', nn.Linear(2048, 2048)),
                        ('bn2', nn.BatchNorm1d(2048)),
                        ('act2', nn.GELU()),
                        ('fc3', nn.Linear(2048, self.cfg.TRAIN.NUM_CLASSES))]
                    )
                )
            elif 'resnet' in backbone_arch:
                enc_dim = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                    OrderedDict([
                        ('fc1', nn.Linear(enc_dim, 2048)),
                        ('bn1', nn.BatchNorm1d(2048)),
                        ('act1', nn.GELU()),
                        ('fc2', nn.Linear(2048, 2048)),
                        ('bn2', nn.BatchNorm1d(2048)),
                        ('act2', nn.GELU()),
                        ('fc3', nn.Linear(2048, self.cfg.TRAIN.NUM_CLASSES))]
                    )
                )
            else:
                raise NotImplementedError
            
        elif eval_method.lower() == 'linear':
            for param in self.model.parameters():
                param.requires_grad = False
            if 'mobilenet' in backbone_arch:
                for name, param in self.model.named_parameters():
                    if name in ['classifier.weight', 'classifier.bias']:
                        param.requires_grad = True
                self.model.classifier.weight.data.normal_(mean=0.0, std=0.01)
                self.model.classifier.bias.data.zero_()
            elif 'resnet' in backbone_arch:
                for name, param in self.model.named_parameters():
                    if name in ['fc.weight', 'fc.bias']:
                        param.requires_grad = True
                self.model.fc.weight.data.normal_(mean=0.0, std=0.01)
                self.model.fc.bias.data.zero_()
            elif 'vit' in backbone_arch:
                input_dim = self.model.embed_dim * self.cfg.VIT.NUM_LAST_BLOCKS
                self.model.head = nn.Linear(input_dim, self.cfg.TRAIN.NUM_CLASSES)
                for name, param in self.model.named_parameters():
                    if name in ['head.weight', 'head.bias']:
                        param.requires_grad = True
                self.model.head.weight.data.normal_(mean=0.0, std=0.01)
                self.model.head.bias.data.zero_()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        num_ps_2 = num_of_trainable_params(self.model)

        if self.pretrained_path != '':
            state_dict = get_model_loader(backbone_arch, self.cfg.SSL_METHOD)(self.pretrained_path, list(self.model.state_dict().keys()), use_head=(eval_method.lower() == 'semi'))       
            msg = self.model.load_state_dict(state_dict, strict=False)
            logger.info(f'Pretrained weights found at {self.pretrained_path} and loaded with msg: {msg}')

        num_ps_3 = num_of_trainable_params(self.model)
        if self.rank == 0:
            logger.info(f"Number of trainable parameters: {num_ps_1}, {num_ps_2}, {num_ps_3}")
        
        if self.cfg.DISTRIBUTED:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                self.model.cuda(self.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu])
                logger.info(f'o==> model DDP set on node {self.node_rank} on GPU {self.gpu} with global rank {self.rank}' )
            else:
                self.model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = torch.nn.parallel.DistributedDataParallel(self.model)        
        else:
            torch.cuda.set_device(self.gpu)
            self.model = self.model.cuda(self.gpu)

    def resume_model(self):
        if self.resume_path == "":
            self.start_epoch = 0
            logger.info("--> No loaded checkpoint!")
        else:
            if osp.isfile(self.resume_path):
                logger.info(f"=> loading checkpoint {self.resume_path}")
                model_path = self.resume_path
                if self.gpu is None:
                    checkpoint = torch.load(model_path)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(self.gpu)
                    checkpoint = torch.load(model_path, map_location=loc)

                self.start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.steps = checkpoint['steps']

                logger.info(f"--> Loaded checkpoint '{model_path}' (epoch {self.start_epoch})")
            else:
                logger.info(f"=> no checkpoint found at {self.resume_path}")
  
    def save_checkpoint(self, epoch, best_acc, is_best):
        state = {
            'config': self.cfg,
            'epoch': epoch + 1,
            'steps': self.steps,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_acc': best_acc,
        }
        if not osp.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        if (epoch+1) % self.saved_epoch == 0 and self.rank == 0:
            torch.save(state,  osp.join(self.ckpt_path, self.cfg.MODEL.BACKBONE_ARCH + '_' + str(epoch) + '.pth.tar'))
        if is_best and self.rank == 0:
            torch.save(state, osp.join(self.ckpt_path, self.cfg.MODEL.BACKBONE_ARCH + '_best.pth.tar'))

    def train_epoch(self, epoch, printer=print):
        if self.cfg.DISTRIBUTED:
            self.data_ins.set_epoch(epoch)

        loss_meter = eval_util.AverageMeter()
        top1_meter = eval_util.AverageMeter()
        top5_meter = eval_util.AverageMeter()

        """
        Switch to eval mode:
        Under the protocol of linear classification on frozen features/models,
        it is not legitimate to change any part of the pre-trained model.
        BatchNorm in train mode may revise running mean/std (even if it receives
        no gradient), which are part of the model parameters too.
        """
        if self.cfg.EVAL_METHOD.lower() == 'linear':
            self.model.eval()
        else:
            self.model.train()
        
        if self.cfg.SOLVER.OPTIMIZING_METHOD != "LBFGS":
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr_schedule[self.steps] * param_group['lr_factor']       
        
        for i, (_, images, labels) in enumerate(self.train_loader):
            images = images.cuda(self.gpu, non_blocking=True)
            labels = labels.cuda(self.gpu, non_blocking=True)
            kwargs = {}
            if 'vit' in self.cfg.MODEL.BACKBONE_ARCH:
                kwargs['n'] = self.cfg.VIT.NUM_LAST_BLOCKS
            logits = self.model(images,**kwargs)
            
            if self.cfg.SOLVER.OPTIMIZING_METHOD != "LBFGS":
                loss = self.loss(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                def closure():
                    loss = self.loss(logits, labels)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    return loss
                loss = self.optimizer.step(closure=closure)

            self.steps += 1
            # measure accuracy and record loss
            acc1, acc5 = eval_util.accuracy(logits, labels, topk=(1, 5))
            loss_meter.update(loss.item(), images.size(0))
            top1_meter.update(acc1[0], images.size(0))
            top5_meter.update(acc5[0], images.size(0))

            if  self.steps % self.log_step == 0 and self.rank == 0:
                self.writer.add_scalar('lr', round(self.optimizer.param_groups[0]['lr'], 5), self.steps)
                self.writer.add_scalar('loss', loss_meter.avg, self.steps)
                self.writer.add_scalar('acc', top1_meter.avg, self.steps)
            
            if self.rank == 0 and self.steps % self.log_step == 0:
                logger.info(f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                        f'Lr {round(self.optimizer.param_groups[0]["lr"], 5)}\t'
                        f'Step {self.steps}\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Top1 {top1_meter.val:.4f} ({top1_meter.avg:.4f})\t'
                        f'Top5 {top5_meter.val:.4f} ({top5_meter.avg:.4f})\t')

    def validate(self, epoch, printer=print):
        loss_meter = eval_util.AverageMeter()
        top1_meter = eval_util.AverageMeter()
        top5_meter = eval_util.AverageMeter()
        
        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            for i, (_, images, labels) in enumerate(self.val_loader):
                images = images.cuda(self.gpu, non_blocking=True).contiguous()
                labels = labels.cuda(self.gpu, non_blocking=True).contiguous()    
                # forward and compute logits
                kwargs = {}
                if 'vit' in self.cfg.MODEL.BACKBONE_ARCH:
                    kwargs['n'] = self.cfg.VIT.NUM_LAST_BLOCKS
                logits = self.model(images,**kwargs)
                loss = self.loss(logits, labels)

                # measure accuracy and record loss
                acc1, acc5 = eval_util.accuracy(logits, labels, topk=(1, 5))
                loss_meter.update(loss.item(), images.size(0))
                top1_meter.update(acc1[0], images.size(0))
                top5_meter.update(acc5[0], images.size(0))
                
                if i % self.log_step == 0 and self.rank == 0:
                    logger.info(f'[{i}/{len(self.val_loader)}]\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Top1 {top1_meter.val:.4f} ({top1_meter.avg:.4f})\t'
                        f'Top5 {top5_meter.val:.4f} ({top5_meter.avg:.4f})\t')

            if self.rank == 0:
                logger.info(f'Finall Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                            f'Finall Top1 {top1_meter.val:.4f} ({top1_meter.avg:.4f})\t'
                            f'Finall Top5 {top5_meter.val:.4f} ({top5_meter.avg:.4f})\t')

            if  self.rank == 0:
                self.writer.add_scalar('val_loss', loss_meter.avg, epoch)
                self.writer.add_scalar('val_top1', top1_meter.avg, epoch)
                self.writer.add_scalar('val_top5', top5_meter.avg, epoch)

        return top1_meter.avg