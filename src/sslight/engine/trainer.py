import os, pprint
import os.path as osp
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

from sslight.model import *
from sslight.utils.loss import *
from sslight.utils import eval_utils
from sslight.utils.optimizer import LARS
from sslight.utils.scheduler import cosine_scheduler, multistep_scheduler
from sslight.utils.param_utils import get_params, has_batchnorms
from sslight.utils.param_utils import num_of_trainable_params

from data.transforms import *
from data.loader  import ImageDatasetLoader


from sslight.utils.log_utils import get_sha
import sslight.utils.log_utils as logging
import sslight.backbone.vision_transformer as vits


logger = logging.get_logger(__name__)


class Trainer():
    def __init__(self, gpu, ngpus_per_node, cfg):
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
        self.cfg = cfg
        logger.info(f"git:\n  {get_sha()}\n")
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

        self.gpu = gpu
        self.steps = 0
        self.queue_length = cfg.SWAV.QUEUE_LENGTH
        self.resume_path = cfg.TRAIN.CHECKPOINT_FILE_PATH
        self.train_batch_size = cfg.TRAIN.BATCH_SIZE # batch size per gpu
        self.global_batch_size = self.train_batch_size * cfg.WORLD_SIZE

        """queue initialization"""
        # build the queue
        self.queue = None
        if self.queue_length > 0:
            self.queue_path = osp.join(cfg.OUTPUT_DIR, "queue" + str(self.rank) + ".pth")
            if osp.isfile(self.queue_path):
                self.queue = torch.load(self.queue_path)["queue"]
            # the queue needs to be divisible by the batch size
            self.queue_length -= self.queue_length % self.global_batch_size

        """get dataloader"""
        self.get_data()

        """get the train parameters!"""
        self.total_epochs = cfg.SOLVER.TOTAL_EPOCHS
        self.warmup_steps = cfg.SOLVER.WARMUP_EPOCHS * self.num_examples // self.global_batch_size
        self.total_steps = self.total_epochs * self.num_examples // self.global_batch_size
        self.start_epoch = 0
        self.saved_epoch = cfg.TRAIN.CHECKPOINT_PERIOD
        self.start_warmup = cfg.SOLVER.START_WARMUP

        # linear scaling rule
        self.base_lr = cfg.SOLVER.BASE_LR * self.global_batch_size / 256.
        self.min_lr  = cfg.SOLVER.MIN_LR

        self.fp16_scaler = None
        if cfg.USE_FP16:
            self.fp16_scaler = torch.cuda.amp.GradScaler()
            logger.info("Using FP16 during training")
        
        """create the model"""
        self.create_model()
        if self.rank == 0:
            logger.info(f"Number of trainable parameters: {num_of_trainable_params(self.model)}")
        
        """get Loss class"""
        self.get_loss()

        """get optimizer"""
        params_groups = get_params(cfg, [self.model])
        if cfg.SOLVER.OPTIMIZING_METHOD == "AdamW":
            self.optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        elif cfg.SOLVER.OPTIMIZING_METHOD == "SGD":
            self.optimizer = torch.optim.SGD(params_groups, lr=self.base_lr, momentum=cfg.SOLVER.MOMENTUM)  
        elif cfg.SOLVER.OPTIMIZING_METHOD == "LARS":
            self.optimizer = LARS(params_groups, lr=self.base_lr, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        else:
            assert False, f"Unknow optimizer: {cfg.SOLVER.OPTIMIZING_METHOD}"

        if cfg.SOLVER.SCHEDULER == "cos":
            self.lr_schedule = cosine_scheduler(
                self.base_lr,
                self.min_lr,
                self.total_epochs, len(self.loader),
                warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
                start_warmup_value=self.start_warmup
            )
        else:
            self.lr_schedule = multistep_scheduler(
                self.base_lr,
                cfg.SOLVER.MILESTONES,
                self.total_epochs, len(self.loader),
                gamma=0.1,
                warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
                start_warmup_value=self.start_warmup
            )

        self.wd_schedule = cosine_scheduler(
            cfg.SOLVER.WEIGHT_DECAY,
            cfg.SOLVER.WEIGHT_DECAY_END,
            self.total_epochs, len(self.loader),
        )
        
        self.momentum_schedule = cosine_scheduler(
            cfg.MODEL.MODEL_MOMENTUM, cfg.MODEL.MODEL_MOMENTUM_END,
            self.total_epochs, len(self.loader)
        )

    def get_loss(self):
        loss_class = globals()[self.cfg.SSL_METHOD+ 'Loss']
        self.loss = loss_class(self.cfg, self.gpu)

    def get_data(self):
        data_aug = globals()['DataAugmentation' + self.cfg.SSL_METHOD]

        self.stage = self.cfg.STAGE
        self.data_ins = ImageDatasetLoader(self.cfg, self.rank)
        self.transforms = data_aug(self.cfg)
        logger.info('START DATA LOADING ... ')
        self.loader, self.num_examples = self.data_ins.get_loader(self.stage, self.train_batch_size, self.transforms)

        val_cfg = deepcopy(self.cfg)
        val_cfg.STAGE = 'TEST'
        self.val_data_ins = ImageDatasetLoader(val_cfg, self.rank)
        val_transforms = data_aug(val_cfg)
        logger.info('LOADING VAL SPLIT... ')
        self.val_loader, self.val_num_examples = self.val_data_ins.get_loader('TEST', self.train_batch_size, val_transforms)

    def create_model(self):
        logger.info("=> creating model '{}'".format(self.cfg.SSL_METHOD))
        ssl_class = globals()[self.cfg.SSL_METHOD]
        self.model = ssl_class(self.cfg)
    
        if self.cfg.DISTRIBUTED:
            if has_batchnorms(self.model):
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                self.model.cuda(self.gpu)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu])
                logger.info(f'o==> model DDP set on node {self.node_rank} on GPU {self.gpu} with global rank {self.rank}' )
            else:
                self.model.cuda()
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
                msg = self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if self.cfg.SSL_METHOD.lower() == 'dino' and 'loss' in checkpoint:
                    self.loss.load_state_dict(checkpoint['loss'])
                self.steps = checkpoint['steps']
                if self.cfg.USE_FP16 is True:
                    self.fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])

                logger.info(f"--> Loaded checkpoint '{model_path}' (epoch {self.start_epoch}), with msg: {msg}")
            else:
                logger.info(f"=> no checkpoint found at {self.resume_path}")
  
    def save_checkpoint(self, epoch):
        state = {
            'config': self.cfg,
            'epoch': epoch + 1,
            'steps': self.steps,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.cfg.SSL_METHOD.lower() == 'dino':
            state['loss'] = self.loss.state_dict()
        if self.fp16_scaler is not None:
            state['fp16_scaler'] = self.fp16_scaler.state_dict()
        if not osp.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        if (epoch+1) % self.saved_epoch == 0 and self.rank == 0:
            torch.save(state,  os.path.join(self.ckpt_path , self.cfg.SSL_METHOD + '_' 
                            + self.cfg.MODEL.BACKBONE_ARCH + '_' + str(epoch) + '.pth.tar'))
        ############################################################################
        if self.rank == 0:
            torch.save(state, osp.join(self.ckpt_path, 
                self.cfg.SSL_METHOD + '_' + self.cfg.MODEL.BACKBONE_ARCH + '_last_ckpt.pth.tar'))
        if self.queue is not None:
            torch.save({"queue": self.queue}, self.queue_path)

    @torch.no_grad()
    def knn_validate(self, epoch, printer=print):
        self.model.eval()
        features = None
        if self.cfg.TRAIN.JOINT_LINEAR_PROBE:
            sup_accu_meter = logging.AverageMeter()

        with torch.no_grad():
            for i, (index, images, labels) in tqdm(enumerate(self.val_loader), disable=(self.rank != 0)):
                ##############################################
                # Preparing data
                ##############################################
                samples = images[0].cuda(self.gpu, non_blocking=True).contiguous()
                labels  = labels.cuda(self.gpu, non_blocking=True).contiguous()
                index   = index.cuda(self.gpu, non_blocking=True).contiguous()

                feats, sup_logits = self.model(samples, extract_features_only=True)
                feats = torch.cat([feats, labels[:,None]], -1).float()
                if self.cfg.TRAIN.JOINT_LINEAR_PROBE:
                    sup_pred = torch.max(sup_logits, dim=-1)[1]
                    sup_accu = torch.eq(sup_pred.long(), labels.long()).float().mean()
                    sup_accu_meter.update(sup_accu.item(), len(samples))

                # init storage feature matrix
                if self.rank == 0 and features is None:
                    features = torch.zeros(len(self.val_loader.dataset), feats.shape[-1])
                    features = features.cuda(non_blocking=True)
                    print(f"Storing features into tensor of shape {features.shape}")

                # get indexes from all processes
                y_all = torch.empty(self.cfg.WORLD_SIZE, index.size(0), dtype=index.dtype, device=index.device)
                y_l = list(y_all.unbind(0))
                y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
                y_all_reduce.wait()
                index_all = torch.cat(y_l)
                # share features between processes
                feats_all = torch.empty(
                    self.cfg.WORLD_SIZE,
                    feats.size(0),
                    feats.size(1),
                    dtype=feats.dtype,
                    device=feats.device,
                )
                output_l = list(feats_all.unbind(0))
                output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
                output_all_reduce.wait()
                # update storage feature matrix
                if self.rank == 0:
                    features.index_copy_(0, index_all, torch.cat(output_l))
                    # features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
                if self.cfg.DISTRIBUTED:
                    dist.barrier()
        
        if self.cfg.TRAIN.JOINT_LINEAR_PROBE:
            global_stats = torch.tensor([sup_accu_meter.sum, sup_accu_meter.count], device=self.gpu)
            if self.cfg.DISTRIBUTED:
                dist.barrier()
                dist.reduce(global_stats, 0, op=dist.ReduceOp.SUM)
        
        if self.rank == 0:
            xs = features[:,:-1]
            ys = features[:,-1].long()
            xs = nn.functional.normalize(xs, dim=-1)
            top1, top5, nn_inds = eval_utils.knn_classifier(
                xs, ys,
                xs, ys,
                20, 0.07, offset=1)
            # logging.pickle_save('nn_inds.pkl', nn_inds)
            self.writer.add_scalar('knn_top1', top1, epoch)
            self.writer.add_scalar('knn_top5', top5, epoch)

            if self.cfg.TRAIN.JOINT_LINEAR_PROBE:
                global_stats = global_stats.cpu().data
                g_accu = float(global_stats[0]/global_stats[1])
                self.writer.add_scalar('linear probe', g_accu, epoch)
                logger.info(f'Epoch [{epoch}] KNN: top1 ({top1:.4f}), top5 ({top5:.4f}), Linear probe ({g_accu:.4f})\t')
            else:
                logger.info(f'Epoch [{epoch}] KNN: top1 ({top1:.4f}), top5 ({top5:.4f})\t')

        
        if self.cfg.DISTRIBUTED:
            dist.barrier()