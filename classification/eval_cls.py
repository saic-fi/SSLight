import os, torch, random
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import numpy as np

import engines as trainer_tools
from config.parser import load_config, parse_args

args = parse_args()
cfg = load_config(args)


def run_task(cfg):
    if cfg.SEED is not None:
        """
        Fix random seeds.
        """
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed_all(cfg.SEED)
        np.random.seed(cfg.SEED)

    ngpus_per_node = torch.cuda.device_count()

    if cfg.DISTRIBUTED:
        if cfg.DIST_URL == "env://":
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            cfg.WORLD_SIZE = ngpus_per_node * cfg.N_NODES

            # Set the master address and port according to mlp 
            if args.job_name is not None:
                os.environ['MASTER_ADDR'] = args.job_name + '-master'
            else:
                os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '9887'
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        # Simply call main_worker function
        main_worker(0, ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, cfg):
    cudnn.benchmark = True
    trainer = trainer_tools.__dict__['CLSTrainer'](gpu, ngpus_per_node, cfg)
    trainer.resume_model()
    acc1 = trainer.validate(0)
    print('Accuracy is: ', acc1.item())

            
if __name__ == '__main__':
    run_task(cfg)
