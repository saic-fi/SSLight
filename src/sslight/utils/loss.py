import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist


class DINOLoss(nn.Module):
    def __init__(self, cfg, gpu):
        super().__init__()
        self.cfg = cfg
        self.student_temp = cfg.DINO.STUDENT_TEMP
        self.teacher_temp = cfg.DINO.TEACHER_TEMP
        self.warmup_teacher_temp = cfg.DINO.WARMUP_TEACHER_TEMP
        self.warmup_teacher_temp_epochs = cfg.DINO.WARMUP_TEACHER_TEMP_EPOCHS

        self.out_dim = cfg.DINOHead.OUTPUT_DIM
        self.ncrops = sum(cfg.MULTI_VIEWS_TRANSFORMS.NMB_CROPS) 
        self.nepochs = cfg.SOLVER.TOTAL_EPOCHS
        self.register_buffer("center", torch.zeros(1, self.out_dim).cuda(gpu))
        self.center_momentum = cfg.DINO.CENTER_MOMENTUM
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(self.warmup_teacher_temp, self.teacher_temp, self.warmup_teacher_temp_epochs),
            np.ones(self.nepochs - self.warmup_teacher_temp_epochs) * self.teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        # type: (torch.Tensor, torch.Tensor, int) -> torch.Tensor
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = nn.functional.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.cfg.MULTI_VIEWS_TRANSFORMS.NMB_CROPS[0])

        # collect the loss between global-global and global-local pairs
        g_loss, l_loss = 0.0, 0.0
        n_g_loss_terms, n_l_loss_terms = 0.0, 0.0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # skip cases where student and teacher operate on the same view
                    continue
                # KL-divergence between teacher and student
                loss = torch.sum(-q * nn.functional.log_softmax(student_out[v], dim=-1), dim=-1)
                # global-global pairs
                if iq < self.cfg.MULTI_VIEWS_TRANSFORMS.NMB_CROPS[0] and v < self.cfg.MULTI_VIEWS_TRANSFORMS.NMB_CROPS[0]:
                    g_loss += loss.mean()
                    n_g_loss_terms += 1
                # global-local pairs
                else:
                    l_loss += loss.mean()
                    n_l_loss_terms += 1
        # re-weight the loss
        alpha, beta = self.cfg.MULTI_VIEWS_TRANSFORMS.LAMBDAS
        if self.cfg.DINO.GLOBAL_ONLY:
            total_loss = g_loss/n_g_loss_terms
        else:
            total_loss = alpha * g_loss/n_g_loss_terms + beta * l_loss/n_l_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if self.cfg.DISTRIBUTED:
            dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class SWAVLoss(nn.Module):
    def __init__(self, cfg, gpu):
        super().__init__()
        self.crops_for_assign = cfg.MULTI_VIEWS_TRANSFORMS.CROPS_FOR_ASSIGN
        self.nmb_crops = cfg.MULTI_VIEWS_TRANSFORMS.NMB_CROPS
        self.sinkhorn_iterations = cfg.SWAV.SINKHORN_ITERATIONS
        self.epsilon = cfg.SWAV.EPSILON
        self.world_size = cfg.WORLD_SIZE
        self.temperature = cfg.SWAV.TEMPERATURE
        self.cfg = cfg

    def forward(self, output, bs, embedding, use_the_queue, queue, model):
        # type: (torch.Tensor, int, torch.Tensor, bool, torch.Tensor, torch.nn.Module) -> torch.Tensor

        g_loss, l_loss = 0.0, 0.0
        n_g_loss_terms, n_l_loss_terms = 0.0, 0.0

        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        if self.cfg.DISTRIBUTED:
                            extra_logits = torch.mm(
                                queue[i],
                                model.module.network.head.prototypes.weight.t()
                            ).detach()
                        else:
                            extra_logits = torch.mm(
                                queue[i],
                                model.network.head.prototypes.weight.t()
                            ).detach()
                        out = torch.cat((extra_logits, out))
                    # fill the queue
                    queue[i, bs:].copy_(queue[i, :-bs].clone())
                    queue[i, :bs].copy_(embedding[crop_id * bs: (crop_id + 1) * bs])

                # get assignments
                q = self.distributed_sinkhorn(out)[-bs:].detach()

            # cluster assignment prediction
            
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                # logits
                x = output[bs * v: bs * (v + 1)] / self.temperature
                # KL-divergence between the assignments q and the logits x
                loss = -torch.sum(q * nn.functional.log_softmax(x, dim=1), dim=1)
                # global-global pairs
                if crop_id < self.nmb_crops[0] and v < self.nmb_crops[0]:
                    g_loss += loss.mean()
                    n_g_loss_terms += 1
                # global-local pairs
                else:
                    l_loss += loss.mean()
                    n_l_loss_terms += 1
        
        # re-weight the loss
        alpha, beta = self.cfg.MULTI_VIEWS_TRANSFORMS.LAMBDAS
        total_loss = alpha * g_loss/n_g_loss_terms + beta * l_loss/n_l_loss_terms
        return total_loss, queue, use_the_queue

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * int(self.world_size) # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if self.cfg.DISTRIBUTED:
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if self.cfg.DISTRIBUTED:
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


class MoCoLoss(nn.Module):
    def __init__(self, cfg, gpu):
        super().__init__()
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss(reduction='none').cuda(gpu)
    
    def forward(self, logits, labels, boxes):
        if self.cfg.MOCO.GLOBAL_ONLY:
            loss = self.criterion(logits, labels)
            return loss.mean()
        else:
            num_local_crops = self.cfg.MULTI_VIEWS_TRANSFORMS.NMB_CROPS[1]

            logits = logits.chunk(num_local_crops + 1)
            labels = labels.chunk(num_local_crops + 1)

            g_loss, l_loss = 0.0, 0.0
            for i in range(1 + num_local_crops):
                loss = self.criterion(logits[i], labels[i])
                if i == 0:
                    g_loss += loss.mean()
                else:
                    l_loss += loss.mean()
            alpha, beta = self.cfg.MULTI_VIEWS_TRANSFORMS.LAMBDAS
            total_loss = alpha * g_loss + beta * l_loss/num_local_crops
            return total_loss
