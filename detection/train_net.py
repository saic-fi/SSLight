import os
import torch, torchvision
from torch import nn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.layers import get_norm
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.modeling.backbone.resnet import BasicBlock, BottleneckBlock, ResNet

from backbones import *


@ROI_HEADS_REGISTRY.register()
class MyRes5ROIHeads(Res5ROIHeads):
    @classmethod
    def _build_res5_block(cls, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on
        depth  = cfg.MODEL.RESNETS.DEPTH 
        if depth == 18:
            blocks = ResNet.make_stage(
                BasicBlock,
                2,
                stride_per_block=[2, 1],
                in_channels=out_channels // 2,
                out_channels=out_channels,
                norm=norm,
            )
        elif depth == 34:
            blocks = ResNet.make_stage(
                BasicBlock,
                3,
                stride_per_block=[2, 1, 1],
                in_channels=out_channels // 2,
                out_channels=out_channels,
                norm=norm,
            )
        else:
            blocks = ResNet.make_stage(
                BottleneckBlock,
                3,
                stride_per_block=[2, 1, 1],
                in_channels=out_channels // 2,
                bottleneck_channels=bottleneck_channels,
                out_channels=out_channels,
                num_groups=num_groups,
                norm=norm,
                stride_in_1x1=stride_in_1x1,
            )
        return nn.Sequential(*blocks), out_channels


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(MyRes5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if "coco" in dataset_name:
            return COCOEvaluator(dataset_name, None, True, output_folder)
        else:
            assert "voc" in dataset_name
            return PascalVOCDetectionEvaluator(dataset_name)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_mobilenet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # model = Trainer.build_model(cfg)
    # print(model.backbone.output_shape())

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    # print(trainer.model.module.backbone.output_shape())

    # base_encoder = torchvision.models.__dict__['mobilenet_v3_large'](pretrained=True)
    # base_state_dict = base_encoder.state_dict()
    # model_state_dict = trainer.model.module.backbone.state_dict()
    # diff = torch.norm(model_state_dict['bottom_up.stem.0.1.weight'].cpu() - base_state_dict['features.0.1.weight'])
    # print('weight diff', diff)


    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )