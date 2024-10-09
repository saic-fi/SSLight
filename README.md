# [Effective Self-supervised Pre-training on Low-compute networks without Distillation](https://arxiv.org/abs/2210.02808)
Fuwen Tan, Fatemeh Saleh, Brais Martinez, ICLR 2023.

## Abstract
Despite the impressive progress of self-supervised learning (SSL), its applicability to low-compute networks has received limited attention. Reported performance has trailed behind standard supervised pre-training by a large margin, barring self-supervised learning from making an impact on models that are deployed on device. Most prior works attribute this poor performance to the capacity bottleneck of the low-compute networks and opt to bypass the problem through the use of knowledge distillation (KD). In this work, we revisit SSL for efficient neural networks, taking a closer look at what are the detrimental factors causing the practical limitations, and whether they are intrinsic to the self-supervised low-compute setting. We find that, contrary to accepted knowledge, there is no intrinsic architectural bottleneck, we diagnose that the performance bottleneck is related to the model complexity vs regularization strength trade-off. In particular, we start by empirically observing that the use of local views can have a dramatic impact on the effectiveness of the SSL methods. This hints at view sampling being one of the performance bottlenecks for SSL on low-capacity networks. We hypothesize that the view sampling strategy for large neural networks, which requires matching views in very diverse spatial scales and contexts, is too demanding for low-capacity architectures. We systematize the design of the view sampling mechanism, leading to a new training methodology that consistently improves the performance across different SSL methods (e.g. MoCo-v2, SwAV or DINO), different low-size networks (convolution-based networks, e.g. MobileNetV2, ResNet18, ResNet34 and vision transformer, e.g. ViT-Ti), and different tasks (linear probe, object detection, instance segmentation and semi-supervised learning). Our best models establish new state-of-the-art for SSL methods on low-compute networks despite not using a KD loss term.

## Software required
The code is only tested on Linux 64:

```
  cd $(SSLIGHT_ROOT)/src
  conda env create -f environment.yml
  conda activate ssl
```

## Experiments

This repo supports pre-training [DINO|SwAV|MoCo] with [MobileNet V2|ResNets|ViTs] as the baseline. To run the training:

```
  cd $(SSLIGHT_ROOT)/src
  python3 main.py --cfg config/exp_yamls/dino/dino_cnn_sslight.yaml DATA.PATH_TO_DATA_DIR $IN1K_PATH OUTPUT_DIR $OUTPUT_PATH
```

To assess the quality of features during pre-training, an additional linear classifier can be trained on the separated features. This ensures that the gradient from the linear classifier does not interfere with the feature learning process:

```
  python3 main.py --cfg config/exp_yamls/dino/dino_cnn_sslight.yaml DATA.PATH_TO_DATA_DIR $IN1K_PATH OUTPUT_DIR $OUTPUT_PATH TRAIN.JOINT_LINEAR_PROBE True
```

Note that the accuracy of this extra classifier is typically lower than a standard linear probing evaluation.

### Pretraining
The table below includes the scripts for the pre-training experiments:


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pre-training</th>  
<th valign="bottom">IN1K <br/>Linear eval</th>
<th valign="bottom">Pretrained ckpts <br/>(re-trained)</th>
<!-- TABLE BODY -->
<tr>
      <td align="left">DINO baseline</td>
      <td align="center">ViT-Tiny/16</td>
      <td align="center"><a href=src/experiments/dino/vit_tiny_16/baseline.sh>script</a></td>
      <td align="left">Accu.: 66.7<br/> <a href=classification/experiments/dino/vit_tiny_16/linear.sh>script</a></td>
      <td align="center"><a href=https://huggingface.co/fwtan/sslight_checkpoints/blob/main/dino_vitt16_baseline.pth.tar>ckpt</a> / <a href=src/experiments/dino/vit_tiny_16/baseline.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO SSLight</td>
      <td align="center">ViT-Tiny/16</td>
      <td align="center"><a href=src/experiments/dino/vit_tiny_16/sslight.sh>script</a></td>
      <td align="left">Accu.: 69.5 (+2.8)<br/> <a href=classification/experiments/dino/vit_tiny_16/linear.sh>script</a></td>
      <td align="center"><a href=https://huggingface.co/fwtan/sslight_checkpoints/blob/main/dino_vitt16_sslight.pth.tar>ckpt</a> / <a href=src/experiments/dino/vit_tiny_16/sslight.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO baseline</td>
      <td align="center">ResNet18</td>
      <td align="center"><a href=src/experiments/dino/resnet18/baseline.sh>script</a></td>
      <td align="left">Accu.: 62.2<br/> <a href=classification/experiments/dino/resnet18/linear.sh>script</a></td>
      <td align="center"><a href=https://huggingface.co/fwtan/sslight_checkpoints/blob/main/dino_r18_baseline.pth.tar>ckpt</a> / <a href=src/experiments/dino/resnet18/baseline.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO SSLight</td>
      <td align="center">ResNet18</td>
      <td align="center"><a href=src/experiments/dino/resnet18/sslight.sh>script</a></td>
      <td align="left">Accu.: 65.7 (+3.5)<br/> <a href=classification/experiments/dino/resnet18/linear.sh>script</a></td>
      <td align="center"><a href=https://huggingface.co/fwtan/sslight_checkpoints/blob/main/dino_r18_sslight.pth.tar>ckpt</a> / <a href=src/experiments/dino/resnet18/sslight.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO baseline</td>
      <td align="center">ResNet34</td>
      <td align="center"><a href=src/experiments/dino/resnet34/baseline.sh>script</a></td>
      <td align="left">Accu.: 67.7<br/> <a href=classification/experiments/dino/resnet34/linear.sh>script</a></td>
      <td align="center"><a href=https://huggingface.co/fwtan/sslight_checkpoints/blob/main/dino_r34_baseline.pth.tar>ckpt</a> / <a href=src/experiments/dino/resnet34/baseline.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO SSLight</td>
      <td align="center">ResNet34</td>
      <td align="center"><a href=src/experiments/dino/resnet34/sslight.sh>script</a></td>
      <td align="left">Accu.: 69.7 (+2.0)<br/> <a href=classification/experiments/dino/resnet34/linear.sh>script</a></td>
      <td align="center"><a href=https://huggingface.co/fwtan/sslight_checkpoints/blob/main/dino_r34_sslight.pth.tar>ckpt</a> / <a href=src/experiments/dino/resnet34/sslight.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO baseline</td>
      <td align="center">MobileNet V2</td>
      <td align="center"><a href=src/experiments/dino/mnv2/baseline.sh>script</a></td>
      <td align="left">Accu.: 66.2<br/> <a href=classification/experiments/dino/mnv2/linear.sh>script</a></td>
      <td align="center"><a href=https://huggingface.co/fwtan/sslight_checkpoints/blob/main/dino_mnv2_baseline.pth.tar>ckpt</a> / <a href=src/experiments/dino/mnv2/baseline.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO SSLight</td>
      <td align="center">MobileNet V2</td>
      <td align="center"><a href=src/experiments/dino/mnv2/sslight.sh>script</a></td>
      <td align="left">Accu.: 68.3 (+2.1)<br/> <a href=classification/experiments/dino/mnv2/linear.sh>script</a></td>
      <td align="center"><a href=https://huggingface.co/fwtan/sslight_checkpoints/blob/main/dino_mnv2_sslight.pth.tar>ckpt</a> / <a href=src/experiments/dino/mnv2/sslight.log>log</a></td>
</tr>
<tr>
      <td align="left">SWAV baseline</td>
      <td align="center">MobileNet V2</td>
      <td align="center"><a href=src/experiments/swav/mnv2/baseline.sh>script</a></td>
      <td align="left">Accu.: 65.2<br/> <a href=classification/experiments/swav/mnv2/linear.sh>script</a></td>
      <td align="center"><a href=https://huggingface.co/fwtan/sslight_checkpoints/blob/main/swav_mnv2_baseline.pth.tar>ckpt</a> / <a href=src/experiments/swav/mnv2/baseline.log>log</a></td>
</tr>
<tr>
      <td align="left">SWAV SSLight</td>
      <td align="center">MobileNet V2</td>
      <td align="center"><a href=src/experiments/swav/mnv2/sslight.sh>script</a></td>
      <td align="left">Accu.: 67.3 (+2.1)<br/> <a href=classification/experiments/swav/mnv2/linear.sh>script</a></td>
      <td align="center"><a href=https://huggingface.co/fwtan/sslight_checkpoints/blob/main/swav_mnv2_sslight.pth.tar>ckpt</a> / <a href=src/experiments/swav/mnv2/sslight.log>log</a></td>
</tr>
<tr>
      <td align="left">MoCo baseline</td>
      <td align="center">MobileNet V2</td>
      <td align="center"><a href=src/experiments/moco/mnv2/baseline.sh>script</a></td>
      <td align="left">Accu.: 60.6<br/> <a href=classification/experiments/moco/mnv2/linear.sh>script</a></td>
      <td align="center"><a href=https://huggingface.co/fwtan/sslight_checkpoints/blob/main/moco_mnv2_baseline.pth.tar>ckpt</a> / <a href=src/experiments/moco/mnv2/baseline.log>log</a></td>
</tr>
<tr>
      <td align="left">MoCo SSLight</td>
      <td align="center">MobileNet V2</td>
      <td align="center"><a href=src/experiments/moco/mnv2/sslight.sh>script</a></td>
      <td align="left">Accu.: 61.6 (+1.0)<br/> <a href=classification/experiments/moco/mnv2/linear.sh>script</a></td>
      <td align="center"><a href=https://huggingface.co/fwtan/sslight_checkpoints/blob/main/moco_mnv2_sslight.pth.tar>ckpt</a> / <a href=src/experiments/moco/mnv2/sslight.log>log</a></td>
</tr>
</tbody></table> 

### Downstream evaluations

The table below includes the scripts for semi-supervised, object detection and instance segmentation evaluations


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">IN1K <br/>Semi-sup 1%</th>
<th valign="bottom">IN1K <br/>Semi-sup 10%</th>
<th valign="bottom">CoCo <br/>Object Detection</th>
<th valign="bottom">CoCo <br/>Instance Segmentation</th>
<!-- TABLE BODY -->
<tr>
      <td align="left">DINO baseline</td>
      <td align="center">ResNet18</td>
      <td align="left">Accu.: 44.5 <br/> <a href=classification/experiments/dino/resnet18/semi_1per.sh>script</a></td>
      <td align="left">Accu.: 59.2 <br/> <a href=classification/experiments/dino/resnet18/semi_10per.sh>script</a></td>
      <td align="left">AP: 32.7 <br/> <a href=detection/experiments/dino/resnet18/det.sh>script</a></td>
      <td align="left">AP: 30.6 <br/> <a href=detection/experiments/dino/resnet18/seg.sh>script</a></td>
</tr>
<tr>
      <td align="left">DINO SSLight</td>
      <td align="center">ResNet18</td>
      <td align="left">Accu.: 49.8 (+5.3) <br/> <a href=classification/experiments/dino/resnet18/semi_1per.sh>script</a></td>
      <td align="left">Accu.: 63.0 (+3.8) <br/> <a href=classification/experiments/dino/resnet18/semi_10per.sh>script</a></td>
      <td align="left">AP: 34.1 (+1.4) <br/> <a href=detection/experiments/dino/resnet18/det.sh>script</a></td>
      <td align="left">AP: 31.8 (+1.2) <br/> <a href=detection/experiments/dino/resnet18/seg.sh>script</a></td>
</tr>
<tr>
      <td align="left">DINO baseline</td>
      <td align="center">ResNet34</td>
      <td align="left">Accu.: 52.4 <br/> <a href=classification/experiments/dino/resnet34/semi_1per.sh>script</a></td>
      <td align="left">Accu.: 65.4 <br/> <a href=classification/experiments/dino/resnet34/semi_10per.sh>script</a></td>
      <td align="left">AP: 37.6 <br/> <a href=detection/experiments/dino/resnet34/det.sh>script</a></td>
      <td align="left">AP: 34.6 <br/> <a href=detection/experiments/dino/resnet34/seg.sh>script</a></td>
</tr>
<tr>
      <td align="left">DINO SSLight</td>
      <td align="center">ResNet34</td>
      <td align="left">Accu.: 55.2 (+2.8) <br/> <a href=classification/experiments/dino/resnet34/semi_1per.sh>script</a></td>
      <td align="left">Accu.: 67.2 (+1.8) <br/> <a href=classification/experiments/dino/resnet34/semi_10per.sh>script</a></td>
      <td align="left">AP: 38.6 (+1.0): <br/> <a href=detection/experiments/dino/resnet34/det.sh>script</a></td>
      <td align="left">AP: 35.5 (+0.9): <br/> <a href=detection/experiments/dino/resnet34/seg.sh>script</a></td>
</tr>
<tr>
      <td align="left">DINO baseline</td>
      <td align="center">MobileNet V2</td>
      <td align="left">Accu.: 47.9 <br/> <a href=classification/experiments/dino/mnv2/semi_1per.sh>script</a></td>
      <td align="left">Accu.: 61.3 <br/> <a href=classification/experiments/dino/mnv2/semi_10per.sh>script</a></td>
      <td align="left">AP: 30.9 <br/> <a href=detection/experiments/dino/mnv2/det.sh>script</a></td>
      <td align="left">AP: 28.1 <br/> <a href=detection/experiments/dino/mnv2/seg.sh>script</a></td>
</tr>
<tr>
      <td align="left">DINO SSLight</td>
      <td align="center">MobileNet V2</td>
      <td align="left">Accu.: 50.6 (+2.7) <br/> <a href=classification/experiments/dino/mnv2/semi_1per.sh>script</a></td>
      <td align="left">Accu.: 63.5 (+2.2) <br/> <a href=classification/experiments/dino/mnv2/semi_10per.sh>script</a></td>
      <td align="left">AP: 32.1 (+1.2) <br/> <a href=detection/experiments/dino/mnv2/det.sh>script</a></td>
      <td align="left">AP: 29.1 (+1.0) <br/> <a href=detection/experiments/dino/mnv2/seg.sh>script</a></td>
</tr>
</tbody></table> 


## Citing

If you find our paper/code useful, please consider citing:

    @inproceedings{sslight2023,
        author = {Fuwen Tan and Fatemeh Saleh and Brais Martinez},
        title = {Effective Self-supervised Pre-training on Low-compute Networks without Distillation},
        booktitle = {International Conference on Learning Representations (ICLR)},
        year = {2023},
     }
