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

The table below includes the scripts for specific pre-training experiments:


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">IN1K Linear Accu.</th>
<th valign="bottom">Training</th>  
<th valign="bottom">Pretrained ckpts (re-trained)</th>
<!-- TABLE BODY -->
<tr>
      <td align="left">DINO baseline</td>
      <td align="center">MobileNet V2</td>
      <td align="center">66.2</td>
      <td align="center"><a href=src/experiments/dino/mnv2/baseline.sh>script</a></td>
      <td align="center"><a href=https://drive.google.com/file/d/18zs2_D0uJicT01_qUlIwgtjlWNKblV27/view?usp=sharing>ckpt</a> / <a href=src/experiments/dino/mnv2/baseline.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO SSLight</td>
      <td align="center">MobileNet V2</td>
      <td align="center">68.3 (+2.1)</td>
      <td align="center"><a href=src/experiments/dino/mnv2/sslight.sh>script</a></td>
      <td align="center"><a href=https://drive.google.com/file/d/1M9qJgYIjUVlmPm3H34WKuvd0liKUw2O6/view?usp=sharing>ckpt</a> / <a href=src/experiments/dino/mnv2/sslight.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO baseline</td>
      <td align="center">ResNet18</td>
      <td align="center">62.2</td>
      <td align="center"><a href=src/experiments/dino/resnet18/baseline.sh>script</a></td>
      <td align="center"><a href=https://drive.google.com/file/d/1ZfuIcGjwFQWfk0RZHNbmdzeMVQFSAY92/view?usp=sharing>ckpt</a> / <a href=src/experiments/dino/resnet18/baseline.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO SSLight</td>
      <td align="center">ResNet18</td>
      <td align="center">65.7 (+3.5)</td>
      <td align="center"><a href=src/experiments/dino/resnet18/sslight.sh>script</a></td>
      <td align="center"><a href=https://drive.google.com/file/d/1_xBGya4AWtEnHDMKHF5rL2sl0fJaMzki/view?usp=sharing>ckpt</a> / <a href=src/experiments/dino/resnet18/sslight.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO baseline</td>
      <td align="center">ResNet34</td>
      <td align="center">67.7</td>
      <td align="center"><a href=src/experiments/dino/resnet34/baseline.sh>script</a></td>
      <td align="center"><a href=https://drive.google.com/file/d/1BcOgYKUzrvzrUfMaZQ8hZgqu--b5dwhS/view?usp=sharing>ckpt</a> / <a href=src/experiments/dino/resnet34/baseline.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO SSLight</td>
      <td align="center">ResNet34</td>
      <td align="center">69.7 (+2.0)</td>
      <td align="center"><a href=src/experiments/dino/resnet34/sslight.sh>script</a></td>
      <td align="center"><a href=https://drive.google.com/file/d/1SZTkqscN7QAaPf_WvYPWiIvjpAeawaXL/view?usp=sharing>ckpt</a> / <a href=src/experiments/dino/resnet34/sslight.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO baseline</td>
      <td align="center">ViT-Tiny/16</td>
      <td align="center">66.7</td>
      <td align="center"><a href=src/experiments/dino/vit_tiny_16/baseline.sh>script</a></td>
      <td align="center"><a href=https://drive.google.com/file/d/1ZJqhW5J3_aKcdvpaPNUNwQEGwZ67G7lE/view?usp=sharing>ckpt</a> / <a href=src/experiments/dino/vit_tiny_16/baseline.log>log</a></td>
</tr>
<tr>
      <td align="left">DINO SSLight</td>
      <td align="center">ViT-Tiny/16</td>
      <td align="center">69.5 (+2.8)</td>
      <td align="center"><a href=src/experiments/dino/vit_tiny_16/sslight.sh>script</a></td>
      <td align="center"><a href=https://drive.google.com/file/d/15iweaPCulIbc1vCBBzUjw080ROucncCH/view?usp=sharing>ckpt</a> / <a href=src/experiments/dino/vit_tiny_16/sslight.log>log</a></td>
</tr>
<tr>
      <td align="left">SWAV baseline</td>
      <td align="center">MobileNet V2</td>
      <td align="center">65.2</td>
      <td align="center"><a href=src/experiments/swav/mnv2/baseline.sh>script</a></td>
      <td align="center"><a href=https://drive.google.com/file/d/1KxSCV0IdIBOnGXOErRN2EmT2wqpD2LEu/view?usp=sharing>ckpt</a> / <a href=src/experiments/swav/mnv2/baseline.log>log</a></td>
</tr>
<tr>
      <td align="left">SWAV SSLight</td>
      <td align="center">MobileNet V2</td>
      <td align="center">67.3 (+2.1)</td>
      <td align="center"><a href=src/experiments/swav/mnv2/sslight.sh>script</a></td>
      <td align="center"><a href=https://drive.google.com/file/d/104XnVgu0o2U7vsV1GajIDnG8r3K4L8XY/view?usp=sharing>ckpt</a> / <a href=src/experiments/swav/mnv2/sslight.log>log</a></td>
</tr>
<tr>
      <td align="left">MoCo baseline</td>
      <td align="center">MobileNet V2</td>
      <td align="center">60.6 </td>
      <td align="center"><a href=src/experiments/moco/mnv2/baseline.sh>script</a></td>
      <td align="center"><a href=https://drive.google.com/file/d/1QoIBw3gDBqSMr2aMQQfWmEYDlT78Qki9/view?usp=sharing>ckpt</a> / <a href=src/experiments/moco/mnv2/baseline.log>log</a></td>
</tr>
<tr>
      <td align="left">MoCo SSLight</td>
      <td align="center">MobileNet V2</td>
      <td align="center">61.6 (+1.0)</td>
      <td align="center"><a href=src/experiments/moco/mnv2/sslight.sh>script</a></td>
      <td align="center"><a href=https://drive.google.com/file/d/1_5R3rGIEJV8ogdHlWzdsDyHpDQw9E5Bf/view?usp=sharing>ckpt</a> / <a href=src/experiments/moco/mnv2/sslight.log>log</a></td>
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
