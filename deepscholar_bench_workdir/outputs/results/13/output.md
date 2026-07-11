# Related Work

## Long-Tailed Recognition

Real-world visual data commonly follow a long-tailed class distribution, which
biases standard classifiers toward head classes and starves the tail. A large
body of work re-balances the loss or the sample selection. Class-Balanced Loss
re-weights each class by the effective number of samples [1](https://arxiv.org/abs/1901.05555),
while LDAM introduces a label-distribution-aware margin that enlarges the
margin of rare classes [2](https://arxiv.org/abs/1906.07413). Logit adjustment
theoretically calibrates class scores according to prior label frequencies at
either training or inference time [3](https://arxiv.org/abs/2007.07314), and
Balanced Meta-Softmax learns a meta-sampler jointly with a balanced softmax to
match the test distribution [4](https://arxiv.org/abs/2007.10740). Distribution
Alignment provides a unified two-stage framework that adjusts classifier
outputs to a target distribution [5](https://arxiv.org/abs/2103.16370).

A second family decouples representation learning from classifier learning.
Kang et al. showed that features learned with instance-balanced sampling are
already strong, and that only the classifier needs re-balancing
[6](https://arxiv.org/abs/1910.09217). BBN follows a similar spirit with a
bilateral branch and a cumulative learning strategy that gradually shifts from
conventional to re-balanced training [7](https://arxiv.org/abs/1912.02413).
MiSLAS further shows that mixup-based training combined with label-aware
smoothing improves calibration under class imbalance
[8](https://arxiv.org/abs/2104.00466), while ResLT reformulates long-tailed
classification as a residual learning problem across head, medium and tail
groups [9](https://arxiv.org/abs/2101.10633). Feature-space augmentation
transfers intra-class variance from head to tail classes to enrich rare-class
representations [10](https://arxiv.org/abs/2008.03673). A recent survey
summarises this rapidly growing area [11](https://arxiv.org/abs/2110.04596).

## Contrastive Representation Learning

Contrastive objectives have become a standard tool for learning transferable
visual features. SimCLR shows that strong data augmentation combined with an
InfoNCE loss over two views is sufficient to reach competitive linear-probe
accuracy [12](https://arxiv.org/abs/2002.05709), and MoCo maintains a momentum
queue of negatives to scale contrastive learning without huge batches
[13](https://arxiv.org/abs/1911.05722). BYOL demonstrates that latent
bootstrapping without explicit negatives can also produce strong
representations [14](https://arxiv.org/abs/2006.07733), and hard-negative
mining further sharpens the geometry of the embedding space
[15](https://arxiv.org/abs/2010.04592). Supervised Contrastive Learning (SCL)
extends InfoNCE to the supervised setting by treating all samples sharing a
label as positives, improving classification accuracy over cross-entropy
baselines [16](https://arxiv.org/abs/2004.11362). Multi-view training is a
central design choice in these frameworks; our theoretical analysis revisits
this choice and shows that, unlike self-supervised contrastive learning, SCL
does not always benefit from more views because gradient conflicts arise
between attraction and repulsion terms across positive/negative pairs.

## Contrastive Learning for Long-Tailed Recognition

Because contrastive objectives shape the feature geometry directly, several
methods use them to fight class imbalance. Wang et al. propose a hybrid network
that couples an SCL branch with a classifier branch, showing that contrastive
features are more balanced than cross-entropy ones
[17](https://arxiv.org/abs/2103.14267). Parametric Contrastive Learning (PaCo)
adds learnable class centers as parametric positives, which acts as an implicit
re-balancing of the supervised contrastive gradient
[18](https://arxiv.org/abs/2107.12028). Targeted Supervised Contrastive
Learning (TSC) predefines uniformly distributed class targets on the
hypersphere to prevent tail-class features from collapsing
[19](https://arxiv.org/abs/2111.13998). Balanced Contrastive Learning (BCL)
introduces class-averaging and class-complement terms so that every class
appears in every mini-batch, restoring symmetry between head and tail classes
in the SCL gradient [20](https://arxiv.org/abs/2207.09052). Complementary
studies argue that self-supervised pre-training is intrinsically more robust
to dataset imbalance than supervised training
[21](https://arxiv.org/abs/2110.05025), and that even unlabeled data can
substantially improve tail-class performance when combined with semi-supervised
or self-supervised objectives [22](https://arxiv.org/abs/2006.07529).

Our Aligned Contrastive Learning (ACL) is in the same spirit as
[16](https://arxiv.org/abs/2004.11362), [18](https://arxiv.org/abs/2107.12028),
[19](https://arxiv.org/abs/2111.13998) and [20](https://arxiv.org/abs/2207.09052),
but differs in that it is derived from an explicit gradient analysis of SCL
under multi-view training. By eliminating gradient conflicts and equalising the
attraction and repulsion magnitudes between positive and negative pairs, ACL
turns additional views into a consistent source of improvement and achieves
state-of-the-art results on long-tailed CIFAR, ImageNet-LT, Places-LT and
iNaturalist.
