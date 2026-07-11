# Related Work

Our work sits at the intersection of active learning, federated learning under
statistical heterogeneity, and their emerging combination — federated active
learning (FAL). We organize prior work into four themes and discuss how CHASe
differs from each.

**Active learning (AL) with deep networks.** Classical AL seeks to reduce
annotation cost by selecting the most informative unlabeled samples for expert
labeling. Uncertainty-based strategies estimate posterior uncertainty either by
maintaining explicit weight distributions [9](https://arxiv.org/abs/1505.05424)
or by using stochastic forward passes to approximate Bayesian inference at test
time [8](https://arxiv.org/abs/1506.02142). Representation-based approaches
instead cover the input distribution: the Core-Set formulation of Sener and
Savarese casts pool-based selection as a geometric covering problem over the
learned feature space [3](https://arxiv.org/abs/1708.00489). Task-agnostic
methods such as Learning Loss attach an auxiliary module that predicts the
target loss of unlabeled inputs and prioritizes those with the largest expected
loss [4](https://arxiv.org/abs/1905.03677), while multiple-instance active
learning transfers uncertainty-based selection to structured tasks such as
object detection [11](https://arxiv.org/abs/2104.02324). More recent work
exploits temporal signals within training itself, showing that discrepancies of
model outputs between successive training steps are a strong indicator of
informativeness [10](https://arxiv.org/abs/2107.14153). CHASe is inspired by
these temporal-consistency ideas but generalizes them to the federated setting,
where the "temporal" signal must be extracted despite periodic global
aggregation that shifts the decision boundary.

**Federated learning and Non-IID heterogeneity.** Federated learning was
introduced by McMahan et al. through the FedAvg algorithm, which learns a
shared model by averaging locally computed updates and highlighted robustness
to unbalanced, non-IID client data as a defining challenge
[1](https://arxiv.org/abs/1602.05629). A long line of work has tackled the
statistical and systems heterogeneity that FedAvg exposes: FedProx introduces
a proximal regularizer that stabilizes local updates under heterogeneous data
and computation [2](https://arxiv.org/abs/1812.06127), and Bayesian
nonparametric aggregation matches neurons across clients before averaging to
handle model-level heterogeneity [14](https://arxiv.org/abs/1905.12022).
Realistic benchmarking of non-IID federations is standardized by LEAF, which
we build upon in our text experiments [12](https://arxiv.org/abs/1812.01097).
A second line of work directly attacks the catastrophic-forgetting effect that
non-IID local training induces: FedReg regularizes locally trained parameters
against pseudo data that encode previously acquired global knowledge
[16](https://arxiv.org/abs/2203.02645), while not-true distillation preserves
global class knowledge that is otherwise erased on each client
[15](https://arxiv.org/abs/2106.03097). Model-Contrastive Federated Learning
(MOON) instead aligns representations between the current local model and the
last global model through a contrastive term
[5](https://arxiv.org/abs/2103.16257). CHASe’s alignment loss is inspired by
these calibration objectives, but it is the first to be conditioned on
per-sample epistemic variation, using the local model to align "easy" samples
and the global model to align "hard" ones.

**Federated active learning.** A small but growing body of work combines FL
and AL. Goetz et al. proposed Active Federated Learning, which biases the
selection of participating clients toward those whose data yield larger
expected loss reductions [13](https://arxiv.org/abs/1909.12641). More recent
work targets sample-level selection within each client: F-AL adapts standard
acquisition functions to FL and demonstrates efficient annotation across
several benchmarks [7](https://arxiv.org/abs/2202.00195), while KAFAL
identifies that ante- and post-aggregation models disagree strongly under
non-IID data and proposes a knowledge-aware discrepancy score to combine both
views [6](https://arxiv.org/abs/2211.13579). These methods still rely on a
single, instantaneous decision boundary — either the local model, the global
model, or a static combination of both. Our work shows empirically that under
strong non-IID and heterogeneous client behavior these single-snapshot
strategies can be inferior even to random selection, motivating the epistemic
variation signal that CHASe tracks across training epochs.

**Efficiency of active data selection.** Because AL scoring must in principle
sweep the entire unlabeled pool at every round, efficiency is a recurring
concern. Core-Set relies on greedy k-center approximations that scale to large
pools [3](https://arxiv.org/abs/1708.00489), and Learning Loss avoids
Monte-Carlo inference by predicting loss directly
[4](https://arxiv.org/abs/1905.03677). MC-Dropout-based scoring, in contrast,
requires many stochastic forward passes and is expensive on edge devices
[8](https://arxiv.org/abs/1506.02142). None of these prior methods exploit the
FL setting where clients participate in many communication rounds and can
therefore accumulate historical statistics. CHASe’s freeze-and-awaken
mechanism, combined with subset sampling, is designed for precisely this
regime: samples with persistently zero epistemic variation are excluded from
subsequent inference passes and only re-scored when boundary shifts warrant it.

In summary, CHASe complements uncertainty and representation-based AL
[3](https://arxiv.org/abs/1708.00489), [4](https://arxiv.org/abs/1905.03677),
[8](https://arxiv.org/abs/1506.02142), [9](https://arxiv.org/abs/1505.05424),
[10](https://arxiv.org/abs/2107.14153), [11](https://arxiv.org/abs/2104.02324)
with a signal that is intrinsic to federated training, and complements
non-IID-aware FL techniques
[1](https://arxiv.org/abs/1602.05629), [2](https://arxiv.org/abs/1812.06127),
[5](https://arxiv.org/abs/2103.16257), [14](https://arxiv.org/abs/1905.12022),
[15](https://arxiv.org/abs/2106.03097), [16](https://arxiv.org/abs/2203.02645)
with a data-selection perspective on decision-boundary calibration. Compared
with existing FAL approaches
[6](https://arxiv.org/abs/2211.13579), [7](https://arxiv.org/abs/2202.00195),
[13](https://arxiv.org/abs/1909.12641), CHASe is, to our knowledge, the first
to explicitly model per-sample epistemic variation caused by client
heterogeneity and to jointly exploit it for selection, model calibration, and
efficiency, using LEAF-style non-IID benchmarks
[12](https://arxiv.org/abs/1812.01097) among others for evaluation.
