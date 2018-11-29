# Differentially Private Generative Adversarial Network
Liyang Xie, Kaixiang Lin, Shu Wang, Fei Wang, Jiayu Zhou

## Paper link
https://arxiv.org/abs/1802.06739

## Abstract
Generative Adversarial Network (GAN) and its variants have re-
cently attracted intensive research interests due to their elegant
theoretical foundation and excellent empirical performance as gen-
erative models. These tools provide a promising direction in the
studies where data availability is limited. One common issue in
GANs is that the density of the learned generative distribution
could concentrate on the training data points, meaning that they
can easily remember training samples due to the high model com-
plexity of deep networks. This becomes a major concern when
GANs are applied to private or sensitive data such as patient medi-
cal records, and the concentration of distribution may divulge criti-
cal patient information. To address this issue, in this paper we pro-
pose a differentially private GAN (DPGAN) model, in which we
achieve differential privacy in GANs by adding carefully designed
noise to gradients during the learning procedure. We provide rig-
orous proof for the privacy guarantee, as well as comprehensive
empirical evidence to support our analysis, where we demonstrate
that our method can generate high quality data points at a reason-
able privacy level.


