# TransUNet: A Reproducibility Blog

This blog is for CS4240 Deep Learning reproducibility project. Created by:

**Zhen Wang** (student ID: 5220750, email: z.wang-42@student.tudelft.nl)

**Wenhui Wang** (student ID: 5279267, email: w.wang-28@student.tudelft.nl)

Zhen Wang did

Wenhui Wang did

## Introduction

Medical image segmentation is one of the hottest research task in the field of deep learning. Medical images have their unique characteristics: most medical images have vague borders and complicated lines, so it need high resolution informations to do precise   segmentation; the internal structures and organs of human body is relatively organized, so for object recognition we only need low resolution information. That is, a great medical image segmentation method should contain both high resolution information and low resolution information.

In the field of medical image segmentation, U-net is one of the most used approaches. As the image shown below, U-net is a U-shape CNN network that the left part function as an encoder doing feature extraction and the right part function as a decoder doing upsampling. U-net also uses skip connection for each stage that connect the upsampling result with the output of the same resolution in the encoder, and use it as the next input of the decoder. One advantage of this structure is that it allows the output map with more low-level features and it allows a feature combination of different stages so that it's more suitable for deep supervision and contain more high resolution information. 

![unet](https://alicia-wenhui.github.io/dlblog.github.io/img/unet.png)

However, due to the inherent locality limitations of convolution operations,  it shows weakness in modeling long-range dependence. For structures that have long range dependences, the performance of U-net may be disappointing. However, this weakness can be solved by self-attention, in other words, the transformer, which is an architecture solely uses self-attention.

Previous researches already show that transformer shows great improvement in global modeling and have some satisfying results in image recognition tasks. However, researches also show that only use transformer in medical segmentation tasks can be disappointing. This is because transformer focus too much on long range modeling it lacks the ability to do local featuring thus lacks high resolution information.  

So, a combination of U-net and transformer should overcome their respective weakness and form a method that can extract both high and low resolution information. TransUNet is therefore proposed. 

## Method

TransUNet is a hybrid CNN-Transformer architecture that takes advantage of both CNN's high resolution feature space and transformer's long range feature information. The overall structure of the TransUNet is shown below.

![unet](https://alicia-wenhui.github.io/dlblog.github.io/img/tranunet.jpg)

### Transformer as Encoder

#### Image Sequentialization

The progress of image sequentialization is shown below. First of all, reshape the input image into a series of 2D patches, linearly project each patch and add position embeddings. Provide the result sequence to transformer encoder, adding a multilayer perceptron head and get the final classification result.

![unet](https://alicia-wenhui.github.io/dlblog.github.io/img/tranencoder.png)

#### Patch Embedding

In order to get positional information, we need to do a position embedding and combine it with patches. The formula is as follows:

$$ z_0 = [x_p^1E;x_p^2E;...;x_p^NE] + E_{pos} $$

Where $x_p$ is patch, $E_{pos}$ is position embedding and $E$ is embedding projection.

The L-th layer output of the transformer encoder can be introduced as:

$$z\dot{}_l = MSA(LN(z_{l-1}))+z_{l-1}$$

$$z_l = MLP(LN(z\dot{}_{l}))+z\dot{}_{l}$$

Where MSA means Multi-head Attention, MLP means Multilayer Perceptron, LN() means the layer normalization operation, $z_l$ is the encoded image presentation.




