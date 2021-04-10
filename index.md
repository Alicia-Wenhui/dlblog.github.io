# TransUNet: A Reproducibility Blog

This blog is for CS4240 Deep Learning reproducibility project. Created by:

**Zhen Wang** (student ID: 5220750, email: z.wang-42@student.tudelft.nl)

**Wenhui Wang** (student ID: 5279267, email: w.wang-28@student.tudelft.nl)

Zhen Wang did

Wenhui Wang did

## Introduction

Medical image segmentation is one of the hottest research task in the field of deep learning. Medical images have their unique characteristics: most medical images have vague borders and complicated lines, so it need high resolution informations to do precise   segmentation; the internal structures and organs of human body is relatively organized, so for object recognition we only need low resolution information. That is, a great medical image segmentation method should contain both high resolution information and low resolution information.

In the field of medical image segmentation, U-net[1] is one of the most used approaches. As the image shown below, U-net is a U-shape CNN network that the left part function as an encoder doing feature extraction and the right part function as a decoder doing upsampling. U-net also uses skip connection for each stage that connect the upsampling result with the output of the same resolution in the encoder, and use it as the next input of the decoder. One advantage of this structure is that it allows the output map with more low-level features and it allows a feature combination of different stages so that it's more suitable for deep supervision and contain more high resolution information. 

![unet](https://alicia-wenhui.github.io/dlblog.github.io/img/unet.png)

However, due to the inherent locality limitations of convolution operations,  it shows weakness in modeling long-range dependence. For structures that have long range dependences, the performance of U-net may be disappointing. However, this weakness can be solved by self-attention, in other words, the transformer, which is an architecture solely uses self-attention.

Previous researches already show that transformer shows great improvement in global modeling and have some satisfying results in image recognition tasks. However, researches also show that only use transformer in medical segmentation tasks can be disappointing. This is because transformer focus too much on long range modeling it lacks the ability to do local featuring thus lacks high resolution information.  

So, a combination of U-net and transformer should overcome their respective weakness and form a method that can extract both high and low resolution information. TransUNet is therefore proposed.




**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

