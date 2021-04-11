<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>


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

### TransUNet

In order to get the final result, we need to return the decoded feature presentations to the original spatial size to output the result image. If we use a transformer encoder and an easy upsampling, the feature space size should be transformed from $\frac{HW}{P^2}$ to $\frac{H}{P}\times \frac{W}{P}$. In this way using a simple $1\times1$ convolution then upsampling the feature map to $H\times W$ to have the original resolution.

However, because of the weakness o f transformer that discussed earlier, the generated $\frac{H}{P}\times \frac{W}{P}$ is a much more smaller resolution compared to the original $H \times W$ resolution, so the result image would be a low resolution image that lacks some specific details. In order to obtain more high resolution information, TransUNet adapting a combined CNN-Transformer encoder.

#### CNN-Transformer Hybrid Encoder

Before the input image goes into transformer, it firstly goes into CNN to retrieve the hidden feature and linear projection. This result of the CNN feature map then goes to transformer to do patch embedding. 

### Experiments

#### Evaluation Metrics

##### DSC

DSC means Dice similarity Coefficient, it is a commonly used evaluation metric for image segmentation tasks. DSC measures the similarity of two samples. It's formula is as follows:

$$Dice(P,T)=\frac{|P_1\and T_1|}{(|P_1|+|T_2|)/2}$$

##### HD

One big weakness for DSC is that it focus on internal padding and lacks sensitivity for boundary features. Hausdorff Distance(HD), it is also a widely used metric for segmentation tasks and it can be used as a supplement for DSC to portray boundary features. Hausdorff Distance measures the distance of subsets in a spatial space. The formula for HD is as follows:

$$d_H(X,Y)=max\{ \mathop{sup}_{x\in X}{inf}_{y\in Y}d(x,y),sup_{y\in Y}inf_{x\in X}d(x,y) \}$$

Where $sup()$ means supremum and $inf()$ means infimum.

In this task we use Hausdorff_95(95% HD), which is the 95th percentile of the distances between boundary points in X and Y. The reason for using it instead of directly using HD is that it can eliminate the impact of a very small subset of the outliers.

#### Abdomen Dataset

We firstly use the Abdomen dataset which was used in this paper. We use the original code to train the model directly and then use the trained model to test on test dataset. The result is shown in Table below. We give the author's result and our own result.

| TransUNet | DSC   | HD    | Aorta | Gallbladder | Kidney(L) | Kidney(R) | Liver | Pancreas | Spleen | Stomach |
| --------- | ----- | ----- | ----- | ----------- | --------- | --------- | ----- | -------- | ------ | ------- |
| Original  | 77.48 | 31.69 | 87.23 | 63.13       | 81.87     | 77.02     | 94.08 | 55.86    | 85.08  | 75.62   |
| Our's     | 77.21 | 31.27 | 87.01 | 64.75       | 80.18     | 76.40     | 93.86 | 56.36    | 84.83  | 74.32   |

We got quite similar result as the author's result in this paper.

#### Cervix Dataset

Then we used a new dataset which consists of planning CT scans of cervical cancer patients that were in varying stages of the disease but that were all eligible for radiotherapy. The CT scans consist of between 148 and 241 axial slices (depending on body size) of 512x512 voxels. The delineations were used in clinical practice and are provided for four structures that have all been renamed consistently: (1) bladder, (2) uterus, (3) rectum, (4) small bowel.

Since the author did not provide the code to pre-processing the dataset into required format, we write the code ourselves.

```python
import os
import numpy as np
import nibabel as nib
import imageio
import matplotlib
from matplotlib import pylab as plt
import h5py

def read_niifile(nii_file_path):
    img = nib.load(nii_file_path)
    img_fdata = img.get_fdata()
    return img_fdata


def save_fig(file_path):
    for i in range(1, 31):
        img_path = file_path + str(i) + '.nii'
        label_path = file_path + str(i) + '-mask.nii'  
    		img_data = read_niifile(img_path)
    		img_data = np.array(img_data)
    		img_data = np.clip(img_data, a_min=-125, a_max=275)
    		max = np.max(img_data)
    		min = np.min(img_data)
    		mean = np.mean(img_data)
    		std = np.std(img_data)

    		img_data = (img_data - min) / (max - min)
    		label_data = read_niifile(label_path)
    		label_data = np.array(label_data)

    		if i > 5:
        		(x, y, z) = img_data.shape
        		for k in range(z):

            		save_path = file_path + 'img/train/case' + '{0:04d}'.format(i) + '_slice' + '{0:03d}'.format(k) + '.npz'
            		print(save_path)

            		img = img_data[:, :, k]
            		img = np.array(img)
            		# img = np.clip(img, a_min=-125, a_max=275)

            		label = label_data[:, :, k]
            		label = np.array(label)

            		np.savez(save_path, image=img, label=label)

        		# imageio.imwrite(os.path.join(savepath, '{}.png'.format(k)), img)
    		else:
        		h5_file_path = file_path + 'img/test/case' + '{0:04d}'.format(i) + '.npy.h5'
        		with h5py.File(h5_file_path, 'w') as f:
            		img_data = img_data.transpose(2, 0, 1)
            		label_data = label_data.transpose(2, 0, 1)
            		f.create_dataset('image', data=img_data)
            		f.create_dataset('label', data=label_data)
                
def extract(file_path):
    save_fig(file_path)

if __name__ == '__main__':
    file_path = './np/train/'
    extract(file_path)
```

We use the same hyperparameters in original code and then save two models, one is trained 20 epochs and one is trained 100 epochs. The test results are shown below.

| TransUNet          | DSC   | HD    | bladder | uterus | rectum | small bowel |
| ------------------ | ----- | ----- | ------- | ------ | ------ | ----------- |
| Result(20 epochs)  | 59.16 | 17.56 | 73.84   | 56.06  | 62.36  | 44.38       |
| Result(100 epochs) | 66.99 | 15.61 | 85.88   | 66.63  | 74.00  | 41.45       |

Here is an example of the predicted mask.

![unet](https://alicia-wenhui.github.io/dlblog.github.io/img/predict.png)




