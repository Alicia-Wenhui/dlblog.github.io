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
