import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import random
import numpy as np
from MICCAI24.data.Cropping import get_bbox_from_mask, bounding_box_to_slice
import math


def Rescale01_clip995(image):
    image = np.clip(image.copy(), a_min=1e-8, a_max=np.percentile(image, 99.5))
    image = image - image.min()
    image = image / np.clip(image.max(), a_min=1e-8, a_max=None)
    return image


def bbox_update(old, new, img_shape):
    mid_x, mid_y, mid_z = [(i[1]+i[0])//2 for i in old]
    x, y, z = [j/2 for j in new]
    new_bbox = [[mid_x-x, mid_x+x], [mid_y-y, mid_y+y], [mid_z-z, mid_z+z]]
    pad_bbox = []
    for bbox_index in range(3):
        if new_bbox[bbox_index][0] < 0:
            pad_1 = abs(new_bbox[bbox_index][0])
            new_bbox[bbox_index][0] = 0
        else:
            pad_1 = 0

        if new_bbox[bbox_index][1] > img_shape[bbox_index]:
            pad_2 = new_bbox[bbox_index][1] - img_shape[bbox_index]
            new_bbox[bbox_index][1] = img_shape[bbox_index]
        else:
            pad_2 = 0
        pad_bbox.append([int((pad_1 + pad_2) // 2), int((pad_1 + pad_2) - (pad_1 + pad_2) // 2)].copy())
        new_bbox[bbox_index][0] = int(new_bbox[bbox_index][0])
        new_bbox[bbox_index][1] = int(new_bbox[bbox_index][1])
    return new_bbox, pad_bbox


def cropApad_noscale(x, y, x_seg, y_seg, mask_x, mask_y, img_size):
    # x_bbox = bounding_box_to_slice(get_bbox_from_mask(mask_x))
    x_bbox = get_bbox_from_mask(mask_x)
    x_bbox_size = [(l[1] - l[0]) for l in x_bbox]
    # y_bbox = bounding_box_to_slice(get_bbox_from_mask(mask_y))
    y_bbox = get_bbox_from_mask(mask_y)
    y_bbox_size = [(l[1] - l[0]) for l in y_bbox]
    max_bbox_size = [max(x_bbox_size[i], y_bbox_size[i]) for i in range(3)]
    new_bbox = [math.ceil(max_bbox_size[i] / 32) * 32 for i in range(3)]

    x_bbox, x_pad = bbox_update(x_bbox, new_bbox, x.shape)
    y_bbox, y_pad = bbox_update(y_bbox, new_bbox, y.shape)
    x = np.pad(x[bounding_box_to_slice(x_bbox)], x_pad, mode='constant', constant_values=0)
    x_seg = np.pad(x_seg[bounding_box_to_slice(x_bbox)], x_pad, mode='constant', constant_values=0)
    y = np.pad(y[bounding_box_to_slice(y_bbox)], y_pad, mode='constant', constant_values=0)
    y_seg = np.pad(y_seg[bounding_box_to_slice(y_bbox)], y_pad, mode='constant', constant_values=0)

    assert x.shape == y.shape
    # x = resize(x[x_bbox], img_size, order=1, preserve_range=True)
    # x_seg = resize(x_seg[x_bbox], img_size, order=0, preserve_range=True)
    # y = resize(y[y_bbox], img_size, order=1, preserve_range=True)
    # y_seg = resize(y_seg[y_bbox], img_size, order=0, preserve_range=True)
    return Rescale01_clip995(x), Rescale01_clip995(y), x_seg, y_seg


def center_pad(x, y, x_seg, y_seg, img_size):
    if x.shape[0] <= img_size[0] or x.shape[1] <= img_size[1] or \
            x.shape[2] <= img_size[2]:
        pw = max((img_size[0] - x.shape[0]) // 2, 0)
        ph = max((img_size[1] - x.shape[1]) // 2, 0)
        pd = max((img_size[2] - x.shape[2]) // 2, 0)
        x = np.pad(x, [(pw, max(img_size[0] - x.shape[0] - pw, 0)),
                       (ph, max(img_size[1] - x.shape[1] - ph, 0)),
                       (pd, max(img_size[2] - x.shape[2] - pd, 0))], mode='constant', constant_values=0)
        x_seg = np.pad(x_seg, [(pw, max(img_size[0] - x_seg.shape[0] - pw, 0)),
                               (ph, max(img_size[1] - x_seg.shape[1] - ph, 0)),
                               (pd, max(img_size[2] - x_seg.shape[2] - pd, 0))], mode='constant', constant_values=0)

    if y.shape[0] <= img_size[0] or y.shape[1] <= img_size[1] or \
            y.shape[2] <= img_size[2]:
        pw = max((img_size[0] - y.shape[0]) // 2, 0)
        ph = max((img_size[1] - y.shape[1]) // 2, 0)
        pd = max((img_size[2] - y.shape[2]) // 2, 0)
        y = np.pad(y, [(pw, max(img_size[0] - y.shape[0] - pw, 0)),
                       (ph, max(img_size[1] - y.shape[1] - ph, 0)),
                       (pd, max(img_size[2] - y.shape[2] - pd, 0))], mode='constant', constant_values=0)
        y_seg = np.pad(y_seg, [(pw, max(img_size[0] - y_seg.shape[0] - pw, 0)),
                               (ph, max(img_size[1] - y_seg.shape[1] - ph, 0)),
                               (pd, max(img_size[2] - y_seg.shape[2] - pd, 0))], mode='constant', constant_values=0)

    assert x.shape == y.shape
    return x, y, x_seg, y_seg


class AbdMR_Dataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        # path = self.paths[index]
        # tar_list = self.paths.copy()
        # tar_list.remove(path)
        # random.shuffle(tar_list)
        # tar_file = tar_list[0]
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path = self.paths[x_index]
        tar_file = self.paths[y_index]

        img = nib.load(path)
        label = nib.load(path.replace("train", "labels_all"))
        x = img.get_fdata().squeeze()
        # affine = img.affine
        x_seg = label.get_fdata().squeeze()

        img_ = nib.load(tar_file)
        label_ = nib.load(tar_file.replace("train", "labels_all"))
        # label_ = nib.load(tar_file.replace("images", "masks"))
        y = img_.get_fdata().squeeze()
        y_seg = label_.get_fdata().squeeze()

        # ROI_cut
        mask_1_path = path.replace("train", "masks_all")
        mask_1 = nib.load(mask_1_path).get_fdata().squeeze()
        mask_2_path = tar_file.replace("train", "masks_all")
        mask_2 = nib.load(mask_2_path).get_fdata().squeeze()
        img_size = mask_1.shape

        x, y, x_seg, y_seg = cropApad_noscale(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), mask_1, mask_2, img_size)
        # x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), img_size)
        x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(),
                                        img_size=[max(x.shape[i], y.shape[i]) for i in range(len(x.shape))])
        # Abdomen needs (192, 160, 192)
        scale_factor = (np.array([192, 160, 192]) / x.shape).min()
        x, y, x_seg, y_seg = scale_ratio(x, y, x_seg, y_seg, ratio=scale_factor)
        x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), img_size=(192, 160, 192))

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)


class AbdMR_InferDataset(Dataset):
    def __init__(self, data_path, transforms, istest=None, spacing=None):
        self.paths = data_path
        self.transforms = transforms
        self.istest = istest

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s

        # path = self.paths[index]
        # if index != len(self.paths) - 1:
        #     tar_file = self.paths[index + 1]
        # else:
        #     tar_file = self.paths[0]
        path = self.paths[x_index]
        tar_file = self.paths[y_index]

        a_name = int(os.path.split(path)[1].split("_")[-2])
        b_name = int(os.path.split(tar_file)[1].split("_")[-2])
        infer_save_name = "Abdomen_" + str(a_name) + "to" + str(b_name) + ".nii.gz"

        if self.istest == True:
            img = nib.load(path)
            label = nib.load(path.replace("test", "labels_all"))
            x = img.get_fdata().squeeze()
            # affine = img.affine
            x_seg = label.get_fdata().squeeze()

            img_ = nib.load(tar_file)
            label_ = nib.load(tar_file.replace("test", "labels_all"))
            # label_ = nib.load(tar_file.replace("images", "masks"))
            y = img_.get_fdata().squeeze()
            y_seg = label_.get_fdata().squeeze()

            # ROI_cut
            mask_1_path = path.replace("test", "masks_all")
            mask_1 = nib.load(mask_1_path).get_fdata().squeeze()
            mask_2_path = tar_file.replace("test", "masks_all")
            mask_2 = nib.load(mask_2_path).get_fdata().squeeze()
            img_size = mask_1.shape

        else:
            img = nib.load(path)
            label = nib.load(path.replace("val", "labels_all"))
            x = img.get_fdata().squeeze()
            # affine = img.affine
            x_seg = label.get_fdata().squeeze()

            img_ = nib.load(tar_file)
            label_ = nib.load(tar_file.replace("val", "labels_all"))
            # label_ = nib.load(tar_file.replace("images", "masks"))
            y = img_.get_fdata().squeeze()
            y_seg = label_.get_fdata().squeeze()

            # ROI_cut
            mask_1_path = path.replace("val", "masks_all")
            mask_1 = nib.load(mask_1_path).get_fdata().squeeze()
            mask_2_path = tar_file.replace("val", "masks_all")
            mask_2 = nib.load(mask_2_path).get_fdata().squeeze()
            img_size = mask_1.shape

        x, y, x_seg, y_seg = cropApad_noscale(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), mask_1, mask_2, img_size)

        x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(),
                                        img_size=[max(x.shape[i], y.shape[i]) for i in range(len(x.shape))])
        # Abdomen needs (192, 160, 192)
        scale_factor = (np.array([192, 160, 192]) / x.shape).min()
        x, y, x_seg, y_seg = scale_ratio(x, y, x_seg, y_seg, ratio=scale_factor)
        x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), img_size=(192, 160, 192))


        # x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), img_size)

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        spacing = (2, 2, 2)
        return x, y, x_seg, y_seg, infer_save_name, spacing

    def __len__(self):
        # return len(self.paths)
        return len(self.paths) * (len(self.paths) - 1)


from skimage.transform import rescale, warp, resize



def scale_ratio(x, y, x_seg, y_seg, ratio):
    x = rescale(x.copy(), ratio, order=1, preserve_range=True)
    y = rescale(y.copy(), ratio, order=1, preserve_range=True)
    # x_seg = zoom(x_seg.copy().astype('int8'), ratio, mode='nearest')
    # y_seg = zoom(y_seg.copy().astype('int8'), ratio, mode='nearest')
    x_seg = rescale(x_seg.copy().astype('int8'), ratio, order=0, preserve_range=True)
    y_seg = rescale(y_seg.copy().astype('int8'), ratio, order=0, preserve_range=True)
    return x, y, x_seg, y_seg


import pickle


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def seg_norm_lpba40(seg):
    seg_table = np.array([0,21,22,23,24,25,26,27,28,29,30,31,32,33,34,41,42,43,
                          44,45,46,47,48,49,50,61,62,63,64,65,66,67,68,81,82,83,
                          84,85,86,87,88,89,90,91,92,101,102,121,122,161,162,163,164,165,166])
    seg_out = np.zeros_like(seg)
    for i in range(len(seg_table)):
        seg_out[seg == seg_table[i]] = i
    return seg_out


def Seg_norm_7regions_FAIM(img):
    Frontal = [i for i in range(21, 35, 1)]
    Parietal = [i for i in range(41, 51, 1)]
    Occiptial = [i for i in range(61, 69, 1)]
    Temporal = [i for i in range(81, 93, 1)]
    Cingulate = [101, 102, 121, 122]
    Hippocampus = [165, 166]
    Putamen = [163, 164]
    seg_table = [Frontal, Parietal, Occiptial, Temporal, Cingulate, Hippocampus, Putamen]
    img_out = np.zeros_like(img)
    for i in range(1, len(seg_table) + 1):
        for index in seg_table[i - 1]:
            img_out[img == index] = i
    return img_out


class LPBA_Dataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        # path = self.paths[index]
        # tar_list = self.paths.copy()
        # tar_list.remove(path)
        # random.shuffle(tar_list)
        # tar_file = tar_list[0]
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path = self.paths[x_index]
        tar_file = self.paths[y_index]

        img = nib.load(path)
        label = nib.load(path.replace("train", "label"))
        x = img.get_fdata()
        x_seg = seg_norm_lpba40(label.get_fdata())

        img_ = nib.load(tar_file)
        label_ = nib.load(tar_file.replace("train", "label"))
        y = img_.get_fdata()
        y_seg = seg_norm_lpba40(label_.get_fdata())

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        # return len(self.paths)
        return len(self.paths) * (len(self.paths) - 1)


class LPBA_InferDataset(Dataset):
    def __init__(self, data_path, transforms, istest=False, spacing=None):
        self.paths = data_path
        self.transforms = transforms
        self.istest = istest

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        # path = self.paths[index]
        # if index != len(self.paths) - 1:
        #     tar_file = self.paths[index + 1]
        # else:
        #     tar_file = self.paths[0]
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path = self.paths[x_index]
        tar_file = self.paths[y_index]

        a_name = int(os.path.split(path)[1].split(".")[0].split("S")[-1])
        b_name = int(os.path.split(tar_file)[1].split(".")[0].split("S")[-1])
        infer_save_name = "LPBA_" + str(a_name) + "to" + str(b_name) + ".nii.gz"

        img = nib.load(path)
        if self.istest:
            label = nib.load(path.replace("test", "label"))
            label_ = nib.load(tar_file.replace("test", "label"))
        else:
            label = nib.load(path.replace("val", "label"))
            label_ = nib.load(tar_file.replace("val", "label"))
        x = img.get_fdata()
        x_seg = seg_norm_lpba40(label.get_fdata())

        img_ = nib.load(tar_file)
        y = img_.get_fdata()
        y_seg = seg_norm_lpba40(label_.get_fdata())

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        spacing = (1, 1, 1)
        return x, y, x_seg, y_seg, infer_save_name, spacing

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)


class LPBA_Infer_AbdMR(Dataset):
    def __init__(self, data_path, transforms, istest=None, spacing=(1,1,1)):
        self.paths = data_path
        self.transforms = transforms
        self.spacing = spacing

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path = self.paths[x_index]
        tar_file = self.paths[y_index]

        a_name = int(os.path.split(path)[1].split("_")[-2])
        b_name = int(os.path.split(tar_file)[1].split("_")[-2])
        infer_save_name = "Abdomen_" + str(a_name) + "to" + str(b_name) + ".nii.gz"

        img = nib.load(path)
        label = nib.load(path.replace("test", "labels_all"))
        x = img.get_fdata().squeeze()
        # affine = img.affine
        x_seg = label.get_fdata().squeeze()

        img_ = nib.load(tar_file)
        label_ = nib.load(tar_file.replace("test", "labels_all"))
        # label_ = nib.load(tar_file.replace("images", "masks"))
        y = img_.get_fdata().squeeze()
        y_seg = label_.get_fdata().squeeze()

        # ROI_cut
        mask_1_path = path.replace("test", "masks_all")
        mask_1 = nib.load(mask_1_path).get_fdata().squeeze()
        mask_2_path = tar_file.replace("test", "masks_all")
        mask_2 = nib.load(mask_2_path).get_fdata().squeeze()
        img_size = mask_1.shape

        x, y, x_seg, y_seg = cropApad_noscale(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), mask_1, mask_2, img_size)
        x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(),
                                        img_size=[max(x.shape[i], y.shape[i]) for i in range(len(x.shape))])


        scale_factor = (np.array([160, 192, 160]) / x.shape).min()
        new_spacing = self.spacing / scale_factor
        x, y, x_seg, y_seg = scale_ratio(x, y, x_seg, y_seg, ratio=scale_factor)
        x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), img_size=(160, 192, 160))

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg, infer_save_name, new_spacing

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)


class LPBA_Infer_Cardiac(Dataset):
    def __init__(self, data_path, transforms, spacing=(1,1,1), istest=False):
        self.paths = data_path
        self.transforms = transforms
        self.spacing = spacing

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s

        # path = self.paths[index]
        # if index != len(self.paths) - 1:
        #     tar_file = self.paths[index + 1]
        # else:
        #     tar_file = self.paths[0]
        path = self.paths[x_index]
        tar_file = self.paths[y_index]

        a_name = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
        b_name = int(os.path.split(tar_file)[1].split(".")[0].split("_")[-1])
        infer_save_name = "Cardiac_" + str(a_name) + "to" + str(b_name) + ".nii.gz"

        img = nib.load(path)
        label = nib.load(path.replace("test", "masks_all"))
        label_ = nib.load(tar_file.replace("test", "masks_all"))

        x = img.get_fdata()
        x_seg = label.get_fdata()

        img_ = nib.load(tar_file)
        y = img_.get_fdata()
        y_seg = label_.get_fdata()

        scale_factor = (np.array([160, 192, 160]) / x.shape).min()
        new_spacing = self.spacing / scale_factor
        x, y, x_seg, y_seg = scale_ratio(x, y, x_seg, y_seg, ratio=scale_factor)
        x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), img_size=(160, 192, 160))

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg, infer_save_name, new_spacing

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)


class AbdMR_Infer_LPBA(Dataset):
    def __init__(self, data_path, transforms, spacing=(1,1,1), istest=False):
        self.paths = data_path
        self.transforms = transforms
        self.spacing = spacing

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s

        # path = self.paths[index]
        # if index != len(self.paths) - 1:
        #     tar_file = self.paths[index + 1]
        # else:
        #     tar_file = self.paths[0]
        path = self.paths[x_index]
        tar_file = self.paths[y_index]

        a_name = int(os.path.split(path)[1].split(".")[0].split("S")[-1])
        b_name = int(os.path.split(tar_file)[1].split(".")[0].split("S")[-1])
        infer_save_name = "LPBA_" + str(a_name) + "to" + str(b_name) + ".nii.gz"

        img = nib.load(path)
        label = nib.load(path.replace("test", "label"))
        label_ = nib.load(tar_file.replace("test", "label"))

        x = img.get_fdata()
        x_seg = seg_norm_lpba40(label.get_fdata())

        img_ = nib.load(tar_file)
        y = img_.get_fdata()
        y_seg = seg_norm_lpba40(label_.get_fdata())

        scale_factor = (np.array([192, 160, 192]) / x.shape).min()
        new_spacing = self.spacing / scale_factor
        x, y, x_seg, y_seg = scale_ratio(x, y, x_seg, y_seg, ratio=scale_factor)
        x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), img_size=(192, 160, 192))

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg, infer_save_name, new_spacing

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)


class AbdMR_Infer_Cardiac(Dataset):
    def __init__(self, data_path, transforms, spacing=(1,1,1), istest=False):
        self.paths = data_path
        self.transforms = transforms
        self.spacing = spacing

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s

        # path = self.paths[index]
        # if index != len(self.paths) - 1:
        #     tar_file = self.paths[index + 1]
        # else:
        #     tar_file = self.paths[0]
        path = self.paths[x_index]
        tar_file = self.paths[y_index]

        a_name = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
        b_name = int(os.path.split(tar_file)[1].split(".")[0].split("_")[-1])
        infer_save_name = "Cardiac_" + str(a_name) + "to" + str(b_name) + ".nii.gz"

        img = nib.load(path)
        label = nib.load(path.replace("test", "masks_all"))
        label_ = nib.load(tar_file.replace("test", "masks_all"))

        x = img.get_fdata()
        x_seg = label.get_fdata()

        img_ = nib.load(tar_file)
        y = img_.get_fdata()
        y_seg = label_.get_fdata()

        scale_factor = (np.array([192, 160, 192]) / x.shape).min()
        new_spacing = self.spacing / scale_factor
        x, y, x_seg, y_seg = scale_ratio(x, y, x_seg, y_seg, ratio=scale_factor)
        x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), img_size=(192, 160, 192))

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg, infer_save_name, new_spacing

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)


class Cardiac_Dataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        # x_index = index // (len(self.paths) - 1)
        # s = index % (len(self.paths) - 1)
        # y_index = s + 1 if s >= x_index else s
        # path = self.paths[x_index]
        # tar_file = self.paths[y_index]

        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]

        img = nib.load(path)
        label = nib.load(path.replace("train", "masks_all"))
        x = img.get_fdata()
        x_seg = label.get_fdata()

        img_ = nib.load(tar_file)
        label_ = nib.load(tar_file.replace("train", "masks_all"))
        y = img_.get_fdata()
        y_seg = label_.get_fdata()

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        # return len(self.paths) * (len(self.paths) - 1)
        return len(self.paths)


class Cardiac_InferDataset(Dataset):
    def __init__(self, data_path, transforms, istest=False, spacing=(1,1,1)):
        self.paths = data_path
        self.transforms = transforms
        self.istest = istest

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path = self.paths[x_index]
        tar_file = self.paths[y_index]

        a_name = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
        b_name = int(os.path.split(tar_file)[1].split(".")[0].split("_")[-1])
        infer_save_name = "Cardiac_" + str(a_name) + "to" + str(b_name) + ".nii.gz"

        img = nib.load(path)
        if self.istest:
            label = nib.load(path.replace("test", "masks_all"))
            label_ = nib.load(tar_file.replace("test", "masks_all"))
        else:
            label = nib.load(path.replace("val", "masks_all"))
            label_ = nib.load(tar_file.replace("val", "masks_all"))
        x = img.get_fdata()
        x_seg = label.get_fdata()

        img_ = nib.load(tar_file)
        y = img_.get_fdata()
        y_seg = label_.get_fdata()

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        spacing=(1,1,1)
        return x, y, x_seg, y_seg, infer_save_name, spacing

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)


class Cardiac_Infer_AbdMR(Dataset):
    def __init__(self, data_path, transforms, spacing, istest=None):
        self.paths = data_path
        self.transforms = transforms
        self.spacing = spacing

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path = self.paths[x_index]
        tar_file = self.paths[y_index]

        a_name = int(os.path.split(path)[1].split("_")[-2])
        b_name = int(os.path.split(tar_file)[1].split("_")[-2])
        infer_save_name = "Abdomen_" + str(a_name) + "to" + str(b_name) + ".nii.gz"

        img = nib.load(path)
        label = nib.load(path.replace("test", "labels_all"))
        x = img.get_fdata().squeeze()
        # affine = img.affine
        x_seg = label.get_fdata().squeeze()

        img_ = nib.load(tar_file)
        label_ = nib.load(tar_file.replace("test", "labels_all"))
        # label_ = nib.load(tar_file.replace("images", "masks"))
        y = img_.get_fdata().squeeze()
        y_seg = label_.get_fdata().squeeze()

        # ROI_cut
        mask_1_path = path.replace("test", "masks_all")
        mask_1 = nib.load(mask_1_path).get_fdata().squeeze()
        mask_2_path = tar_file.replace("test", "masks_all")
        mask_2 = nib.load(mask_2_path).get_fdata().squeeze()
        img_size = mask_1.shape

        x, y, x_seg, y_seg = cropApad_noscale(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), mask_1, mask_2, img_size)
        x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(),
                                        img_size=[max(x.shape[i], y.shape[i]) for i in range(len(x.shape))])

        # OASIS needs (160, 224, 192)
        # Abdomen (160, 192, 192)
        scale_factor = (np.array([128, 128, 96]) / x.shape).min()
        new_spacing = self.spacing / scale_factor
        x, y, x_seg, y_seg = scale_ratio(x, y, x_seg, y_seg, ratio=scale_factor)
        x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), img_size=(128, 128, 96))



        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg, infer_save_name, new_spacing

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)


class Cardiac_Infer_LPBA(Dataset):
    def __init__(self, data_path, transforms, spacing=(1,1,1), istest=False):
        self.paths = data_path
        self.transforms = transforms
        self.spacing = spacing

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s

        # path = self.paths[index]
        # if index != len(self.paths) - 1:
        #     tar_file = self.paths[index + 1]
        # else:
        #     tar_file = self.paths[0]
        path = self.paths[x_index]
        tar_file = self.paths[y_index]

        a_name = int(os.path.split(path)[1].split(".")[0].split("S")[-1])
        b_name = int(os.path.split(tar_file)[1].split(".")[0].split("S")[-1])
        infer_save_name = "LPBA_" + str(a_name) + "to" + str(b_name) + ".nii.gz"

        img = nib.load(path)
        label = nib.load(path.replace("test", "label"))
        label_ = nib.load(tar_file.replace("test", "label"))

        x = img.get_fdata()
        x_seg = seg_norm_lpba40(label.get_fdata())

        img_ = nib.load(tar_file)
        y = img_.get_fdata()
        y_seg = seg_norm_lpba40(label_.get_fdata())

        scale_factor = (np.array([128, 128, 96]) / x.shape).min()
        new_spacing = self.spacing / scale_factor
        x, y, x_seg, y_seg = scale_ratio(x, y, x_seg, y_seg, ratio=scale_factor)
        x, y, x_seg, y_seg = center_pad(x.copy(), y.copy(), x_seg.copy(), y_seg.copy(), img_size=(128, 128, 96))

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg, infer_save_name, new_spacing

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)