import glob
import time

from torch.utils.tensorboard import SummaryWriter
import os
import losses, utils
import sys
from torch.utils.data import DataLoader
from data import trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from data.datasets_3 import *
# model list
from models.EncoderReg import EncoderReg

import nibabel
import  xlsxwriter

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def make_one_hot(mask, num_class):
    # 数据转为one hot 类型
    # mask_unique = np.unique(mask)
    mask_unique = [m for m in range(num_class)]
    one_hot_mask = [mask == i for i in mask_unique]
    one_hot_mask = np.stack(one_hot_mask)
    return one_hot_mask


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(dataset=None, model_name=None, stage=4):
    target = dataset.split("_")[-1]
    source = dataset.split("_")[0]

    if target == 'AbdMR':
        test_dir = './Abdomen_MR/test'
    elif target == 'LPBA':
        test_dir = './LPBA40/test'
    else:
        test_dir = './Cardiac_MM/test'

    imgs_path = glob.glob(os.path.join(test_dir, "*.nii.gz"))
    imgs_path.sort()

    if source == "AbdMR":
        img_size = (192, 160, 192)
    elif source == "LPBA":
        img_size = (160, 192, 160)
    elif source == "Cardiac":
        img_size = (128, 128, 96)
    else:
        raise ValueError

    model_root = os.path.join('./reg_UR', source)
    model_path = os.path.join(model_root, model_name, "stage_{}".format(stage))
    if not os.path.exists(model_path):
        print(model_path)
        raise ValueError


    if target == "AbdMR":
        weights = [2, 1, 2]  # loss weights
        spacing = (2, 2, 2)
    else:
        weights = [1, 1, 1]
        spacing = (1, 1, 1)

    save_root = os.path.join("./results", dataset)
    save_seg = save_root + '/segmentations/' + model_name + "_stage_{}".format(stage)
    save_loss = save_root + '/loss_file/' + model_name + "_stage_{}".format(stage)
    save_moving = save_root + '/moving'
    save_fixed = save_root + '/fixed'
    if not os.path.exists(save_seg):
        os.makedirs(save_seg)
    if not os.path.exists(save_loss):
        os.makedirs(save_loss)
    if not os.path.exists(save_moving):
        os.makedirs(save_moving)
    if not os.path.exists(save_fixed):
        os.makedirs(save_fixed)



    new_book = xlsxwriter.Workbook(os.path.join(save_loss, "loss_det.xls"))
    loss_table = new_book.add_worksheet('loss')
    field_table = new_book.add_worksheet('field')
    dsc_table = new_book.add_worksheet('dsc')
    time_table = new_book.add_worksheet('time')


    lr = 0.0001
    epoch_start = 0
    max_epoch = 20

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    updated_lr = lr

    '''
    Initialize training
    '''
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    if target == "AbdMR":
        num_class = 5
    elif target == "LPBA":
        num_class = 55
    elif target == "Cardiac":
        num_class = 4
    else:
        raise ValueError

    if source == target:
        infer_name = globals()["{}_InferDataset".format(source)]
    else:
        infer_name = globals()[dataset]
    test_set = infer_name(imgs_path, transforms=test_composed, spacing=spacing, istest=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]

    best_dsc = 0
    idx = 0

    train_time_all = utils.AverageMeter()
    infer_time_all = utils.AverageMeter()
    loading_time_all = utils.AverageMeter()

    for data in test_loader:
        print('Training Starts')
        eval_loss_50 = utils.AverageMeter()
        loss_all = utils.AverageMeter()
        idx += 1

        save_name = data[4][0]
        print(save_name)

        '''
            Initialize model
        '''
        a_start = time.time()
        model = EncoderReg(img_size)
        model.cuda()
        best_model = torch.load(model_path +"/"+ natsorted(os.listdir(model_path))[-1], map_location='cuda:0')['state_dict']
        model.load_state_dict(best_model)

        optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
        a_end = time.time()
        loading_time_all.update(a_end - a_start)

        data = [t.cuda() for t in data[:4]]
        x = data[0]
        y = data[1]
        x_seg = data[2]
        y_seg = data[3]
        r_index = 0
        f_r_index = 0
        for epoch in range(epoch_start, max_epoch + 1):
            r_index += 1
            if epoch in [0, 5, 10, 20]:
                f_r_index += 1
                with torch.no_grad():
                    model.eval()
                    b_start = time.time()
                    if stage == 3:
                        output = model(x, y, stage=4)
                    else:
                        output = model(x, y, stage=stage)
                    def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                    b_end = time.time()
                    infer_time_all.update(b_end - b_start)

                    dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), num_clus=num_class)
                    dsc_table.write(f_r_index, idx, "%.3e" % (dsc_trans))

                    out = def_out.detach().cpu().numpy()[0, 0, :, :, :]
                    save_path = os.path.join(save_seg, str(epoch))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    nibabel.save(nibabel.Nifti1Image(out.astype('int8'), np.eye(4)), os.path.join(save_path, save_name))

                    flow = output[1]
                    tar = y.detach().cpu().numpy()[0, 0, :, :, :]
                    jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])

                field_table.write(f_r_index, idx, "%.3e"%(np.sum(jac_det <= 0) / np.prod(tar.shape)))


            model.train()
            c_start = time.time()
            if stage == 3:
                output = model(x, y, stage=4)
            else:
                output = model(x, y, stage=stage)


            adjust_learning_rate(optimizer, epoch, max_epoch, updated_lr)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                if n < 2:
                    curr_loss = loss_function(output[n], y) * weights[n]
                else:
                    curr_loss = loss_function(output[0], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_table.write(r_index, idx, "%.3e"%(loss))
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            c_end = time.time()
            train_time_all.update(c_end - c_start)

            print('Epoch {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(epoch + 1, max_epoch, loss.item(),
                                                                                    loss_vals[0].item(),
                                                                                    loss_vals[1].item()))

        loss_all.reset()

    time_table.write(1, 1, 'time_cost')
    time_table.write(1, 1, '{:.1f} ± {:.1f}'.format(train_time_all.avg, train_time_all.std))
    time_table.write(2, 1, '{:.1f} ± {:.1f}'.format(infer_time_all.avg, infer_time_all.std))
    time_table.write(3, 1, '{:.1f} ± {:.1f}'.format(loading_time_all.avg, loading_time_all.std))

    new_book.close()


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))

    main(dataset="LPBA_Infer_AbdMR", model_name="EncoderReg", stage=3)









