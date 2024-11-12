import glob
import sys
import losses, utils
from torch.utils.data import DataLoader
from data.datasets_3 import *
from data import trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.EncoderReg import EncoderReg
import nibabel
from resnet import resnet50
from sklearn.decomposition import PCA
import torch.nn.functional as nnF

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(Dataset):
    batch_size = 1
    if Dataset == "Cardiac":
        num_class = 4
        data_root = './Cardiac_MM'
        weights = [2, 1, 2]  # loss weights
        max_epoch = 300
    elif Dataset == "AbdMR":
        num_class = 5
        data_root = './Abdomen_MR'
        weights = [2, 1, 2]
        max_epoch = 40
    elif Dataset == "LPBA":
        num_class = 55
        data_root = './LPBA40'
        weights = [1, 1, 1]
        max_epoch = 50
    else:
        raise ValueError

    stage1 = int(max_epoch)
    stage2 = int(max_epoch*0.75)
    stage3 = int(max_epoch*0.25)
    max_epoch = stage1 + stage2 + stage3

    train_dir = os.path.join(data_root, 'train')
    train_imgs = glob.glob(os.path.join(train_dir, "*.nii.gz"))
    val_imgs = glob.glob(os.path.join(train_dir.replace("train", "val"), "*.nii.gz"))
    img_size = nibabel.load(train_imgs[0]).get_fdata().shape

    save_root = './reg_UR/{}'.format(Dataset)
    save_exp = save_root + '/EncoderReg/'
    if not os.path.exists(save_exp):
        os.makedirs(save_exp)

    lr = 0.0001
    epoch_start = 0
    cont_training = False

    '''
    Initialize model
    '''
    model = EncoderReg(img_size)
    model.cuda()

    '''
    Initialize Gen model
    '''
    gen_model = nn.DataParallel(resnet50(sample_input_D=img_size[0],
                                         sample_input_H=img_size[1],
                                         sample_input_W=img_size[2],
                                         num_seg_classes=2))
    gen_model.cuda()
    gen_model.load_state_dict(torch.load('./resnet_50_23dataset.pth')['state_dict'])
    gen_model.eval()
    epoch_use_mi = int(stage1 * 0.8)

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 0
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        best_model = torch.load(save_exp + natsorted(os.listdir(save_exp))[-3])['state_dict']
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),
                                       ])

    train_dataset_name = "{}_Dataset".format(Dataset)
    val_dataset_name = "{}_InferDataset".format(Dataset)
    train_dataloader = globals()[train_dataset_name]
    train_set = train_dataloader(train_imgs, transforms=train_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    val_dataloader = globals()[val_dataset_name]
    val_set = val_dataloader(val_imgs, transforms=val_composed, istest=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    best_dsc = 0
    train_stage = 1
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        if epoch > epoch_use_mi and epoch < stage1:
            criterions = [losses.NCC_vxm(), losses.Grad3d(penalty='l2'), losses.MutualInformation()]
            train_stage = 1
        elif epoch == stage1:
            # 此阶段开始要加载第一阶段最好模型，同时开启第二阶段了
            print("loading stage1 best model")
            best_model = torch.load(save_exp + "stage_1/" + natsorted(os.listdir(save_exp + "stage_1/"))[-1])[
                'state_dict']
            model.load_state_dict(best_model)
            criterions = [losses.SSIM3D(), losses.Grad3d(penalty='l2')]
            train_stage = 2
        elif epoch == epoch_use_mi:
            criterions = [losses.NCC_vxm(), losses.Grad3d(penalty='l2'), losses.MutualInformation()]
            train_stage = 1
            # 删除前面保存的模型，因为那些不具有泛化性
            for f in os.listdir(save_exp + "stage_1/"):
                os.remove(os.path.join(save_exp + "stage_1/", f))
        elif epoch < epoch_use_mi:
            criterions = [losses.NCC_vxm(), losses.Grad3d(penalty='l2')]
            train_stage = 1
        elif epoch > stage1 and epoch < stage1 + stage2:
            # criterions = [losses.SSIM3D(), losses.Grad3d(penalty='l2'), nn.MSELoss()]
            criterions = [losses.SSIM3D(), losses.Grad3d(penalty='l2')]
            train_stage = 2
        elif epoch == stage1 + stage2:
            print("loading stage2 best model")
            best_model = torch.load(save_exp + "stage_2/" + natsorted(os.listdir(save_exp + "stage_2/"))[-1])[
                'state_dict']
            model.load_state_dict(best_model)
            criterions = [losses.NCC_vxm(), losses.Grad3d(penalty='l2'), losses.SSIM3D()]
            train_stage = 3
        else:
            criterions = [losses.NCC_vxm(), losses.Grad3d(penalty='l2'), losses.SSIM3D()]
            train_stage = 3
        for data in train_loader:
            if epoch < stage1:
                adjust_learning_rate(optimizer, epoch, stage1, lr)
            elif epoch >= stage1 and epoch < stage1 + stage2:
                adjust_learning_rate(optimizer, epoch - stage1, stage2, lr)
            elif epoch >= stage1 + stage2:
                adjust_learning_rate(optimizer, epoch - stage1 - stage2, stage3, lr)
            idx += 1
            model.train()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            # x_in = torch.cat((x, y), dim=1)
            output = model(x, y, stage=train_stage)

            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                if train_stage == 1:
                    if n <= 1:
                        curr_loss = loss_function(output[n], y) * weights[n]
                    else:
                        M1, F1 = output[n]
                        M_Res = gen_model(x)
                        F_Res = gen_model(y)
                        pca = PCA(n_components=M1.size(1))

                        B, C, H, W, D = M_Res.size()
                        M_Res_flatten = M_Res.view(B*C, H*W*D).transpose(0,1).detach().cpu().numpy()
                        F_Res_flatten = F_Res.view(B * C, H * W * D).transpose(0,1).detach().cpu().numpy()
                        pca.fit(M_Res_flatten)
                        M_Res_decom = torch.tensor(pca.transform(M_Res_flatten)).cuda().transpose(0, 1).view(B, M1.size(1), H, W, D)
                        pca.fit(F_Res_flatten)
                        F_Res_decom = torch.tensor(pca.transform(F_Res_flatten)).cuda().transpose(0, 1).view(B, F1.size(1), H, W, D)


                        curr_loss = (loss_function(M_Res_decom, nnF.interpolate(M1, size=(H,W,D), mode='trilinear')) +
                                     loss_function(F_Res_decom, nnF.interpolate(F1, size=(H,W,D), mode='trilinear'))) / 2 * weights[n]
                elif train_stage == 2:
                    curr_loss = loss_function(output[n], y) * weights[n]
                elif train_stage == 3:
                    if n <= 1:
                        curr_loss = loss_function(output[n], y) * weights[n]
                    else:
                        curr_loss = loss_function(output[0], y) * weights[n]
                else:
                    raise ValueError

                    # continue
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item()))

        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data[:4]]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                output = model(x, y, stage=train_stage)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                dsc = utils.dice_val(def_out.long(), y_seg.long(), num_clus=num_class)
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir=save_exp+"stage_{}/".format(train_stage), filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        loss_all.reset()



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


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
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
    datasets = ["Cardiac", "AbdMR", "LPBA"]
    for d in datasets:
        main(Dataset=d)
