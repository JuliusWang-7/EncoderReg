'''
VoxelMorph

Original code retrieved from:
https://github.com/sungonce/SENet

Original paper:
Lee S, Lee S, Seong H, et al. Revisiting self-similarity: Structural embedding for image retrieval[C]//
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 23412-23421.

Modified and tested by:
Zhuoyuan Wang
2018222020@email.szu.edu.cn
Shenzhen University
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import unfoldNd


class SSM(nn.Module):
    def __init__(self, in_ch, mid_ch, unfold_size=7, ksize=3):
        super(SSM, self).__init__()
        self.ch_reduction_encoder = nn.Conv3d(in_ch, mid_ch, kernel_size=1, bias=False, padding=0)
        self.SCC = SelfCorrelationComputation(unfold_size=unfold_size)
        self.SSE = SelfSimilarityEncoder(in_ch, mid_ch, unfold_size=unfold_size, ksize=ksize)
        self.FFN = nn.Sequential(nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=True, padding=0),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=True, padding=0))

    def forward(self, ssm_input_feat):
        q = self.ch_reduction_encoder(ssm_input_feat)
        q = F.normalize(q, dim=1, p=2)

        self_sim = self.SCC(q)
        self_sim_feat = self.SSE(self_sim)
        ssm_output_feat = ssm_input_feat + self_sim_feat
        ssm_output_feat = self.FFN(ssm_output_feat)

        return ssm_output_feat


class SelfCorrelationComputation(nn.Module):
    def __init__(self, unfold_size=5):
        super(SelfCorrelationComputation, self).__init__()
        self.unfold_size = (unfold_size, unfold_size, unfold_size)
        self.padding_size = unfold_size // 2
        # self.unfold = nn.Unfold(kernel_size=self.unfold_size, padding=self.padding_size)

    def forward(self, q):
        b, c, l, w, h = q.shape

        q_unfold = unfoldNd.unfoldNd(q, kernel_size=self.unfold_size, padding=self.padding_size)  # b, c, uvt, l, w, h
        q_unfold = q_unfold.view(b, c, self.unfold_size[0],
                                 self.unfold_size[1], self.unfold_size[2], l, w, h)  # b, c, u, v, t, l, w, h
        self_sim = q_unfold * q.unsqueeze(2).unsqueeze(2).unsqueeze(2)  # b, c, u, v, t, l, w, h * b, c, 1, 1, 1, l, w, h
        self_sim = self_sim.permute(0, 1, 5, 6, 7, 2, 3, 4).contiguous()  # b, c, l, w, h, u, v, t

        return self_sim.clamp(min=0)


class SelfSimilarityEncoder(nn.Module):
    def __init__(self, in_ch, mid_ch, unfold_size, ksize):
        super(SelfSimilarityEncoder, self).__init__()

        def make_building_conv_block(in_channel, out_channel, ksize, padding=(0, 0, 0), stride=(1, 1, 1), bias=True,
                                     conv_group=1):
            building_block_layers = []
            building_block_layers.append(nn.Conv3d(in_channel, out_channel, (ksize, ksize, ksize),
                                                   stride=stride, bias=bias, groups=conv_group, padding=padding))
            building_block_layers.append(nn.BatchNorm3d(out_channel))
            building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        conv_in_block_list = [make_building_conv_block(mid_ch, mid_ch, ksize) for _ in range(unfold_size // 2)]
        self.conv_in = nn.Sequential(*conv_in_block_list)
        self.conv1x1_out = nn.Sequential(
            nn.Conv3d(mid_ch, in_ch, kernel_size=1, bias=True, padding=0),
            nn.InstanceNorm3d(in_ch),
            nn.LeakyReLU(0.1))

    def forward(self, x):
        b, c, l, w, h, u, v, t = x.shape
        x = x.permute(0, 5, 6, 7, 1, 2, 3, 4).contiguous()
        x = x.view(b * l * w * h, c, u, v, t)
        x = self.conv_in(x)
        c = x.shape[1]
        x = x.view(b, l, w, h, c, 1, 1, 1).permute(0, 4, 1, 2, 3, 5, 6, 7).contiguous()
        x = x.mean(dim=[-1, -2, -3])
        x = self.conv1x1_out(x)  # [B, C3, H, W] -> [B, C4, H, W]

        return x





