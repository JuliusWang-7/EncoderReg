'''
VoxelMorph

Original code retrieved from:
https://github.com/voxelmorph/voxelmorph

Original paper:
Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019).
VoxelMorph: a learning framework for deformable medical image registration.
IEEE transactions on medical imaging, 38(8), 1788-1800.

Modified and tested by:
Zhuoyuan Wang
2018222020@email.szu.edu.cn
Shenzhen University
'''

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
from functional import modetqkrpb_cu
from self_similarity import SSM


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

    def freeze(self):
        self.main.requires_grad = False

    def unfreeze(self):
        self.main.requires_grad = True


class Encoder_G(nn.Module):
    def __init__(self, in_channel=1, first_out_channel=4):
        super(Encoder_G, self).__init__()

        c = first_out_channel

        self.conv0 = nn.Sequential(
            ConvBlock(in_channel, c),
            ConvInsBlock(c, 2*c),
            ConvInsBlock(2*c, 2*c)
        )

        self.conv1 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(2 * c, 4 * c),
            ConvInsBlock(4 * c, 4 * c)
        )

        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(4 * c, 8 * c),
            ConvInsBlock(8 * c, 8 * c)
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(8 * c, 16* c),
            ConvInsBlock(16 * c, 16 * c)
        )

        self.conv4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(16 * c, 32 * c),
            ConvInsBlock(32 * c, 32 * c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/8

        return out0, out1, out2, out3, out4

    def freeze(self):
        for block in self.conv0.modules():
            if isinstance(block, ConvInsBlock):
                block.freeze()

        for block in self.conv1.modules():
            if isinstance(block, ConvInsBlock):
                block.freeze()

        for block in self.conv2.modules():
            if isinstance(block, ConvInsBlock):
                block.freeze()

        for block in self.conv3.modules():
            if isinstance(block, ConvInsBlock):
                block.freeze()

        for block in self.conv4.modules():
            if isinstance(block, ConvInsBlock):
                block.freeze()

    def unfreeze(self):
        for block in self.conv0.modules():
            if isinstance(block, ConvInsBlock):
                block.unfreeze()

        for block in self.conv1.modules():
            if isinstance(block, ConvInsBlock):
                block.unfreeze()

        for block in self.conv2.modules():
            if isinstance(block, ConvInsBlock):
                block.unfreeze()

        for block in self.conv3.modules():
            if isinstance(block, ConvInsBlock):
                block.unfreeze()

        for block in self.conv4.modules():
            if isinstance(block, ConvInsBlock):
                block.unfreeze()


class Encoder_S(nn.Module):
    def __init__(self, in_channel=1, first_out_channel=4):
        super(Encoder_S, self).__init__()

        c = first_out_channel

        self.conv0 = nn.Sequential(
            ConvBlock(in_channel, c),
            ConvInsBlock(c, 2*c),
            ConvInsBlock(2*c, 2*c)
        )

        self.conv1 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(2 * c, 4 * c),
            ConvInsBlock(4 * c, 4 * c)
        )

        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(4 * c, 8 * c),
            ConvInsBlock(8 * c, 8 * c)
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(8 * c, 16* c),
            ConvInsBlock(16 * c, 16 * c),
            SSM(in_ch=16 * c, mid_ch=8 * c)
        )

        self.conv4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(16 * c, 32 * c),
            ConvInsBlock(32 * c, 32 * c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/8
        return out0, out1, out2, out3, out4

    def freeze(self):
        for block in self.conv0.modules():
            if isinstance(block, ConvInsBlock):
                block.freeze()

        for block in self.conv1.modules():
            if isinstance(block, ConvInsBlock):
                block.freeze()

        for block in self.conv2.modules():
            if isinstance(block, ConvInsBlock):
                block.freeze()

        for block in self.conv3.modules():
            if isinstance(block, ConvInsBlock):
                block.freeze()

        for block in self.conv4.modules():
            if isinstance(block, ConvInsBlock):
                block.freeze()

    def unfreeze(self):
        for block in self.conv0.modules():
            if isinstance(block, ConvInsBlock):
                block.unfreeze()

        for block in self.conv1.modules():
            if isinstance(block, ConvInsBlock):
                block.unfreeze()

        for block in self.conv2.modules():
            if isinstance(block, ConvInsBlock):
                block.unfreeze()

        for block in self.conv3.modules():
            if isinstance(block, ConvInsBlock):
                block.unfreeze()

        for block in self.conv4.modules():
            if isinstance(block, ConvInsBlock):
                block.unfreeze()


class PositionalEncodingLayer(nn.Module):
    def __init__(self, in_channels, dim=6, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(dim)
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = self.norm(self.proj(feat))
        return feat


class DFIBlock(nn.Module):
    def __init__(self, in_channels):
        super(DFIBlock, self).__init__()
        self.conv3 = nn.Conv3d(in_channels, 3, 3, 1, 1)
        self.conv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv3.weight.shape))
        self.conv3.bias = nn.Parameter(torch.zeros(self.conv3.bias.shape))

    def forward(self, x):
        # x = self.upsample(x)
        # res = self.conv1(x)
        # x = self.conv2(res)
        # x = res + x
        x = self.conv3(x)
        return x

    def freeze(self):
        self.conv3.requires_grad = False


class ModeT(nn.Module):
    def __init__(self, img_size, dim, num_heads, kernel_size=3, qk_scale=None, use_rpb=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = kernel_size
        self.use_rpb = use_rpb
        if use_rpb:
            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.rpb_size, self.rpb_size, self.rpb_size))
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [kernel_size] * 3]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids, -1).type(torch.FloatTensor)
        v = grid.reshape(self.kernel_size ** 3, 3)
        self.register_buffer('v', v)

    def forward(self, q, k):
        B, H, W, T, C = q.shape
        N = H * W * T
        num_tokens = int(self.kernel_size ** 3)
        q = q.reshape(B, H, W, T, self.num_heads, C // self.num_heads).permute(0,4,1,2,3,5) * self.scale  #1,heads,H,W,T,dims
        pd = self.kernel_size - 1  # 2
        pdr = pd // 2  # 1
        k = k.permute(0, 4, 1, 2, 3)  # 1, C, H, W, T
        k = nnf.pad(k, (pdr, pdr, pdr, pdr, pdr, pdr))  # 1, C, H+2, W+2, T+2
        k = k.reshape(B, self.num_heads, C // self.num_heads, H+pd,W+pd,T+pd).permute(0, 1, 3, 4, 5, 2) # 1,heads,H+2,W+2,T+2,dims
        attn = modetqkrpb_cu(q,k,self.rpb)
        attn = attn.softmax(dim=-1)  # B h H W T num_tokens
        attn = attn.permute(0,2,3,4,1,5).reshape(B,N,self.num_heads,1,num_tokens)
        # v: B, N, heads, num_tokens, 3
        x = (attn @ self.v)  # B x N x heads x 1 x 3
        x = x.reshape(B, H, W, T, self.num_heads*3).permute(0, 4, 1, 2, 3)

        return x


class EncoderReg(nn.Module):
    def __init__(self,
                 inshape=(160, 192, 160),
                 in_channel=1,
                 channels=8,
                 head_dim=6,
                 num_heads=[8, 4, 2, 1, 1],
                 scale=1):
        super(EncoderReg, self).__init__()
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        dims = len(inshape)

        c = self.channels
        self.encoder_g = Encoder_G(in_channel=in_channel, first_out_channel=c)
        self.encoder_s = Encoder_S(in_channel=in_channel, first_out_channel=c)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)#nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.peblock1 = PositionalEncodingLayer(2*c, dim=head_dim*num_heads[4])
        self.modet1 = ModeT(inshape, head_dim*num_heads[4], num_heads[4], qk_scale=scale)
        self.dfi1 = DFIBlock(3 * num_heads[4], 3 * 2 * num_heads[4])

        self.peblock2 = PositionalEncodingLayer(4*c, dim=head_dim*num_heads[3])
        self.modet2 = ModeT([s//2 for s in inshape],head_dim*num_heads[3], num_heads[3], qk_scale=scale)
        self.dfi2 = DFIBlock(3 * num_heads[3], 3 * 2 * num_heads[3])

        self.peblock3 = PositionalEncodingLayer(8*c, dim=head_dim*num_heads[2])
        self.modet3 = ModeT([s//4 for s in inshape],head_dim*num_heads[2], num_heads[2], qk_scale=scale)
        self.dfi3 = DFIBlock(3 * num_heads[2], 3 * num_heads[2] * 2)

        self.peblock4 = PositionalEncodingLayer(16*c, dim=head_dim*num_heads[1])
        self.modet4 = ModeT([s//8 for s in inshape],head_dim*num_heads[1], num_heads[1], qk_scale=scale)
        self.dfi4 = DFIBlock(3 * num_heads[1], 3 * num_heads[1] * 2)

        self.peblock5 = PositionalEncodingLayer(32*c, dim=head_dim*num_heads[0])
        self.modet5 = ModeT([s//16 for s in inshape], head_dim*num_heads[0], num_heads[0], qk_scale=scale)
        self.dfi5 = DFIBlock(3*num_heads[0], 3*num_heads[0]*2)

        self.fusion_blocks = nn.ModuleList([nn.Conv3d(2 * c * 2**(i+1), c * 2**(i+1), 3, 1, 1) for i in range(len(num_heads))])

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in inshape]))

    def forward(self, moving, fixed, stage=3):

        if stage == 1:
            # encode stage
            self.encoder_s.freeze()
            M1, M2, M3, M4, M5 = self.encoder_g(moving)
            F1, F2, F3, F4, F5 = self.encoder_g(fixed)

            originM1F1 = (M1.clone(), F1.clone())

            q5, k5 = self.peblock5(F5), self.peblock5(M5)
            w = self.modet5(q5, k5)
            w = self.dfi5(w)
            flow = self.upsample_trilin(2 * w)

            M4 = self.transformer[3](M4, flow)
            q4, k4 = self.peblock4(F4), self.peblock4(M4)
            w = self.modet4(q4, k4)
            w = self.dfi4(w)
            flow = self.upsample_trilin(2 * (self.transformer[3](flow, w) + w))

            M3 = self.transformer[2](M3, flow)
            q3, k3 = self.peblock3(F3), self.peblock3(M3)
            w = self.modet3(q3, k3)
            w = self.dfi3(w)
            flow = self.upsample_trilin(2 * (self.transformer[2](flow, w) + w))

            M2 = self.transformer[1](M2, flow)
            q2, k2 = self.peblock2(F2), self.peblock2(M2)
            w = self.modet2(q2, k2)
            w = self.dfi2(w)
            flow = self.upsample_trilin(2 * (self.transformer[1](flow, w) + w))

            M1 = self.transformer[0](M1, flow)
            q1, k1 = self.peblock1(F1), self.peblock1(M1)
            w = self.modet1(q1, k1)
            w = self.dfi1(w)
            flow = self.transformer[0](flow, w) + w

            y_moved = self.transformer[0](moving, flow)

            return y_moved, flow, originM1F1

        elif stage == 2:
            self.encoder_g.freeze()
            self.encoder_s.unfreeze()
            # encode2 stage
            m1, m2, m3, m4, m5 = self.encoder_s(moving)
            f1, f2, f3, f4, f5 = self.encoder_S(fixed)

            q5, k5 = self.peblock5(f5), self.peblock5(m5)
            w = self.modet5(q5, k5)
            w = self.dfi5(w)
            flow = self.upsample_trilin(2 * w)

            m4 = self.transformer[3](m4, flow)
            q4, k4 = self.peblock4(f4), self.peblock4(m4)
            w = self.modet4(q4, k4)
            w = self.dfi4(w)
            flow = self.upsample_trilin(2 * (self.transformer[3](flow, w) + w))

            m3 = self.transformer[2](m3, flow)
            q3, k3 = self.peblock3(f3), self.peblock3(m3)
            w = self.modet3(q3, k3)
            w = self.dfi3(w)
            flow = self.upsample_trilin(2 * (self.transformer[2](flow, w) + w))

            m2 = self.transformer[1](m2, flow)
            q2, k2 = self.peblock2(f2), self.peblock2(m2)
            w = self.modet2(q2, k2)
            w = self.dfi2(w)
            flow = self.upsample_trilin(2 * (self.transformer[1](flow, w) + w))

            m1 = self.transformer[0](m1, flow)
            q1, k1 = self.peblock1(f1), self.peblock1(m1)
            w = self.modet1(q1, k1)
            w = self.dfi1(w)
            flow = self.transformer[0](flow, w) + w

            y_moved = self.transformer[0](moving, flow)

            return y_moved, flow, []

        elif stage == 3:
            self.encoder_g.freeze()
            self.encoder_s.freeze()
            M1, M2, M3, M4, M5 = self.encoder_g(moving)
            F1, F2, F3, F4, F5 = self.encoder_g(fixed)

            m1, m2, m3, m4, m5 = self.encoder_s(moving)
            f1, f2, f3, f4, f5 = self.encoder_s(fixed)

            F5, M5 = self.fusion_blocks[4](torch.cat((F5, f5), dim=1)), self.fusion_blocks[4](torch.cat((M5, m5), dim=1))
            q5, k5 = self.peblock5(F5), self.peblock5(M5)
            w = self.modet5(q5, k5)
            w = self.dfi5(w)
            flow = self.upsample_trilin(2 * w)

            F4, M4 = self.fusion_blocks[3](torch.cat((F4, f4), dim=1)), self.fusion_blocks[3](torch.cat((M4, m4), dim=1))
            M4 = self.transformer[3](M4, flow)
            q4, k4 = self.peblock4(F4), self.peblock4(M4)
            w = self.modet4(q4, k4)
            w = self.dfi4(w)
            flow = self.upsample_trilin(2 * (self.transformer[3](flow, w) + w))

            F3, M3 = self.fusion_blocks[2](torch.cat((F3, f3), dim=1)), self.fusion_blocks[2](torch.cat((M3, m3), dim=1))
            M3 = self.transformer[2](M3, flow)
            q3, k3 = self.peblock3(F3), self.peblock3(M3)
            w = self.modet3(q3, k3)
            w = self.dfi3(w)
            flow = self.upsample_trilin(2 * (self.transformer[2](flow, w) + w))

            F2, M2 = self.fusion_blocks[1](torch.cat((F2, f2), dim=1)), self.fusion_blocks[1](torch.cat((M2, m2), dim=1))
            M2 = self.transformer[1](M2, flow)
            q2, k2 = self.peblock2(F2), self.peblock2(M2)
            w = self.modet2(q2, k2)
            w = self.dfi2(w)
            flow = self.upsample_trilin(2 * (self.transformer[1](flow, w) + w))

            F1, M1 = self.fusion_blocks[0](torch.cat((F1, f1), dim=1)), self.fusion_blocks[0](torch.cat((M1, m1), dim=1))
            M1 = self.transformer[0](M1, flow)
            q1, k1 = self.peblock1(F1), self.peblock1(M1)
            w = self.modet1(q1, k1)
            w = self.dfi1(w)
            flow = self.transformer[0](flow, w) + w

            y_moved = self.transformer[0](moving, flow)

            return y_moved, flow, []

        elif stage == 4:  # one-shot stage
            self.encoder_g.freeze()
            self.encoder_s.unfreeze()
            M1, M2, M3, M4, M5 = self.encoder_g(moving)
            F1, F2, F3, F4, F5 = self.encoder_g(fixed)

            m1, m2, m3, m4, m5 = self.encoder_s(moving)
            f1, f2, f3, f4, f5 = self.encoder_s(fixed)

            F5, M5 = self.fusion_blocks[4](torch.cat((F5, f5), dim=1)), self.fusion_blocks[4](torch.cat((M5, m5), dim=1))
            q5, k5 = self.peblock5(F5), self.peblock5(M5)
            w = self.modet5(q5, k5)
            w = self.dfi5(w)
            flow = self.upsample_trilin(2 * w)

            F4, M4 = self.fusion_blocks[3](torch.cat((F4, f4), dim=1)), self.fusion_blocks[3](torch.cat((M4, m4), dim=1))
            M4 = self.transformer[3](M4, flow)
            q4, k4 = self.peblock4(F4), self.peblock4(M4)
            w = self.modet4(q4, k4)
            w = self.dfi4(w)
            flow = self.upsample_trilin(2 * (self.transformer[3](flow, w) + w))

            F3, M3 = self.fusion_blocks[2](torch.cat((F3, f3), dim=1)), self.fusion_blocks[2](torch.cat((M3, m3), dim=1))
            M3 = self.transformer[2](M3, flow)
            q3, k3 = self.peblock3(F3), self.peblock3(M3)
            w = self.modet3(q3, k3)
            w = self.dfi3(w)
            flow = self.upsample_trilin(2 * (self.transformer[2](flow, w) + w))

            F2, M2 = self.fusion_blocks[1](torch.cat((F2, f2), dim=1)), self.fusion_blocks[1](torch.cat((M2, m2), dim=1))
            M2 = self.transformer[1](M2, flow)
            q2, k2 = self.peblock2(F2), self.peblock2(M2)
            w = self.modet2(q2, k2)
            w = self.dfi2(w)
            flow = self.upsample_trilin(2 * (self.transformer[1](flow, w) + w))

            F1, M1 = self.fusion_blocks[0](torch.cat((F1, f1), dim=1)), self.fusion_blocks[0](torch.cat((M1, m1), dim=1))
            M1 = self.transformer[0](M1, flow)
            q1, k1 = self.peblock1(F1), self.peblock1(M1)
            w = self.modet1(q1, k1)
            w = self.dfi1(w)
            flow = self.transformer[0](flow, w) + w

            y_moved = self.transformer[0](moving, flow)

            return y_moved, flow, []
        else:
            raise ValueError


    def freeze_decoder(self):
        # self.dfi5.freeze()
        # self.dfi4.freeze()
        # self.dfi3.freeze()
        # self.dfi2.freeze()
        # self.dfi1.freeze()

        for param in self.modet5.parameters():
            param.requires_grad = False

        for param in self.modet4.parameters():
            param.requires_grad = False

        for param in self.modet3.parameters():
            param.requires_grad = False

        for param in self.modet2.parameters():
            param.requires_grad = False

        for param in self.modet1.parameters():
            param.requires_grad = False
