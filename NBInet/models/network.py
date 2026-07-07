import torch
import torch.nn as nn

from models.network_module import *


def weights_init(net, init_type = 'normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


class BaseModel(nn.Module):

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt

    def load_ckpt(self, model_path, force_load = False):
        state_dict = torch.load(model_path, map_location = torch.device('cpu'))

        if force_load:
            state_dict_temp = self.state_dict()
            key_temp = set(list(state_dict_temp.keys()))

            for n, p in state_dict.items():
                key_temp.discard(n)
                if n in state_dict_temp.keys():
                    if state_dict_temp[n].shape != p.data.shape:
                        print('%s size mismatch, pass!' % n)
                        continue
                    state_dict_temp[n].copy_(p.data)
                else:
                    print('%s not exist, pass!' % n)

            state_dict = state_dict_temp

            if len(key_temp) != 0:
                for k in key_temp:
                    print("param %s not found in state dict!" % k)

        self.load_state_dict(state_dict)
        print("Load checkpoint {} successfully!".format(model_path))


class NBINet(BaseModel):
    """轻量化三帧融合网络（面向 RK3588 NPU 部署）

    输入：3 帧 RAW Bayer（短1+短2 拼接 [B,8,H,W] + 长帧 [B,4,H,W]）
    输出：RGB 图像 [B,3,2H,2W]（含 demosaic 2x 上采样）
    """

    def __init__(self, opt):
        super(NBINet, self).__init__(opt)

        self.in_channel = opt.in_channel
        self.out_channel = opt.out_channel
        self.activ = opt.activ
        self.norm = opt.norm
        self.pad_type = opt.pad_type
        self.final_activ = opt.final_activ

        self.ngf = opt.ngf if hasattr(opt, 'ngf') else 10
        self.res_num = opt.res_num if hasattr(opt, 'res_num') else 2
        self.bottleneck_res_num = opt.bottleneck_res_num if hasattr(opt, 'bottleneck_res_num') else 4

        self.enc_ch = [self.ngf, self.ngf * 2, self.ngf * 4]
        self.bottleneck_ch = self.ngf * 4
        self.final_ps_ch = self.out_channel * 4

        self.build_layers()

    def build_layers(self):
        # 短帧编码器（输入：两个短帧拼接，8 通道）
        self.short_enc0_conv = Conv2dLayer(
            self.in_channel * 2, self.enc_ch[0], 3, stride=1, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.short_enc0_res = self._make_resblocks(self.enc_ch[0], self.res_num)

        self.short_enc1_conv = Conv2dLayer(
            self.enc_ch[0], self.enc_ch[1], 3, stride=2, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.short_enc1_res = self._make_resblocks(self.enc_ch[1], self.res_num)

        self.short_enc2_conv = Conv2dLayer(
            self.enc_ch[1], self.enc_ch[2], 3, stride=2, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.short_enc2_res = self._make_resblocks(self.enc_ch[2], self.res_num)

        self.short_enc3_conv = Conv2dLayer(
            self.enc_ch[2], self.bottleneck_ch, 3, stride=2, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.short_enc3_res = self._make_resblocks(self.bottleneck_ch, self.bottleneck_res_num)

        # 长帧编码器（输入：长曝光帧，4 通道）
        self.long_enc0_conv = Conv2dLayer(
            self.in_channel, self.enc_ch[0], 3, stride=1, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.long_enc0_res = self._make_resblocks(self.enc_ch[0], self.res_num)

        self.long_enc1_conv = Conv2dLayer(
            self.enc_ch[0], self.enc_ch[1], 3, stride=2, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.long_enc1_res = self._make_resblocks(self.enc_ch[1], self.res_num)

        self.long_enc2_conv = Conv2dLayer(
            self.enc_ch[1], self.enc_ch[2], 3, stride=2, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.long_enc2_res = self._make_resblocks(self.enc_ch[2], self.res_num)

        self.long_enc3_conv = Conv2dLayer(
            self.enc_ch[2], self.bottleneck_ch, 3, stride=2, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.long_enc3_res = self._make_resblocks(self.bottleneck_ch, self.bottleneck_res_num)

        # 瓶颈层融合（concat + 1x1 conv）
        self.fusion_conv = Conv2dLayer(
            self.bottleneck_ch * 2, self.bottleneck_ch, 1, stride=1, padding=0,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.fusion_res = self._make_resblocks(self.bottleneck_ch, self.bottleneck_res_num)

        # 解码器（PixelShuffle 上采样 + skip 连接）
        dec2_up_ch = self.bottleneck_ch // 4
        dec1_up_ch = self.enc_ch[1] // 4
        dec0_up_ch = self.final_ps_ch // 4

        self.dec2_up = nn.PixelShuffle(upscale_factor=2)
        self.dec2_compress = Conv2dLayer(
            dec2_up_ch + self.enc_ch[2] * 2, self.enc_ch[1], 3, stride=1, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.dec2_res = self._make_resblocks(self.enc_ch[1], self.res_num)

        self.dec1_up = nn.PixelShuffle(upscale_factor=2)
        self.dec1_compress = Conv2dLayer(
            dec1_up_ch + self.enc_ch[1] * 2, self.final_ps_ch, 3, stride=1, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.dec1_res = self._make_resblocks(self.final_ps_ch, self.res_num)

        self.dec0_up = nn.PixelShuffle(upscale_factor=2)
        self.dec0_compress = Conv2dLayer(
            dec0_up_ch + self.enc_ch[0] * 2, self.ngf, 3, stride=1, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.dec0_res = self._make_resblocks(self.ngf, self.res_num)

        # 输出层（2x 上采样到全分辨率 RGB）
        self.residual_conv = Conv2dLayer(
            self.ngf, self.final_ps_ch, 3, stride=1, padding=1,
            pad_type=self.pad_type, activation='none', norm='none')
        self.final_up = nn.PixelShuffle(upscale_factor=2)

        self.ref_conv = Conv2dLayer(
            self.in_channel, self.final_ps_ch, 1, stride=1, padding=0,
            pad_type=self.pad_type, activation='none', norm='none')
        self.ref_up = nn.PixelShuffle(upscale_factor=2)

        if self.final_activ == 'tanh':
            self.final_activation = nn.Tanh()

    def _make_resblocks(self, channels, num_blocks):
        blocks = nn.ModuleList()
        for _ in range(num_blocks):
            blocks.append(ResBlock(dim=channels, kernel_size=3, stride=1, padding=1,
                                   pad_type=self.pad_type, activation=self.activ, norm=self.norm))
        return blocks

    def _apply_resblocks(self, x, resblocks):
        for block in resblocks:
            x = block(x)
        return x

    def forward(self, short_cat, long_raw):
        """
        参数:
            short_cat  -- [B, 8, H, W] 两短帧 RGGB 拼接
            long_raw   -- [B, 4, H, W] 长曝光帧 RGGB
        返回:
            output     -- [B, 3, 2H, 2W] RGB
        """
        # 短帧编码器
        short_e0 = self._apply_resblocks(self.short_enc0_conv(short_cat), self.short_enc0_res)
        short_e1 = self._apply_resblocks(self.short_enc1_conv(short_e0), self.short_enc1_res)
        short_e2 = self._apply_resblocks(self.short_enc2_conv(short_e1), self.short_enc2_res)
        short_bottleneck = self._apply_resblocks(self.short_enc3_conv(short_e2), self.short_enc3_res)

        # 长帧编码器
        long_e0 = self._apply_resblocks(self.long_enc0_conv(long_raw), self.long_enc0_res)
        long_e1 = self._apply_resblocks(self.long_enc1_conv(long_e0), self.long_enc1_res)
        long_e2 = self._apply_resblocks(self.long_enc2_conv(long_e1), self.long_enc2_res)
        long_bottleneck = self._apply_resblocks(self.long_enc3_conv(long_e2), self.long_enc3_res)

        # 瓶颈层融合
        fused = torch.cat([short_bottleneck, long_bottleneck], dim=1)
        fused = self._apply_resblocks(self.fusion_conv(fused), self.fusion_res)

        # 解码器
        dec = self.dec2_up(fused)
        dec = self._apply_resblocks(self.dec2_compress(torch.cat([dec, short_e2, long_e2], dim=1)), self.dec2_res)

        dec = self.dec1_up(dec)
        dec = self._apply_resblocks(self.dec1_compress(torch.cat([dec, short_e1, long_e1], dim=1)), self.dec1_res)

        dec = self.dec0_up(dec)
        dec = self._apply_resblocks(self.dec0_compress(torch.cat([dec, short_e0, long_e0], dim=1)), self.dec0_res)

        # 2x 上采样残差输出
        residual = self.final_up(self.residual_conv(dec))
        ref = self.ref_up(self.ref_conv(long_raw))
        output = ref + residual

        if self.final_activ == 'tanh':
            output = self.final_activation(output)

        return output


if __name__ == "__main__":
    import argparse

    print("--- 测试 NBINet ---")
    opt_rk = argparse.Namespace(
        in_channel=4, out_channel=3, ngf=10,
        activ='relu', norm='none', pad_type='zero',
        final_activ='none', res_num=2, bottleneck_res_num=4
    )

    net_rk = NBINet(opt_rk)

    short_cat = torch.randn(1, 8, 256, 256)
    long_raw = torch.randn(1, 4, 256, 256)
    out_rk = net_rk(short_cat, long_raw)
    print("NBINet 输出维度:", out_rk.shape)
    assert out_rk.shape == (1, 3, 512, 512)

    total_params = sum(p.numel() for p in net_rk.parameters())
    print("NBINet 参数量: %.2fK" % (total_params / 1000))
