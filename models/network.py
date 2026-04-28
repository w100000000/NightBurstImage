import torch            # PyTorch 基础库，提供张量(Tensor)操作、模型保存/加载等核心功能
import torch.nn as nn   # PyTorch 神经网络模块，提供 nn.Module（所有网络的基类）、各种层（卷积、归一化等）

from models.network_module import *   # 导入本项目自定义的网络组件：Conv2dLayer（卷积层）、ResBlock（残差块）、
                                       # DWT/IWT（离散小波变换/逆小波变换）、TransposeConv2dLayer（转置卷积层）、
                                       # PixelShuffleAlign（像素重排上采样层）等
from models import dcn_module          # 导入可变形卷积（DCN, Deformable Convolution Network）对齐模块，
                                       # 用于 DenoiseNet 中对短曝光和长曝光特征做空间对齐

from util.singleton import Singleton   # 单例模式工具类（实际在本文件中未被使用，属于残留导入）


# ========================================
#         网络权重初始化函数
# ========================================
# 作用：对网络中所有层的权重进行初始化，替代 PyTorch 默认的初始化方式
# 来源：这段代码是深度学习社区（尤其是 GAN / 图像复原领域）的通用模板，
#       最早流行自 pix2pix/CycleGAN 项目
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """初始化网络权重
    参数:
        net (network)        -- 要初始化的网络
        init_type (str)      -- 初始化方法的名称: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- normal、xavier、orthogonal 方法的缩放因子
    论文中使用的默认设置：均值为 0、标准差为 0.02 的高斯分布
    """
    def init_func(m):
        """
        这个内部函数会被 net.apply() 递归地应用到网络的每一个子模块上。
        参数 m 就是当前正在处理的子模块（比如某一个 Conv2d 层或 BatchNorm2d 层）
        """
        classname = m.__class__.__name__  # 获取当前模块的类名，比如 "Conv2d"、"BatchNorm2d"

        # 如果是卷积层（类名中包含 'Conv'），对权重进行初始化
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                # 正态分布初始化：均值 0，标准差 init_gain（默认 0.02）
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                # Xavier 初始化：根据输入/输出维度自动调整方差，适合 sigmoid/tanh 激活函数
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                # Kaiming 初始化：专为 ReLU/LeakyReLU 设计，保持前向传播方差稳定
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                # 正交初始化：权重矩阵为正交矩阵，有助于梯度稳定传播
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

        # 如果是 BatchNorm 层，权重初始化为均值 1.0、标准差 0.02 的正态分布，偏置设为 0
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # net.apply(init_func) 会递归遍历 net 的所有子模块，对每个子模块调用 init_func
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# ========================================
#       网络基类 BaseModel
# ========================================
# 作用：为所有网络提供公共功能（配置加载、权重加载等）
# 继承关系：nn.Module → BaseModel → DeblurNet_v2 / DenoiseNet_v2
class BaseModel(nn.Module):

    def __init__(self, opt):
        """
        构造函数
        参数:
            opt -- 包含所有超参数的配置对象（由 argparse 解析命令行参数得到）
        """
        # 必须调用父类 nn.Module 的构造函数，否则 PyTorch 的参数注册、.cuda()、
        # .parameters() 等核心功能全部无法工作
        super(BaseModel, self).__init__()
        # 保存配置对象，这样网络内部任何地方都能通过 self.opt.xxx 访问超参数
        self.opt = opt

    @staticmethod
    def load_config_from_json(file):
        """预留接口：从 JSON 文件加载网络配置（未实现）"""
        pass

    def dump_config_to_json(self, file):
        """预留接口：把网络配置导出为 JSON 文件（未实现）"""
        pass

    def load_ckpt(self, model_path, force_load = False):
        """
        加载预训练权重（checkpoint）
        参数:
            model_path (str)    -- 权重文件的路径（.pkl 或 .pth 文件）
            force_load (bool)   -- 是否使用容错模式加载
                False（默认）：严格模式，参数名和 shape 必须完全匹配，否则报错
                True：容错模式，能匹配的就加载，不匹配的跳过并打印警告
        """
        # 从文件加载权重字典（参数名 → 参数张量的映射）
        # map_location='cpu' 确保无论权重是在哪个 GPU 上保存的，都先加载到 CPU，避免 GPU 编号不一致报错
        state_dict = torch.load(model_path, map_location = torch.device('cpu'))

        if force_load:
            # === 容错加载模式 ===
            # 获取当前网络的参数字典（作为模板）
            state_dict_temp = self.state_dict()
            # 记录当前网络所有参数名，用于最后检查哪些参数在 checkpoint 中没找到
            key_temp = set(list(state_dict_temp.keys()))

            # 遍历 checkpoint 中的每个参数
            for n, p in state_dict.items():
                # 注释掉的代码是作者之前重命名层时用的临时代码，现在已不需要
                # if 'res_layer' in n:
                #     n = n.replace('res_layers.', 'res_block_')

                # 从待检查集合中移除这个参数名（说明 checkpoint 里有它）
                key_temp.remove(n)

                if n in state_dict_temp.keys():
                    # 当前网络中存在同名参数
                    if state_dict_temp[n].shape != p.data.shape:
                        # shape 不同（比如通道数变了），跳过这个参数
                        print('%s size mismatch, pass!' % n)
                        continue
                    # shape 相同，把 checkpoint 的值复制到当前网络
                    state_dict_temp[n].copy_(p.data)
                else:
                    # 当前网络中不存在这个参数名（可能是删掉了某些层）
                    print('%s not exist, pass!' % n)

            # 用填充好的模板替换原始 state_dict
            state_dict = state_dict_temp

            # 检查：当前网络有、但 checkpoint 中没有的参数（这些参数会保持随机初始化）
            if len(key_temp) != 0:
                for k in key_temp:
                    print("param %s not found in state dict!" % k)

        # 将权重加载到网络中
        self.load_state_dict(state_dict)
        print("Load checkpoint {} successfully!".format(model_path))


# ========================================
#          第一阶段：去模糊网络 DeblurNet_v2
# ========================================
# 功能：融合短曝光图（清晰但噪声大）和长曝光图（噪声小但有运动模糊），
#       输出一张去除运动模糊的图像
# 核心思路：在小波域（DWT 下采样后的频域空间）做去模糊处理，
#           然后用 IWT 恢复到图像域
# 网络结构：
#   输入 → 拼接 → DWT下采样 → 卷积 → DWT下采样 → 卷积融合
#   → 残差块组1 → IWT上采样 → 卷积 → IWT上采样
#   → 残差块组2 → 输出卷积 → 加上长曝光图（残差学习）→ 输出
class DeblurNet_v2(BaseModel):

    def __init__(self, opt):
        """
        构造函数：从配置中读取超参数，然后构建网络层
        """
        super(DeblurNet_v2, self).__init__(opt)

        self.in_channel = opt.in_channel        # 输入通道数（RGB 图像为 3）
        self.out_channel = opt.out_channel      # 输出通道数（RGB 图像为 3）
        self.activ = opt.activ                  # 激活函数类型（如 'lrelu' = LeakyReLU）
        self.norm = opt.norm                    # 归一化类型（如 'none', 'batch', 'instance'）
        self.pad_type = opt.pad_type            # 填充类型（如 'zero', 'reflect'）
        self.deblur_res_num = opt.deblur_res_num    # 第一组残差块的数量（在低分辨率小波域）
        self.deblur_res_num2 = opt.deblur_res_num2  # 第二组残差块的数量（在原始分辨率）
        self.final_activ = opt.final_activ      # 最终激活函数（如 'tanh' 将输出限制在 [-1,1]）

        # 基础通道数，默认 16，所有层的通道数都是 ngf 的倍数
        if hasattr(opt, 'ngf'):
            self.ngf = opt.ngf
        else:
            self.ngf = 16

        # 构建所有网络层
        self.build_layers()

    def build_upsample_layer(self, in_channel, out_channel, upsample_level = None):
        """
        上采样层的工厂方法（根据配置选择不同的上采样方式），将分辨率放大 2 倍
        注意：这个方法在当前代码中实际未被调用，是早期实验残留
        """
        if self.opt.upsample_layer == 'pixelshuffle':
            # 亚像素卷积：把通道维度的数据重排成空间维度
            # [B, C×4, H, W] → [B, C, H×2, W×2]
            return PixelShuffleAlign(upscale_factor = 2, mode = self.opt.shuffle_mode)
        elif self.opt.upsample_layer == 'bilinear':
            # 双线性插值放大 + 卷积细化
            return nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
                                 nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = 1))

    def build_layers(self):
        """构建去模糊网络的所有层"""

        # DWT（离散小波变换）：将图像分解为低频和高频子带，分辨率减半，通道数变为 4 倍
        # 相当于一种保留频率信息的下采样
        self.dwt = DWT()
        # IWT（逆小波变换）：DWT 的逆操作，分辨率加倍，通道数变为 1/4
        self.idwt = IWT()

        # 融合卷积层：
        # 输入通道 = in_channel(3) × 2(短+长拼接) × 4(第一次DWT) × 4(第二次DWT) = 96（当 in_channel=3）
        # 输出通道 = ngf × 4 × 4 = ngf × 16
        # 作用：将两次 DWT 下采样后的高维特征压缩到 ngf×16 个通道
        self.fusion_conv = Conv2dLayer(self.in_channel * 2 * 4 * 4,
                                       self.ngf * 4 * 4, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                       activation = self.activ, norm = self.norm)

        # 中间下采样卷积层：在两次 DWT 之间对特征做进一步处理
        # 输入输出通道都是 in_channel × 2 × 4 = 24（当 in_channel=3）
        self.downsample_conv = Conv2dLayer(self.in_channel * 2 * 4,
                                         self.in_channel * 2 * 4, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)

        # 第一组残差块：在低分辨率（原图的 1/4）小波域中做去模糊处理
        # 残差块能学习输入和输出之间的差异，有助于保持信息不丢失
        for i in range(self.deblur_res_num):
            in_channels = self.ngf * 4 * 4  # 通道数 = ngf × 16
            block = ResBlock(dim = in_channels,
                             kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type,
                             activation = self.activ, norm = self.norm)
            # 用 setattr 动态创建属性名，如 self.deblur_res_block_0, self.deblur_res_block_1, ...
            setattr(self, 'deblur_res_block_%d' % i, block)

        # 第一次 IWT 上采样后的卷积层
        # IWT 后通道数变为 ngf×16/4 = ngf×4，这里保持 ngf×4 不变
        self.upsample_conv = Conv2dLayer(self.ngf * 4,
                                         self.ngf * 4, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)

        # 最终输出卷积层：将 ngf 通道映射回 3 通道 RGB 图像
        # 不使用激活函数和归一化（activation='none', norm='none'），因为输出是残差
        self.deblur_layer = Conv2dLayer(self.ngf, 3, 3,
                                        stride = 1, padding = 1, pad_type = self.pad_type,
                                        activation = 'none', norm = 'none')

        # 第二组残差块：在原始分辨率上做精细化处理
        # IWT 两次上采样后回到原始分辨率，通道数为 ngf
        for i in range(self.deblur_res_num2):
            in_channels = self.ngf
            block = ResBlock(dim = in_channels,
                             kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type,
                             activation = self.activ, norm = self.norm)
            setattr(self, 'deblur_res_block2_%d' % i, block)

        # 如果配置了 tanh 激活，创建 Tanh 层（将输出限制在 [-1, 1] 范围）
        if self.opt.final_activ == 'tanh':
            self.final_activ = nn.Tanh()

    def forward(self, short_img, long_img):
        """
        前向传播（定义数据如何流过网络）
        参数:
            short_img -- 短曝光图像 [B, 3, H, W]，清晰但噪声大
            long_img  -- 长曝光图像 [B, 3, H, W]，噪声小但有运动模糊
        返回:
            deblur_out -- 去模糊后的图像 [B, 3, H, W]
        """

        # Step 1: 将短曝光和长曝光图像在通道维度拼接
        # [B, 3, H, W] + [B, 3, H, W] → [B, 6, H, W]
        sl = torch.cat([short_img, long_img], dim = 1)

        # Step 2: 第一次 DWT 下采样
        # [B, 6, H, W] → [B, 24, H/2, W/2]（通道×4，分辨率÷2）
        sl = self.dwt(sl)

        # Step 3: 中间卷积处理
        sl = self.downsample_conv(sl)

        # Step 4: 第二次 DWT 下采样
        # [B, 24, H/2, W/2] → [B, 96, H/4, W/4]
        sl = self.dwt(sl)

        # Step 5: 融合卷积，降低通道数
        # [B, 96, H/4, W/4] → [B, ngf×16, H/4, W/4]
        sl = self.fusion_conv(sl)

        # Step 6: 第一组残差块处理（在低分辨率小波域做去模糊的核心计算）
        deblur_sl = sl  # 保存输入用于残差连接
        for i in range(self.deblur_res_num):
            resblock = getattr(self, 'deblur_res_block_%d' % i)  # 获取第 i 个残差块
            deblur_sl = resblock(deblur_sl)
        # 残差连接：将残差块的输出加回输入（这样网络只需学习"差异"而非完整映射）
        sl = sl + deblur_sl

        # Step 7: 第一次 IWT 上采样
        # [B, ngf×16, H/4, W/4] → [B, ngf×4, H/2, W/2]
        sl = self.idwt(sl)
        sl = self.upsample_conv(sl)

        # Step 8: 第二次 IWT 上采样，回到原始分辨率
        # [B, ngf×4, H/2, W/2] → [B, ngf, H, W]
        sl = self.idwt(sl)

        # Step 9: 第二组残差块处理（在原始分辨率做精细化）
        deblur_sl = sl
        for i in range(self.deblur_res_num2):
            resblock = getattr(self, 'deblur_res_block2_%d' % i)
            deblur_sl = resblock(deblur_sl)
        sl = deblur_sl + sl  # 残差连接

        # Step 10: 输出卷积，将 ngf 通道映射回 3 通道
        # [B, ngf, H, W] → [B, 3, H, W]（这里输出的是残差/修正量）
        deblur_sl = self.deblur_layer(sl)

        # Step 11: 残差学习 —— 网络学习的是"修正量"，加到长曝光图上得到最终结果
        # 这比直接让网络预测完整图像更容易训练
        deblur_out = long_img + deblur_sl

        # 可选：用 tanh 将输出限制在 [-1, 1]
        if self.opt.final_activ == 'tanh':
            deblur_out = self.final_activ(deblur_out)

        return deblur_out


# ========================================
#        第二阶段：去噪网络 DenoiseNet_v2
# ========================================
# 功能：在第一阶段去模糊结果的基础上，进一步去除噪声
# 输入：短曝光图 + 长曝光图 + 第一阶段的去模糊输出
# 核心结构：U-Net 编码器-解码器 + DCN（可变形卷积）特征对齐
#
# 网络结构示意图（U-Net with skip connections）：
#
#   短曝光编码器                长曝光编码器               解码器
#   short_fea1 (1x)  --------  long_fea1 (1x)  ----→ DCN对齐 → 拼接 → level 1 输出
#       ↓                          ↓                      ↑
#   short_fea2 (1/2) --------  long_fea2 (1/2) ----→ DCN对齐 → 拼接 → level 2
#       ↓                          ↓                      ↑
#   short_fea3 (1/4) --------  long_fea3 (1/4) ----→ DCN对齐 → 拼接 → level 3
#       ↓                          ↓                      ↑
#   short_fea4 (1/8) --------  long_fea4 (1/8) ----→ DCN对齐 → 拼接 → level 4
#       ↓                          ↓                      ↑
#   short_fea5 (1/16) -------  long_fea5 (1/16) ---→ DCN对齐 → 拼接 → level 5（瓶颈层）
#
# DCN 对齐的作用：短曝光和长曝光之间可能存在位移（手抖、物体运动），
#               可变形卷积能学习不规则的采样偏移量(offset)，将短曝光特征对齐到长曝光特征
#               offset 从底层向高层级联传递，实现从粗到细的对齐
class DenoiseNet_v2(BaseModel):

    def __init__(self, opt):
        """
        构造函数：从配置中读取超参数，然后构建网络层
        """
        super(DenoiseNet_v2, self).__init__(opt)

        self.in_channel = opt.in_channel            # 输入通道数（RGB = 3）
        self.out_channel = opt.out_channel          # 输出通道数（RGB = 3）
        self.activ = opt.activ                      # 激活函数类型
        self.norm = opt.norm                        # 归一化类型
        self.pad_type = opt.pad_type                # 填充类型
        self.denoise_res_num = opt.denoise_res_num      # 去噪残差块数量（编码器中，当前版本未使用）
        self.denoise_res_num2 = opt.denoise_res_num2    # 去噪残差块数量（解码器后的精细化）
        self.final_activ = opt.final_activ          # 最终激活函数
        self.groups = opt.groups                    # DCN 可变形卷积的分组数

        if hasattr(opt, 'ngf'):
            self.ngf = opt.ngf
        else:
            self.ngf = 16

        self.build_layers()

    def build_layers(self):
        """构建去噪网络的所有层"""

        # ========================
        # 短曝光图编码器（5 级下采样）
        # ========================
        # 输入是短曝光图 + 去模糊输出的拼接（通道数 = in_channel × 2 = 6）
        # 每一级：卷积降维/降分辨率 + 残差块提取特征

        # Level 1: [B, 6, H, W] → [B, ngf, H, W]（保持分辨率，只改变通道数）
        self.downsample_short_conv1 = nn.Sequential(
            Conv2dLayer(self.in_channel * 2, self.ngf, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # Level 2: [B, ngf, H, W] → [B, ngf×2, H/2, W/2]（stride=2 使分辨率减半）
        self.downsample_short_conv2 = nn.Sequential(
            Conv2dLayer(self.ngf, self.ngf * 2, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 2, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # Level 3: [B, ngf×2, H/2, W/2] → [B, ngf×4, H/4, W/4]
        self.downsample_short_conv3 = nn.Sequential(
            Conv2dLayer(self.ngf * 2, self.ngf * 4, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 4, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # Level 4: [B, ngf×4, H/4, W/4] → [B, ngf×8, H/8, W/8]
        self.downsample_short_conv4 = nn.Sequential(
            Conv2dLayer(self.ngf * 4, self.ngf * 8, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 8, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # Level 5: [B, ngf×8, H/8, W/8] → [B, ngf×16, H/16, W/16]（最低分辨率）
        self.downsample_short_conv5 = nn.Sequential(
            Conv2dLayer(self.ngf * 8, self.ngf * 16, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 16, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # ========================
        # 长曝光图编码器（5 级下采样，结构与短曝光编码器类似）
        # ========================
        # 输入是长曝光图（通道数 = in_channel = 3）

        # Level 1: [B, 3, H, W] → [B, ngf, H, W]
        self.downsample_long_conv1 = nn.Sequential(
            Conv2dLayer(self.in_channel, self.ngf, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # Level 2: → [B, ngf×2, H/2, W/2]
        self.downsample_long_conv2 = nn.Sequential(
            Conv2dLayer(self.ngf, self.ngf * 2, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 2, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # Level 3: → [B, ngf×4, H/4, W/4]
        self.downsample_long_conv3 = nn.Sequential(
            Conv2dLayer(self.ngf * 2, self.ngf * 4, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 4, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # Level 4: → [B, ngf×8, H/8, W/8]
        self.downsample_long_conv4 = nn.Sequential(
            Conv2dLayer(self.ngf * 4, self.ngf * 8, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 8, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # Level 5: → [B, ngf×16, H/16, W/16]
        self.downsample_long_conv5 = nn.Sequential(
            Conv2dLayer(self.ngf * 8, self.ngf * 16, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 16, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # ========================
        # 解码器（自底向上逐级上采样 + DCN 对齐 + 跳跃连接）
        # ========================
        # 每一级的处理流程：
        #   1. DCN 对齐：用可变形卷积将短曝光特征对齐到长曝光特征
        #   2. 拼接：对齐后的特征与原始短曝光特征拼接
        #   3. 卷积压缩通道
        #   4. 残差块精细处理
        #   5. 转置卷积上采样到下一级分辨率

        # --- 解码器 Level 5（瓶颈层，分辨率 1/16）---
        # DCN 对齐模块：学习短曝光→长曝光的空间偏移量，对短曝光特征做对齐变形
        self.upsample_alignblock5 = dcn_module.Align_module(self.ngf * 16, self.groups)
        # 对齐后的特征与原始短曝光特征拼接后，通道数变为 ngf×32，用卷积压缩回 ngf×16
        self.upsample_comb5 = Conv2dLayer(self.ngf * 32, self.ngf * 16, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        # 4 个残差块做特征精细化
        self.upsample_resblock5 = nn.Sequential(
            ResBlock(dim = self.ngf * 16, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 16, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 16, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 16, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # --- 解码器 Level 4（分辨率 1/8）---
        # 转置卷积上采样：[B, ngf×16, H/16, W/16] → [B, ngf×8, H/8, W/8]
        self.upsample_conv4 = TransposeConv2dLayer(self.ngf * 16,
                                         self.ngf * 8, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)
        # DCN 对齐（接收上一级的 offset，实现从粗到细的级联对齐）
        self.upsample_alignblock4 = dcn_module.Align_module(self.ngf * 8, self.groups)
        self.upsample_comb4 = Conv2dLayer(self.ngf * 16, self.ngf * 8, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        self.upsample_resblock4 = nn.Sequential(
            ResBlock(dim = self.ngf * 8, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 8, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 8, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 8, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # --- 解码器 Level 3（分辨率 1/4）---
        # 输入通道 = ngf×8（上采样结果）+ ngf×8（level 4 的 skip connection）= ngf×16
        self.upsample_conv3 = TransposeConv2dLayer(self.ngf * 8 + self.ngf * 8,
                                         self.ngf * 4, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)
        self.upsample_alignblock3 = dcn_module.Align_module(self.ngf * 4, self.groups)
        self.upsample_comb3 = Conv2dLayer(self.ngf * 8, self.ngf * 4, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        self.upsample_resblock3 = nn.Sequential(
            ResBlock(dim = self.ngf * 4, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 4, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 4, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 4, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # --- 解码器 Level 2（分辨率 1/2）---
        self.upsample_conv2 = TransposeConv2dLayer(self.ngf * 4 + self.ngf * 4,
                                         self.ngf * 2, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)
        self.upsample_alignblock2 = dcn_module.Align_module(self.ngf * 2, self.groups)
        self.upsample_comb2 = Conv2dLayer(self.ngf * 4, self.ngf * 2, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        self.upsample_resblock2 = nn.Sequential(
            ResBlock(dim = self.ngf * 2, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 2, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 2, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 2, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # --- 解码器 Level 1（原始分辨率）---
        self.upsample_conv1 = TransposeConv2dLayer(self.ngf * 2 + self.ngf * 2,
                                         self.ngf, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)
        self.upsample_alignblock1 = dcn_module.Align_module(self.ngf, self.groups)
        self.upsample_comb1 = Conv2dLayer(self.ngf * 2, self.ngf, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        # 最终拼接后的卷积：将 skip connection 和上采样结果合并
        self.upsample_conv0 = Conv2dLayer(self.ngf + self.ngf,
                                         self.ngf, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)

        # ========================
        # 辅助输入卷积层
        # ========================
        # 将原始图像（3 通道）映射到特征空间（ngf 通道），用于最终的特征融合
        self.short_conv = Conv2dLayer(self.in_channel, self.ngf, 3,
                                      stride = 1, padding = 1, pad_type = self.pad_type,
                                      activation = self.activ, norm = self.norm)
        self.long_conv = Conv2dLayer(self.in_channel, self.ngf, 3,
                                      stride = 1, padding = 1, pad_type = self.pad_type,
                                      activation = self.activ, norm = self.norm)
        self.deblur_out_conv = Conv2dLayer(self.in_channel, self.ngf, 3,
                                      stride = 1, padding = 1, pad_type = self.pad_type,
                                      activation = self.activ, norm = self.norm)

        # 最终输出卷积：将 ngf 通道映射回 3 通道 RGB（无激活无归一化，输出残差）
        self.denoise_layer = Conv2dLayer(self.ngf, 3, 3,
                                        stride = 1, padding = 1, pad_type = self.pad_type,
                                        activation = 'none', norm = 'none')

        # 精细化残差块：在原始分辨率上做最终的去噪精调
        for i in range(self.denoise_res_num2):
            in_channels = self.ngf
            block = ResBlock(dim = in_channels,
                             kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type,
                             activation = self.activ, norm = self.norm)
            setattr(self, 'denoise_res_block2_%d' % i, block)

        if self.opt.final_activ == 'tanh':
            self.final_activ = nn.Tanh()

    def forward(self, short_img, long_img, deblur_out):
        """
        前向传播
        参数:
            short_img  -- 短曝光图像 [B, 3, H, W]
            long_img   -- 长曝光图像 [B, 3, H, W]
            deblur_out -- 第一阶段去模糊网络的输出 [B, 3, H, W]
        返回:
            denoise_out -- 最终去噪后的清晰图像 [B, 3, H, W]
        """

        # ========================
        # 编码器：提取多尺度特征
        # ========================

        # 短曝光编码器（输入 = 短曝光图 + 去模糊输出 的拼接）
        short_fea1 = torch.cat((short_img, deblur_out), 1)         # [B, 6, H, W]
        short_fea1 = self.downsample_short_conv1(short_fea1)       # [B, ngf, H, W]
        short_fea2 = self.downsample_short_conv2(short_fea1)       # [B, ngf×2, H/2, W/2]
        short_fea3 = self.downsample_short_conv3(short_fea2)       # [B, ngf×4, H/4, W/4]
        short_fea4 = self.downsample_short_conv4(short_fea3)       # [B, ngf×8, H/8, W/8]
        short_fea5 = self.downsample_short_conv5(short_fea4)       # [B, ngf×16, H/16, W/16]

        # 长曝光编码器
        long_fea1 = self.downsample_long_conv1(long_img)           # [B, ngf, H, W]
        long_fea2 = self.downsample_long_conv2(long_fea1)          # [B, ngf×2, H/2, W/2]
        long_fea3 = self.downsample_long_conv3(long_fea2)          # [B, ngf×4, H/4, W/4]
        long_fea4 = self.downsample_long_conv4(long_fea3)          # [B, ngf×8, H/8, W/8]
        long_fea5 = self.downsample_long_conv5(long_fea4)          # [B, ngf×16, H/16, W/16]

        # ========================
        # 解码器：自底向上逐级上采样 + DCN 对齐
        # ========================

        # --- Level 5（瓶颈层，分辨率 1/16）---
        # DCN 对齐：将 short_fea5 对齐到 long_fea5，同时输出 offset（偏移量）
        # offset 会传递给上一层，实现从粗到细的级联对齐
        short_fea, offset_5 = self.upsample_alignblock5(short_fea5, long_fea5)
        # 对齐后的特征与原始短曝光特征拼接（保留未对齐的信息作为补充）
        short_fea = torch.cat((short_fea, short_fea5), 1)           # [B, ngf×32, H/16, W/16]
        short_fea = self.upsample_comb5(short_fea)                  # 压缩回 [B, ngf×16, H/16, W/16]
        short_fea = self.upsample_resblock5(short_fea)              # 4 个残差块精细化

        # --- Level 4（分辨率 1/8）---
        short_fea = self.upsample_conv4(short_fea)                  # 上采样到 [B, ngf×8, H/8, W/8]
        # DCN 对齐，传入上一级的 offset_5 作为初始偏移量参考
        short_cut4, offset_4 = self.upsample_alignblock4(short_fea4, long_fea4, offset_5)
        short_cut4 = torch.cat((short_cut4, short_fea4), 1)         # 拼接
        short_cut4 = self.upsample_comb4(short_cut4)                # 压缩通道
        short_cut4 = self.upsample_resblock4(short_cut4)            # 残差块
        # 跳跃连接：将当前级的对齐结果与上采样结果拼接
        short_fea = torch.cat((short_fea, short_cut4), 1)

        # --- Level 3（分辨率 1/4）---
        short_fea = self.upsample_conv3(short_fea)                  # 上采样到 1/4 分辨率
        short_cut3, offset_3 = self.upsample_alignblock3(short_fea3, long_fea3, offset_4)
        short_cut3 = torch.cat((short_cut3, short_fea3), 1)
        short_cut3 = self.upsample_comb3(short_cut3)
        short_cut3 = self.upsample_resblock3(short_cut3)
        short_fea = torch.cat((short_fea, short_cut3), 1)

        # --- Level 2（分辨率 1/2）---
        short_fea = self.upsample_conv2(short_fea)                  # 上采样到 1/2 分辨率
        short_cut2, offset_2 = self.upsample_alignblock2(short_fea2, long_fea2, offset_3)
        short_cut2 = torch.cat((short_cut2, short_fea2), 1)
        short_cut2 = self.upsample_comb2(short_cut2)
        short_cut2 = self.upsample_resblock2(short_cut2)
        short_fea = torch.cat((short_fea, short_cut2), 1)

        # --- Level 1（原始分辨率）---
        short_fea = self.upsample_conv1(short_fea)                  # 上采样到原始分辨率
        short_cut1, offset_1 = self.upsample_alignblock1(short_fea1, long_fea1, offset_2)
        short_cut1 = torch.cat((short_cut1, short_fea1), 1)
        short_cut1 = self.upsample_comb1(short_cut1)
        short_fea = torch.cat((short_fea, short_cut1), 1)
        short_fea = self.upsample_conv0(short_fea)                  # 最终融合卷积 → [B, ngf, H, W]

        # ========================
        # 精细化 + 输出
        # ========================

        # 残差块组做最终精调
        sl = short_fea
        for i in range(self.denoise_res_num2):
            resblock = getattr(self, 'denoise_res_block2_%d' % i)
            sl = resblock(sl)
        short_fea = sl + short_fea  # 残差连接

        # 输出卷积：[B, ngf, H, W] → [B, 3, H, W]（输出残差/修正量）
        short_fea = self.denoise_layer(short_fea)

        # 残差学习：最终输出 = 去模糊结果 + 去噪修正量
        # 网络只需要学习"还需要修正多少"，而不是从零预测完整图像
        denoise_out = deblur_out + short_fea

        if self.opt.final_activ == 'tanh':
            denoise_out = self.final_activ(denoise_out)

        return denoise_out


# ========================================
#    统一轻量化网络 D2HNet_RK（面向 RK3588 NPU 部署）
# ========================================
# 设计目标：
#   - 输入：3 帧 RAW Bayer（短-长-短），4 通道 RGGB 格式
#   - 输出：1 帧 RGB 图像（3 通道）
#   - 所有算子 RKNN 兼容（无 DWT/IWT/DCN）
#   - 轻量化（ngf=10，3 级 U-Net，~77K 参数）
#
# 算子替换说明：
#   DWT（小波下采样） → Conv2d(stride=2)（可学习，RKNN 支持）
#   IWT（小波上采样） → nn.PixelShuffle(2)（RKNN DepthToSpace）
#   DCN（可变形卷积） → concat + Conv2d（1/8 分辨率下对齐误差已很小）
class D2HNet_RK(BaseModel):

    def __init__(self, opt):
        super(D2HNet_RK, self).__init__(opt)

        self.in_channel = opt.in_channel      # 每帧 RAW 通道数（RGGB = 4）
        self.out_channel = opt.out_channel    # 输出通道数（RGB = 3）
        self.activ = opt.activ                # 激活函数（默认 'lrelu'）
        self.norm = opt.norm                  # 归一化类型（默认 'none'）
        self.pad_type = opt.pad_type          # 填充类型（默认 'zero'）
        self.final_activ = opt.final_activ    # 最终激活函数

        # 基础通道数，默认 10（轻量化设计）
        self.ngf = opt.ngf if hasattr(opt, 'ngf') else 10
        # 编码器每级的残差块数量
        self.res_num = opt.res_num if hasattr(opt, 'res_num') else 2
        # 瓶颈层残差块数量（瓶颈层计算量小，可以多放几层）
        self.bottleneck_res_num = opt.bottleneck_res_num if hasattr(opt, 'bottleneck_res_num') else 4

        # 编码器各级通道数：[ngf, ngf*2, ngf*4]
        # 瓶颈层通道数：ngf*4
        self.enc_ch = [self.ngf, self.ngf * 2, self.ngf * 4]
        self.bottleneck_ch = self.ngf * 4

        self.build_layers()

    def build_layers(self):
        """
        构建所有网络层

        结构总览：
        短帧编码器（4 级）→ 瓶颈融合 → 解码器（3 级）→ 输出
        长帧编码器（4 级）↗
        """

        # ========================
        # 短帧编码器（输入：两个短帧拼接，8 通道）
        # ========================
        # Level 0：保持分辨率，只改变通道数
        self.short_enc0_conv = Conv2dLayer(
            self.in_channel * 2, self.enc_ch[0], 3, stride=1, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.short_enc0_res = self._make_resblocks(self.enc_ch[0], self.res_num)

        # Level 1：stride=2 下采样，分辨率减半（替代 DWT）
        self.short_enc1_conv = Conv2dLayer(
            self.enc_ch[0], self.enc_ch[1], 3, stride=2, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.short_enc1_res = self._make_resblocks(self.enc_ch[1], self.res_num)

        # Level 2：stride=2 下采样，分辨率 1/4
        self.short_enc2_conv = Conv2dLayer(
            self.enc_ch[1], self.enc_ch[2], 3, stride=2, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.short_enc2_res = self._make_resblocks(self.enc_ch[2], self.res_num)

        # Level 3（瓶颈）：stride=2 下采样，分辨率 1/8
        self.short_enc3_conv = Conv2dLayer(
            self.enc_ch[2], self.bottleneck_ch, 3, stride=2, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.short_enc3_res = self._make_resblocks(self.bottleneck_ch, self.bottleneck_res_num)

        # ========================
        # 长帧编码器（输入：长曝光帧，4 通道）
        # ========================
        # 结构与短帧编码器完全相同，只是输入通道不同（4 而非 8）
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

        # ========================
        # 瓶颈层融合（替代 DCN 对齐模块）
        # ========================
        # 在 1/8 分辨率下，短帧和长帧的空间对齐误差已经很小
        # 用 concat + 1×1 卷积实现通道级特征选择
        self.fusion_conv = Conv2dLayer(
            self.bottleneck_ch * 2, self.bottleneck_ch, 1, stride=1, padding=0,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.fusion_res = self._make_resblocks(self.bottleneck_ch, self.bottleneck_res_num)

        # ========================
        # 解码器（PixelShuffle 上采样 + skip 连接）
        # ========================
        # 注意：PixelShuffle(2) 要求输入通道数是 4 的倍数
        # 解码器每级：PixelShuffle 上采样 → skip 拼接 → 通道压缩 → 残差块

        # Level 2 解码：[B,40,135,240] → [B,10,270,480]
        # cat(short_skip2, long_skip2) → [B,10+40+40=90,270,480] → 压缩到 20
        self.dec2_up = nn.PixelShuffle(upscale_factor=2)  # 40 → 10
        self.dec2_compress = Conv2dLayer(
            10 + self.enc_ch[2] * 2, self.enc_ch[1], 3, stride=1, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.dec2_res = self._make_resblocks(self.enc_ch[1], self.res_num)

        # Level 1 解码：[B,20,270,480] → [B,5,540,960]
        # cat(short_skip1, long_skip1) → [B,5+20+20=45,540,960] → 压缩到 12
        # 注意：输出 12 通道是为了下一级 PixelShuffle 的通道对齐（12/4=3）
        self.dec1_up = nn.PixelShuffle(upscale_factor=2)  # 20 → 5
        self.dec1_compress = Conv2dLayer(
            5 + self.enc_ch[1] * 2, 12, 3, stride=1, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.dec1_res = self._make_resblocks(12, self.res_num)

        # Level 0 解码：[B,12,540,960] → [B,3,1080,1920]
        # cat(short_skip0, long_skip0) → [B,3+10+10=23,1080,1920] → 压缩到 10
        self.dec0_up = nn.PixelShuffle(upscale_factor=2)  # 12 → 3
        self.dec0_compress = Conv2dLayer(
            3 + self.enc_ch[0] * 2, self.ngf, 3, stride=1, padding=1,
            pad_type=self.pad_type, activation=self.activ, norm=self.norm)
        self.dec0_res = self._make_resblocks(self.ngf, self.res_num)

        # ========================
        # 输出层
        # ========================
        # ref_conv：将长曝光 RAW（4ch）转为 RGB 参考图（3ch），作为残差学习的基准
        self.ref_conv = Conv2dLayer(
            self.in_channel, self.out_channel, 1, stride=1, padding=0,
            pad_type=self.pad_type, activation='none', norm='none')

        # residual_conv：将解码器输出（ngf 通道）映射为 3 通道残差修正量
        self.residual_conv = Conv2dLayer(
            self.ngf, self.out_channel, 3, stride=1, padding=1,
            pad_type=self.pad_type, activation='none', norm='none')

        # 最终激活函数
        if self.final_activ == 'tanh':
            self.final_activation = nn.Tanh()

    def _make_resblocks(self, channels, num_blocks):
        """创建指定数量的残差块，包装为 nn.ModuleList"""
        blocks = nn.ModuleList()
        for _ in range(num_blocks):
            blocks.append(ResBlock(dim=channels, kernel_size=3, stride=1, padding=1,
                                   pad_type=self.pad_type, activation=self.activ, norm=self.norm))
        return blocks

    def _apply_resblocks(self, x, resblocks):
        """依次通过残差块组"""
        for block in resblocks:
            x = block(x)
        return x

    def forward(self, short1_raw, long_raw, short2_raw):
        """
        前向传播

        参数:
            short1_raw -- 短曝光帧1（头帧） [B, 4, H, W]（RGGB RAW）
            long_raw   -- 长曝光帧          [B, 4, H, W]（RGGB RAW）
            short2_raw -- 短曝光帧2（尾帧） [B, 4, H, W]（RGGB RAW）
        返回:
            output     -- 复原的 RGB 图像    [B, 3, H, W]
        """

        # Step 1: 拼接两个短曝光帧
        short_cat = torch.cat([short1_raw, short2_raw], dim=1)  # [B, 8, H, W]

        # Step 2: 短帧编码器（提取多尺度特征，保存 skip 连接）
        short_e0 = self.short_enc0_conv(short_cat)                    # [B, 10, H, W]
        short_e0 = self._apply_resblocks(short_e0, self.short_enc0_res)

        short_e1 = self.short_enc1_conv(short_e0)                     # [B, 20, H/2, W/2]
        short_e1 = self._apply_resblocks(short_e1, self.short_enc1_res)

        short_e2 = self.short_enc2_conv(short_e1)                     # [B, 40, H/4, W/4]
        short_e2 = self._apply_resblocks(short_e2, self.short_enc2_res)

        short_bottleneck = self.short_enc3_conv(short_e2)             # [B, 40, H/8, W/8]
        short_bottleneck = self._apply_resblocks(short_bottleneck, self.short_enc3_res)

        # Step 3: 长帧编码器
        long_e0 = self.long_enc0_conv(long_raw)                       # [B, 10, H, W]
        long_e0 = self._apply_resblocks(long_e0, self.long_enc0_res)

        long_e1 = self.long_enc1_conv(long_e0)                        # [B, 20, H/2, W/2]
        long_e1 = self._apply_resblocks(long_e1, self.long_enc1_res)

        long_e2 = self.long_enc2_conv(long_e1)                        # [B, 40, H/4, W/4]
        long_e2 = self._apply_resblocks(long_e2, self.long_enc2_res)

        long_bottleneck = self.long_enc3_conv(long_e2)                # [B, 40, H/8, W/8]
        long_bottleneck = self._apply_resblocks(long_bottleneck, self.long_enc3_res)

        # Step 4: 瓶颈层融合（concat + 1×1 conv 替代 DCN 对齐）
        fused = torch.cat([short_bottleneck, long_bottleneck], dim=1)  # [B, 80, H/8, W/8]
        fused = self.fusion_conv(fused)                                # [B, 40, H/8, W/8]
        fused = self._apply_resblocks(fused, self.fusion_res)

        # Step 5: 解码器（PixelShuffle 上采样 + skip 连接）

        # Level 2: 1/8 → 1/4 分辨率
        dec = self.dec2_up(fused)                                      # [B, 10, H/4, W/4]
        dec = torch.cat([dec, short_e2, long_e2], dim=1)              # [B, 90, H/4, W/4]
        dec = self.dec2_compress(dec)                                  # [B, 20, H/4, W/4]
        dec = self._apply_resblocks(dec, self.dec2_res)

        # Level 1: 1/4 → 1/2 分辨率
        dec = self.dec1_up(dec)                                        # [B, 5, H/2, W/2]
        dec = torch.cat([dec, short_e1, long_e1], dim=1)              # [B, 45, H/2, W/2]
        dec = self.dec1_compress(dec)                                  # [B, 12, H/2, W/2]
        dec = self._apply_resblocks(dec, self.dec1_res)

        # Level 0: 1/2 → 原始分辨率
        dec = self.dec0_up(dec)                                        # [B, 3, H, W]
        dec = torch.cat([dec, short_e0, long_e0], dim=1)              # [B, 23, H, W]
        dec = self.dec0_compress(dec)                                  # [B, 10, H, W]
        dec = self._apply_resblocks(dec, self.dec0_res)

        # Step 6: 残差学习输出
        residual = self.residual_conv(dec)                             # [B, 3, H, W]
        ref = self.ref_conv(long_raw)                                  # [B, 3, H, W]
        output = ref + residual

        if self.final_activ == 'tanh':
            output = self.final_activation(output)

        return output


# ========================================
#              测试代码
# ========================================
# 当直接运行这个文件时（python network.py），会执行以下测试
# 用于验证网络结构是否正确、输入输出维度是否匹配
if __name__ == "__main__":

    import argparse

    # 创建命令行参数解析器，定义所有超参数的默认值
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channel', type = int, default = 3, help = '输入通道数（RGB=3）')
    parser.add_argument('--out_channel', type = int, default = 3, help = '输出通道数（RGB=3）')
    parser.add_argument('--ngf', type = int, default = 64, help = '基础特征通道数')
    parser.add_argument('--ngf2', type = int, default = 8, help = '第二网络的基础通道数')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = '激活函数类型')
    parser.add_argument('--norm', type = str, default = 'none', help = '归一化类型')
    parser.add_argument('--pad', type = str, default = 'zero', help = '填充类型')

    parser.add_argument('--deblur_res_num', type = int, default = 8, help = '去模糊残差块数量（低分辨率）')
    parser.add_argument('--deblur_res_num2', type = int, default = 4, help = '去模糊残差块数量（原始分辨率）')
    parser.add_argument('--denoise_res_num', type = int, default = 8, help = '去噪残差块数量')
    parser.add_argument('--denoise_res_num2', type = int, default = 4, help = '去噪精细化残差块数量')
    parser.add_argument('--groups', type = int, default = 8, help = 'DCN 可变形卷积的分组数')

    parser.add_argument('--final_activ', type = str, default = 'none', help = '最终激活函数')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = '填充类型')
    parser.add_argument('--upsample_layer', type = str, default = 'pixelshuffle', help = '上采样方式')
    parser.add_argument('--shuffle_mode', type = str, default = 'caffe', help = 'PixelShuffle 模式')

    opt = parser.parse_args()

    # 创建一个随机的 256×256 RGB 图像作为测试输入
    a = torch.randn(1, 3, 256, 256).cuda()  # [batch=1, channel=3, height=256, width=256]

    # 测试去模糊网络：输入两张图（短曝光、长曝光），输出一张去模糊图
    net = DeblurNet_v2(opt).cuda()
    out = net(a, a)  # 这里用同一张图作为短曝光和长曝光（仅测试维度是否正确）

    # 测试去噪网络（被注释掉了）：输入三张图（短曝光、长曝光、去模糊输出）
    #net = DenoiseNet_v2(opt).cuda()
    #out = net(a, a, a)

    # 打印输出维度，验证是否为 [1, 3, 256, 256]
    print(out.shape)

    # ========================
    # 测试 D2HNet_RK（轻量化网络）
    # ========================
    print("\n--- 测试 D2HNet_RK ---")
    opt_rk = argparse.Namespace(
        in_channel=4,         # RAW RGGB 4 通道
        out_channel=3,        # RGB 输出
        ngf=10,               # 基础通道数
        activ='lrelu',
        norm='none',
        pad_type='zero',
        final_activ='none',
        res_num=2,            # 每级残差块数
        bottleneck_res_num=4  # 瓶颈层残差块数
    )

    net_rk = D2HNet_RK(opt_rk)
    # 测试输入：3 帧 RAW（短-长-短），256×256
    raw_input = torch.randn(1, 4, 256, 256)
    out_rk = net_rk(raw_input, raw_input, raw_input)
    print("D2HNet_RK 输出维度:", out_rk.shape)  # 期望: [1, 3, 256, 256]

    # 统计参数量
    total_params = sum(p.numel() for p in net_rk.parameters())
    print("D2HNet_RK 参数量: %.2fK" % (total_params / 1000))
