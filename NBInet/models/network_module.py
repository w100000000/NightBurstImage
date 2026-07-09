import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ----------------------------------------
#               Conv2d 层
# ----------------------------------------
class Conv2dLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0,
                 dilation = 1, pad_type = 'zero',
                 activation = 'lrelu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        self.layers = []

        # 初始化填充方式
        p = padding
        if p > 0:
            if pad_type == 'reflect':
                self.layers += [nn.ReflectionPad2d(padding)]
                p = 0
            elif pad_type == 'replicate':
                self.layers += [nn.ReplicationPad2d(padding)]
                p = 0
            elif pad_type == 'zero':
                pass
            else:
                raise NotImplementedError('padding type %s is not supported.' % pad_type)

        # 初始化卷积层
        if sn:
            self.layers += [SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = p, dilation = dilation))]
        else:
            self.layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = p, dilation = dilation)]

        # 初始化归一化类型
        if norm == 'bn':
            self.layers += [nn.BatchNorm2d(out_channels)]
        elif norm == 'in':
            self.layers += [nn.InstanceNorm2d(out_channels)]
        elif norm == 'ln':
            self.layers += [LayerNorm(out_channels)]
        elif norm == 'none':
            pass
        else:
            raise NotImplementedError('norm layer %s is not supported.' % norm)

        # 初始化激活函数
        if activation == 'relu':
            self.layers += [nn.ReLU(inplace = True)]
        elif activation == 'lrelu':
            self.layers += [nn.LeakyReLU(0.2, inplace = True)]
        elif activation == 'prelu':
            self.layers += [nn.PReLU()]
        elif activation == 'selu':
            self.layers += [nn.SELU(inplace = True)]
        elif activation == 'tanh':
            self.layers += [nn.Tanh()]
        elif activation == 'sigmoid':
            self.layers += [nn.Sigmoid()]
        elif activation == 'none':
            pass
        else:
            raise NotImplementedError('activation %s is not supported.' % activation)

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


# ----------------------------------------
#                ResBlock
# ----------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero',
                 activation = 'lrelu', norm = 'none', sn = False):
        super(ResBlock, self).__init__()
        self.conv2d = nn.Sequential(
            Conv2dLayer(dim, dim, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn),
            Conv2dLayer(dim, dim, kernel_size, stride, padding, dilation, pad_type, activation = 'none', norm = norm, sn = sn)
        )

    def forward(self, x):
        residual = x
        out = self.conv2d(x)
        out = out + residual
        return out


# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# ----------------------------------------
#           Spectral Norm Block
# ----------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name = 'weight', power_iterations = 1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
