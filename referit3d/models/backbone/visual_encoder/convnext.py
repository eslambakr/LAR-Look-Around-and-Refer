# Should replace all sequential to work with JIT:
# https://discuss.pytorch.org/t/how-to-add-annotation-to-work-with-jit-for-nn-sequential/95683
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torch import Tensor
import torch.jit as jit


class Block(jit.ScriptModule):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm_CL(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    @jit.script_method
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(jit.ScriptModule):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers_dot_0_dot_0 = nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4)
        self.downsample_layers_dot_0_dot_1 = LayerNorm_CF(dims[0], eps=1e-6)
        self.downsample_layers_dot_1_dot_0 = LayerNorm_CF(dims[0], eps=1e-6)
        self.downsample_layers_dot_1_dot_1 = nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        self.downsample_layers_dot_2_dot_0 = LayerNorm_CF(dims[1], eps=1e-6)
        self.downsample_layers_dot_2_dot_1 = nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
        self.downsample_layers_dot_3_dot_0 = LayerNorm_CF(dims[2], eps=1e-6)
        self.downsample_layers_dot_3_dot_1 = nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)

        self.stages_dot_0_dot_0 = Block(dim=dims[0], drop_path=0.0,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_0_dot_1 = Block(dim=dims[0], drop_path=0.0059,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_0_dot_2 = Block(dim=dims[0], drop_path=0.0118,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_1_dot_0 = Block(dim=dims[1], drop_path=0.0176,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_1_dot_1 = Block(dim=dims[1], drop_path=0.0235,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_1_dot_2 = Block(dim=dims[1], drop_path=0.0294,
                                        layer_scale_init_value=layer_scale_init_value)

        self.stages_dot_2_dot_0 = Block(dim=dims[2], drop_path=0.0352,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_2_dot_1 = Block(dim=dims[2], drop_path=0.0411,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_2_dot_2 = Block(dim=dims[2], drop_path=0.047,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_2_dot_3 = Block(dim=dims[2], drop_path=0.053,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_2_dot_4 = Block(dim=dims[2], drop_path=0.0588,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_2_dot_5 = Block(dim=dims[2], drop_path=0.0647,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_2_dot_6 = Block(dim=dims[2], drop_path=0.0705,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_2_dot_7 = Block(dim=dims[2], drop_path=0.0764,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_2_dot_8 = Block(dim=dims[2], drop_path=0.0823,
                                        layer_scale_init_value=layer_scale_init_value)

        self.stages_dot_3_dot_0 = Block(dim=dims[3], drop_path=0.088,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_3_dot_1 = Block(dim=dims[3], drop_path=0.0941,
                                        layer_scale_init_value=layer_scale_init_value)
        self.stages_dot_3_dot_2 = Block(dim=dims[3], drop_path=0.1,
                                        layer_scale_init_value=layer_scale_init_value)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    @jit.script_method
    def forward(self, x):
        x = self.downsample_layers_dot_0_dot_0(x)
        x = self.downsample_layers_dot_0_dot_1(x)
        x = self.stages_dot_0_dot_0(x)
        x = self.stages_dot_0_dot_1(x)
        x = self.stages_dot_0_dot_2(x)
        x = self.downsample_layers_dot_1_dot_0(x)
        x = self.downsample_layers_dot_1_dot_1(x)
        x = self.stages_dot_1_dot_0(x)
        x = self.stages_dot_1_dot_1(x)
        x = self.stages_dot_1_dot_2(x)
        x = self.downsample_layers_dot_2_dot_0(x)
        x = self.downsample_layers_dot_2_dot_1(x)
        x = self.stages_dot_2_dot_0(x)
        x = self.stages_dot_2_dot_1(x)
        x = self.stages_dot_2_dot_2(x)
        x = self.stages_dot_2_dot_3(x)
        x = self.stages_dot_2_dot_4(x)
        x = self.stages_dot_2_dot_5(x)
        x = self.stages_dot_2_dot_6(x)
        x = self.stages_dot_2_dot_7(x)
        x = self.stages_dot_2_dot_8(x)
        x = self.downsample_layers_dot_3_dot_0(x)
        x = self.downsample_layers_dot_3_dot_1(x)
        x = self.stages_dot_3_dot_0(x)
        x = self.stages_dot_3_dot_1(x)
        x = self.stages_dot_3_dot_2(x)
        x = self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        x = self.head(x)
        return x


class LayerNorm_CL(jit.ScriptModule):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    @jit.script_method
    def forward(self, x) -> Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class LayerNorm_CF(jit.ScriptModule):  # jit.ScriptModule
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    @jit.script_method
    def forward(self, x) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


# @register_model
def convnext_tiny(pretrained=False, num_classes=1000):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes,
                     drop_path_rate=0.1, head_init_scale=1.0, )
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        # Correct weights name:
        mappedWeights = {}
        for key, value in checkpoint['model'].items():
            tempKey = key.replace(".", "_dot_")
            tempKey = tempKey.replace("_dot_weight", ".weight")
            tempKey = tempKey.replace("_dot_bias", ".bias")
            tempKey = tempKey.replace("_dot_gamma", ".gamma")
            tempKey = tempKey.replace("_dot_dwconv", ".dwconv")
            tempKey = tempKey.replace("_dot_norm", ".norm")
            tempKey = tempKey.replace("_dot_pwconv", ".pwconv")
            mappedWeights[tempKey] = value
        mappedWeights.pop('head.weight', None)
        mappedWeights.pop('head.bias', None)
        model.load_state_dict(mappedWeights, strict=False)
    return model
