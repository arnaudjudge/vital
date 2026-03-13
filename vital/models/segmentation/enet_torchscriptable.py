from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _InitialBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 0,
                 bias: bool = False, relu: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

        self.main_branch = nn.Conv2d(
            in_channels, out_channels - in_channels,
            kernel_size=[kernel_size, kernel_size], stride=[2, 2], padding=[padding, padding], bias=bias,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        if relu:
            self.out_activation: nn.Module = nn.ReLU()
        else:
            self.out_activation = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        main = self.main_branch(x)
        # Use F.max_pool2d directly instead of nn.MaxPool2d submodule —
        # TorchScript re-reads stored scalar attributes of nn.MaxPool2d at
        # script time and fails with "Expected IntList but got Int".
        ext = F.max_pool2d(x, [self.kernel_size, self.kernel_size],
                           stride=[2, 2], padding=[self.padding, self.padding])
        out = torch.cat((main, ext), 1)
        out = self.batch_norm(out)
        return self.out_activation(out)


class _RegularBottleneck(nn.Module):
    def __init__(self, channels: int, internal_ratio: int = 4, kernel_size: int = 3, padding: int = 0,
                 dilation: int = 1, asymmetric: bool = False, dropout: float = 0.0,
                 bias: bool = False, relu: bool = True):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError(f"Value out of range. Expected value in the interval [1, {channels}], got internal_ratio={internal_ratio}.")
        internal_channels = channels // internal_ratio

        if relu:
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(channels, internal_channels, kernel_size=[1, 1], stride=[1, 1], bias=bias),
                nn.BatchNorm2d(internal_channels), nn.ReLU(),
            )
        else:
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(channels, internal_channels, kernel_size=[1, 1], stride=[1, 1], bias=bias),
                nn.BatchNorm2d(internal_channels), nn.PReLU(),
            )

        if asymmetric:
            if relu:
                self.ext_conv2 = nn.Sequential(
                    nn.Conv2d(internal_channels, internal_channels, kernel_size=[kernel_size, 1], stride=[1, 1],
                              padding=[padding, 0], dilation=[dilation, dilation], bias=bias),
                    nn.BatchNorm2d(internal_channels), nn.ReLU(),
                    nn.Conv2d(internal_channels, internal_channels, kernel_size=[1, kernel_size], stride=[1, 1],
                              padding=[0, padding], dilation=[dilation, dilation], bias=bias),
                    nn.BatchNorm2d(internal_channels), nn.ReLU(),
                )
            else:
                self.ext_conv2 = nn.Sequential(
                    nn.Conv2d(internal_channels, internal_channels, kernel_size=[kernel_size, 1], stride=[1, 1],
                              padding=[padding, 0], dilation=[dilation, dilation], bias=bias),
                    nn.BatchNorm2d(internal_channels), nn.PReLU(),
                    nn.Conv2d(internal_channels, internal_channels, kernel_size=[1, kernel_size], stride=[1, 1],
                              padding=[0, padding], dilation=[dilation, dilation], bias=bias),
                    nn.BatchNorm2d(internal_channels), nn.PReLU(),
                )
        else:
            if relu:
                self.ext_conv2 = nn.Sequential(
                    nn.Conv2d(internal_channels, internal_channels, kernel_size=[kernel_size, kernel_size],
                              stride=[1, 1], padding=[padding, padding], dilation=[dilation, dilation], bias=bias),
                    nn.BatchNorm2d(internal_channels), nn.ReLU(),
                )
            else:
                self.ext_conv2 = nn.Sequential(
                    nn.Conv2d(internal_channels, internal_channels, kernel_size=[kernel_size, kernel_size],
                              stride=[1, 1], padding=[padding, padding], dilation=[dilation, dilation], bias=bias),
                    nn.BatchNorm2d(internal_channels), nn.PReLU(),
                )

        if relu:
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(internal_channels, channels, kernel_size=[1, 1], stride=[1, 1], bias=bias),
                nn.BatchNorm2d(channels), nn.ReLU(),
            )
            self.out_prelu: nn.Module = nn.ReLU()
        else:
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(internal_channels, channels, kernel_size=[1, 1], stride=[1, 1], bias=bias),
                nn.BatchNorm2d(channels), nn.PReLU(),
            )
            self.out_prelu = nn.PReLU()

        self.ext_regul = nn.Dropout2d(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_prelu(main + ext)


class _DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, internal_ratio: int = 4, kernel_size: int = 3,
                 padding: int = 0, return_indices: bool = False, dropout: float = 0.0,
                 bias: bool = False, relu: bool = True):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(f"Value out of range. Expected value in the interval [1, {in_channels}], got internal_ratio={internal_ratio}.")
        internal_channels = in_channels // internal_ratio

        # Store pool parameters for use in forward() with F.max_pool2d,
        # avoiding nn.MaxPool2d whose stored scalar attrs break TorchScript.
        self.pool_kernel_size = kernel_size
        self.pool_padding = padding

        if relu:
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(in_channels, internal_channels, kernel_size=[2, 2], stride=[2, 2], bias=bias),
                nn.BatchNorm2d(internal_channels), nn.ReLU(),
            )
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=[kernel_size, kernel_size],
                          stride=[1, 1], padding=[padding, padding], bias=bias),
                nn.BatchNorm2d(internal_channels), nn.ReLU(),
            )
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(internal_channels, out_channels, kernel_size=[1, 1], stride=[1, 1], bias=bias),
                nn.BatchNorm2d(out_channels), nn.ReLU(),
            )
            self.out_prelu: nn.Module = nn.ReLU()
        else:
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(in_channels, internal_channels, kernel_size=[2, 2], stride=[2, 2], bias=bias),
                nn.BatchNorm2d(internal_channels), nn.PReLU(),
            )
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=[kernel_size, kernel_size],
                          stride=[1, 1], padding=[padding, padding], bias=bias),
                nn.BatchNorm2d(internal_channels), nn.PReLU(),
            )
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(internal_channels, out_channels, kernel_size=[1, 1], stride=[1, 1], bias=bias),
                nn.BatchNorm2d(out_channels), nn.PReLU(),
            )
            self.out_prelu = nn.PReLU()

        self.ext_regul = nn.Dropout2d(p=dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # F.max_pool2d with return_indices=True always returns Tuple[Tensor, Tensor],
        # giving TorchScript a single static return type to reason about.
        main, max_indices = F.max_pool2d(
            x,
            [self.pool_kernel_size, self.pool_kernel_size],
            stride=[2, 2],
            padding=[self.pool_padding, self.pool_padding],
            return_indices=True,
        )

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        pad = torch.zeros(n, ch_ext - ch_main, h, w, device=main.device, dtype=main.dtype)
        main = torch.cat((main, pad), 1)

        return self.out_prelu(main + ext), max_indices


class _UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, internal_ratio: int = 4, kernel_size: int = 3,
                 padding: int = 0, dropout: float = 0.0, bias: bool = False, relu: bool = True):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(f"Value out of range. Expected value in the interval [1, {in_channels}], got internal_ratio={internal_ratio}.")
        internal_channels = in_channels // internal_ratio

        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=[1, 1], bias=bias),
            nn.BatchNorm2d(out_channels),
        )
        # nn.MaxUnpool2d suffers the same scalar-attr problem as nn.MaxPool2d.
        # Use F.max_unpool2d in forward() instead.

        if relu:
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(in_channels, internal_channels, kernel_size=[1, 1], bias=bias),
                nn.BatchNorm2d(internal_channels), nn.ReLU(),
            )
            self.ext_conv2 = nn.Sequential(
                nn.ConvTranspose2d(internal_channels, internal_channels,
                                   kernel_size=[kernel_size, kernel_size], stride=[2, 2],
                                   padding=[padding, padding], output_padding=[1, 1], bias=bias),
                nn.BatchNorm2d(internal_channels), nn.ReLU(),
            )
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(internal_channels, out_channels, kernel_size=[1, 1], bias=bias),
                nn.BatchNorm2d(out_channels), nn.ReLU(),
            )
            self.out_prelu: nn.Module = nn.ReLU()
        else:
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(in_channels, internal_channels, kernel_size=[1, 1], bias=bias),
                nn.BatchNorm2d(internal_channels), nn.PReLU(),
            )
            self.ext_conv2 = nn.Sequential(
                nn.ConvTranspose2d(internal_channels, internal_channels,
                                   kernel_size=[kernel_size, kernel_size], stride=[2, 2],
                                   padding=[padding, padding], output_padding=[1, 1], bias=bias),
                nn.BatchNorm2d(internal_channels), nn.PReLU(),
            )
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(internal_channels, out_channels, kernel_size=[1, 1], bias=bias),
                nn.BatchNorm2d(out_channels), nn.PReLU(),
            )
            self.out_prelu = nn.PReLU()

        self.ext_regul = nn.Dropout2d(p=dropout)

    def forward(self, x: Tensor, max_indices: Tensor) -> Tensor:
        main = self.main_conv1(x)
        main = F.max_unpool2d(main, max_indices, kernel_size=[2, 2])
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_prelu(main + ext)


class Enet(nn.Module):
    """Implementation of the ENet model.

    References:
        - Paper that introduced the model: http://arxiv.org/abs/1606.02147
    """

    def __init__(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                 init_channels: int = 16, dropout: float = 0.1,
                 encoder_relu: bool = True, decoder_relu: bool = True):
        super().__init__()
        in_channels = input_shape[0]
        out_channels = output_shape[0]

        self.initial_block = _InitialBlock(in_channels, init_channels, padding=1, relu=encoder_relu)

        self.downsample1_0 = _DownsamplingBottleneck(init_channels, init_channels * 2, padding=1, return_indices=True, dropout=0.1 * dropout, relu=encoder_relu)
        self.regular1_1 = _RegularBottleneck(init_channels * 2, padding=1, dropout=0.1 * dropout, relu=encoder_relu)
        self.regular1_2 = _RegularBottleneck(init_channels * 2, padding=1, dropout=0.1 * dropout, relu=encoder_relu)
        self.regular1_3 = _RegularBottleneck(init_channels * 2, padding=1, dropout=0.1 * dropout, relu=encoder_relu)
        self.regular1_4 = _RegularBottleneck(init_channels * 2, padding=1, dropout=0.1 * dropout, relu=encoder_relu)

        self.downsample2_0 = _DownsamplingBottleneck(init_channels * 2, init_channels * 4, padding=1, return_indices=True, dropout=dropout, relu=encoder_relu)
        self.regular2_1 = _RegularBottleneck(init_channels * 4, padding=1, dropout=dropout, relu=encoder_relu)
        self.dilated2_2 = _RegularBottleneck(init_channels * 4, dilation=2, padding=2, dropout=dropout, relu=encoder_relu)
        self.asymmetric2_3 = _RegularBottleneck(init_channels * 4, kernel_size=5, padding=2, asymmetric=True, dropout=dropout, relu=encoder_relu)
        self.dilated2_4 = _RegularBottleneck(init_channels * 4, dilation=4, padding=4, dropout=dropout, relu=encoder_relu)
        self.regular2_5 = _RegularBottleneck(init_channels * 4, padding=1, dropout=dropout, relu=encoder_relu)
        self.dilated2_6 = _RegularBottleneck(init_channels * 4, dilation=8, padding=8, dropout=dropout, relu=encoder_relu)
        self.asymmetric2_7 = _RegularBottleneck(init_channels * 4, kernel_size=5, asymmetric=True, padding=2, dropout=dropout, relu=encoder_relu)
        self.dilated2_8 = _RegularBottleneck(init_channels * 4, dilation=16, padding=16, dropout=dropout, relu=encoder_relu)

        self.regular3_0 = _RegularBottleneck(init_channels * 4, padding=1, dropout=dropout, relu=encoder_relu)
        self.dilated3_1 = _RegularBottleneck(init_channels * 4, dilation=2, padding=2, dropout=dropout, relu=encoder_relu)
        self.asymmetric3_2 = _RegularBottleneck(init_channels * 4, kernel_size=5, padding=2, asymmetric=True, dropout=dropout, relu=encoder_relu)
        self.dilated3_3 = _RegularBottleneck(init_channels * 4, dilation=4, padding=4, dropout=dropout, relu=encoder_relu)
        self.regular3_4 = _RegularBottleneck(init_channels * 4, padding=1, dropout=dropout, relu=encoder_relu)
        self.dilated3_5 = _RegularBottleneck(init_channels * 4, dilation=8, padding=8, dropout=dropout, relu=encoder_relu)
        self.asymmetric3_6 = _RegularBottleneck(init_channels * 4, kernel_size=5, asymmetric=True, padding=2, dropout=dropout, relu=encoder_relu)
        self.dilated3_7 = _RegularBottleneck(init_channels * 4, dilation=16, padding=16, dropout=dropout, relu=encoder_relu)

        self.upsample4_0 = _UpsamplingBottleneck(init_channels * 4, init_channels * 2, padding=1, dropout=dropout, relu=decoder_relu)
        self.regular4_1 = _RegularBottleneck(init_channels * 2, padding=1, dropout=dropout, relu=decoder_relu)
        self.regular4_2 = _RegularBottleneck(init_channels * 2, padding=1, dropout=dropout, relu=decoder_relu)

        self.upsample5_0 = _UpsamplingBottleneck(init_channels * 2, init_channels, padding=1, dropout=dropout, relu=decoder_relu)
        self.regular5_1 = _RegularBottleneck(init_channels, padding=1, dropout=dropout, relu=decoder_relu)

        self.transposed_conv = nn.ConvTranspose2d(
            init_channels, out_channels,
            kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], output_padding=[1, 1], bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.initial_block(x)

        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)

        return self.transposed_conv(x)