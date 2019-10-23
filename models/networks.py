"""
Pose Estimation Network
"""

from models.basic import *
from torchvision import transforms


class PoseNet(nn.Module):
    def __init__(self, latent=''):
        super(PoseNet, self).__init__()


class ResNet2HeadGenerator(nn.Module):
    def __init__(self, latent=ResNet18Features(), out_channels=3, norm=nn.BatchNorm2d,
                 last_layer=nn.LeakyReLU(0.2, inplace=True)):
        super(ResNet2HeadGenerator, self).__init__()

        model = [latent]
        in_ch = 256
        out_ch = in_ch // 2
        dropout = False
        for idx in range(3):
            if idx == 2:
                dropout = True
            # else:
            #     dropout = False
            model += [nn.LeakyReLU(0.2, inplace=True),
                      UpConvLayer(in_ch=in_ch, out_ch=out_ch, stride=2, dropout=dropout, skip=False, norm=norm),
                      ]
            # if idx < 2:
            #     model += [MultiDilation(dim_out=out_ch, dilation=1)]
            in_ch = out_ch
            out_ch = in_ch // 2

        self.model = nn.Sequential(*model)

        self.output_list = nn.ModuleList()
        for idx in range(2):
            if idx is 0:
                out_ch = 2
            else:
                out_ch = out_channels

            seq = [
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.2),
                UpConvLayer(in_ch=in_ch, out_ch=out_ch, stride=2, dropout=dropout, skip=False, norm=None),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.2),
                MultiDilation(dim_out=out_ch, dilation=1, norm_layer=None)
            ]
            if idx == 1:
                seq += [last_layer]
            self.output_list.append(nn.Sequential(*seq))

    def forward(self, x):
        net = self.model(x)
        outputs = []
        for idx in range(2):
            outputs.append(self.output_list[idx](net))

        return outputs


class ResNetGenerator(nn.Module):
    def __init__(self, latent=ResNet18Features(), out_channels=3, norm=nn.BatchNorm2d,
                 last_layer=nn.LeakyReLU(0.2, inplace=True)):
        super(ResNetGenerator, self).__init__()

        model = [latent]
        in_ch = 256
        out_ch = in_ch // 2
        dropout = False
        for idx in range(4):
            if idx == 2:
                dropout = True
            # else:
            #     dropout = False
            if idx == 3:
                out_ch = out_channels
            model += [nn.LeakyReLU(0.2, inplace=True),
                      UpConvLayer(in_ch=in_ch, out_ch=out_ch, stride=2, dropout=dropout, skip=False, norm=norm)
                      ]
            if idx < 2:
                model += [nn.LeakyReLU(0.2, inplace=True),
                          MultiDilation(dim_out=out_ch, dilation=1)]
            in_ch = out_ch
            out_ch = in_ch // 2

        model += [last_layer]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Unet2HeadGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, outermost=True,
                 use_dropout=True, dilation=1, combine_resnet=None, last_layer=nn.Tanh(), block_type=1):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Unet2HeadGenerator, self).__init__()

        if block_type is 2:
            UnetSkipConnectionBlock = UnetSkipConnectionBlock2
        else:
            UnetSkipConnectionBlock = UnetSkipConnectionBlock1

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, dilation=dilation)  # add the innermost layer

        if combine_resnet:
            unet_block += combine_resnet
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout, dilation=dilation)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, dilation=dilation)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, dilation=dilation)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, dilation=dilation)
        self.model = UnetSkipConnectionBlock(16, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=outermost, norm_layer=norm_layer, last_layer=nn.LeakyReLU(0.2))  # add the outermost layer

        self.output_list = nn.ModuleList()
        for idx in range(2):
            if idx is 0:
                out_ch = 2
            else:
                out_ch = output_nc

            seq = [
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.2),
                ConvLayer(in_ch=16, out_ch=out_ch, norm=norm_layer, filter=3, stride=1, pad=1),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.2),
                MultiDilation(dim_out=out_ch, dilation=1, norm_layer=None)
            ]
            if idx == 1:
                seq += [last_layer]
            self.output_list.append(nn.Sequential(*seq))

    def forward(self, input):
        """Standard forward"""
        net = self.model(input)
        outputs = []
        for idx in range(2):
            outputs.append(self.output_list[idx](net))

        return outputs


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, outermost=True,
                 use_dropout=True, dilation=1, combine_resnet=None, last_layer=nn.Tanh(), block_type=1):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()

        if block_type is 2:
            UnetSkipConnectionBlock = UnetSkipConnectionBlock2
        else:
            UnetSkipConnectionBlock = UnetSkipConnectionBlock1

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, dilation=dilation)  # add the innermost layer

        if combine_resnet:
            unet_block += combine_resnet
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout, dilation=dilation)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, dilation=dilation)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, dilation=dilation)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, dilation=dilation)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=outermost, norm_layer=norm_layer, last_layer=last_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock1(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 dilation=1, last_layer=nn.Tanh()):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock1, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.LeakyReLU(0.2, True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            norm_layer = None
        seq = [nn.LeakyReLU(0.2),
               MultiDilation(dim_out=outer_nc, dilation=dilation, norm_layer=norm_layer)
               ]

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)

            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up + seq + [last_layer]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)

            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up + seq
        else:

            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)

            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + seq + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up + seq

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetSkipConnectionBlock2(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 dilation=1):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock2, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.LeakyReLU(0.2, True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            norm_layer = None
            seq2 = [nn.LeakyReLU(0.2),
                    MultiDilation(dim_out=outer_nc, dilation=dilation, norm_layer=norm_layer)
                    ]
        seq = [nn.LeakyReLU(0.2),
               MultiDilation(dim_out=inner_nc * 2, dilation=dilation, norm_layer=norm_layer)
               ]

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # for idx in range(2):
            #     seq += [ConvLayer(outer_nc, outer_nc, filter=3, BN=False)]
            #     if idx is 0:
            #         seq += [nn.LeakyReLU(0.2)]
            # seq += [nn.LeakyReLU(0.2),
            #         MultiDilation(dim_out=outer_nc, dilation=1)
            #         ]
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + seq + up + seq2 + [nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            # for idx in range(2):
            #     seq += [ConvLayer(outer_nc, outer_nc, filter=3, BN=norm_layer)]
            #     if idx is 0:
            #         seq += [nn.LeakyReLU(0.2)]
            # seq += [nn.LeakyReLU(0.2),
            #         MultiDilation(dim_out=outer_nc, dilation=1)
            #         ]
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:

            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            # for idx in range(2):
            #     seq += [conv_layer(outer_nc, outer_nc, dilation=2, filter=3, pad=2, BN=norm_layer)]
            #     if idx is 0:
            #         seq += [nn.LeakyReLU(0.2)]
            # seq = [nn.LeakyReLU(0.2),
            #        MultiDilation(dim_out=outer_nc, dilation=1)
            #        ]
            # seq += [conv_layer(outer_nc, outer_nc, filter=3, BN=norm_layer)]
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + seq + up + [nn.Dropout(0.2)]
            else:
                model = down + [submodule] + seq + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class ResUnetGenerator(nn.Module):
    """Create a ResUnet generator"""

    def __init__(self, output_nc, norm_layer=nn.BatchNorm2d, last_layer=nn.Tanh(),
                 latent=ResNet18Features(final_layer=-2)):

        super(ResUnetGenerator, self).__init__()
        latent = nn.Sequential(*list(*latent.children()))

        # index_list = list(range(8, 1, -1))
        index_list = [8, 7, 6, 4, 2]

        module_res = [latent[index_list[idx + 1]:index] if idx > 0 else latent[:(index + 1)] for idx, index in
                      enumerate(reversed(index_list[:-1]))]

        self.module_res = nn.ModuleList(module_res)

        in_ch = 512

        out_size = 8

        module_out = []
        skip = False
        while out_size < 256:
            out_ch = in_ch // 2
            if out_size == 128:
                out_ch = output_nc
            module_out += [UpConvLayer(in_ch=in_ch, out_ch=out_ch, stride=2, norm=norm_layer, skip=skip)]
            skip = True
            in_ch = in_ch // 2
            out_size *= 2

        self.module_out = nn.ModuleList(module_out)

    def forward(self, input):

        compute_list = [input]
        for idx, layer in enumerate(self.module_res):
            compute_list.append(layer(compute_list[-1]))

        out = compute_list.pop(-1)
        skip = None
        for idx, layer in enumerate(self.module_out):
            if idx > 0:
                skip = compute_list.pop(-1)
            out = layer(out, skip)

        return out


class ResUnetGeneratorTest(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, outermost=True,
                 use_dropout=True, dilation=1, combine_resnet=None, last_layer=nn.Tanh(),
                 latent=ResNet18Features(final_layer=-2)):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(ResUnetGeneratorTest, self).__init__()
        latent = nn.Sequential(*list(*latent.children()))

        index_list = [7, 6, 5, 4, 3, 2]

        in_ch = 512
        # out_ch = in_ch // 2
        # construct unet structure
        unet_block = ResUnetSkipConnectionBlock(in_ch , in_ch, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, down_block=latent[index_list.pop(0)], dilation=dilation)  # add the innermost layer

        if combine_resnet:
            unet_block += combine_resnet

        dropout = False

        for idx in range(4):
            in_ch = in_ch // 2
            if idx == 1 and use_dropout is True:
                dropout = True
            num_chanels = in_ch + in_ch // 2
            # if idx == 1:
            #     num_chanels = in_ch
            unet_block = ResUnetSkipConnectionBlock(in_ch, num_chanels, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer, use_dropout=dropout,
                                                    down_block=latent[index_list.pop(0)], dilation=dilation)

        self.model = ResUnetSkipConnectionBlock(output_nc, in_ch // 2, input_nc=input_nc,
                                                submodule=unet_block, outermost=outermost, norm_layer=norm_layer,
                                                down_block=latent[:(index_list.pop(0) + 1)], last_layer=last_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResUnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, down_block, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 last_layer=nn.Tanh(), dilation=1):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        uprelu = nn.LeakyReLU(0.2, True)
        upnorm = norm_layer(outer_nc)

        down = [down_block]
        if outermost:
            norm_layer = None
        seq = [nn.LeakyReLU(0.2),
               MultiDilation(dim_out=outer_nc, dilation=dilation, norm_layer=norm_layer)
               ]

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)

            # down = [down_block]
            up = [uprelu, upconv]
            model = down + [submodule] + up + seq + [last_layer]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)

            # down = [downrelu, down_block]
            up = [uprelu, upconv, upnorm]
            model = down + up + seq
        else:

            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)

            # down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + seq + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up + seq

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


if __name__ == "__main__":
    model = ResUnetGenerator(output_nc=3)
    # model.train()
    # model = torch.nn.Sequential(*model.children())
    # print(model[0][0])
    # # model = torch.nn.Sequential(*model[0][0].children())
    # for param in model[0][0].parameters():
    #     print(param.requires_grad)
    inp = torch.randn((2, 3, 256, 256))
    out = model(inp)
    print(out.shape)
