"""
Pose Estimation Network
"""

from models.basic import *


class PoseNet(nn.Module):
    def __init__(self, latent=''):
        super(PoseNet, self).__init__()


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, outermost=True,
                 use_dropout=True, dilation=1, combine_resnet=None, block_type=1):
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
                                             outermost=outermost, norm_layer=norm_layer)  # add the outermost layer

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
            norm_layer = False
        seq = [nn.LeakyReLU(0.2),
               MultiDilation(dim_out=outer_nc, dilation=dilation, norm_layer=norm_layer)
               ]

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)

            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up + seq + [nn.Tanh()]
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
            norm_layer = False
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
                model = down + [submodule]  + seq + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
