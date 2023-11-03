"""
from https://github.com/AghdamAmir/3D-UNet/blob/main/unet3d.py
"""

from torch import nn
import torch
import time

class ASPPConv3D(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv3D, self).__init__(*modules)


class ASPPPooling3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling3D, self).__init__()
        self.aspp_pooling = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )

    def set_image_pooling(self, pool_size=None):
        if pool_size is None:
            self.aspp_pooling[0] = nn.AdaptiveAvgPool3d(1)
        else:
            self.aspp_pooling[0] = nn.AvgPool3d(kernel_size=pool_size, stride=1)

    def forward(self, x):
        size = x.shape[-3:]
        x = self.aspp_pooling(x)
        return F.interpolate(x, size=size, mode='trilinear', align_corners=True)


class ASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(ASPP3D, self).__init__()

        modules = []

        for rate in dilation_rates:
            modules.append(ASPPConv3D(in_channels, out_channels, rate))
        modules.append(ASPPPooling3D(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d((len(dilation_rates) + 1) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def set_image_pooling(self, pool_size):
        self.convs[-1].set_image_pooling(pool_size)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)


    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res




class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, out_channels=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and out_channels==None) or (last_layer==True and out_channels!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=out_channels, kernel_size=(1,1,1))


    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out


class PanopticDeepLab3D(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512, aspp=False,
                 dilation_rates=[1,6,12], separate_decoders=False, scale_offsets=1,
                 seg_loss_weight=1, heatmap_loss_weight=1, offsets_loss_weight=1) -> None:
        super(PanopticDeepLab3D, self).__init__()

        self.scale_offsets = scale_offsets
        self.seg_loss_weight = seg_loss_weight
        self.heatmap_loss_weight = heatmap_loss_weight
        self.offsets_loss_weight = offsets_loss_weight
        self.separate_decoders = separate_decoders
        self.num_classes = num_classes

        # Analysis Path
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_channels[0])
        self.a_block2 = Conv3DBlock(in_channels=level_channels[0], out_channels=level_channels[1])
        self.a_block3 = Conv3DBlock(in_channels=level_channels[1], out_channels=level_channels[2])
        self.bottleNeck = Conv3DBlock(in_channels=level_channels[2], out_channels=bottleneck_channel, bottleneck=True)
        self.aspp = ASPP3D(bottleneck_channel, bottleneck_channel, dilation_rates) if aspp else None
        
        if seg_loss_weight > 0:
            # Semantic Decoding Path
            self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_channels[2])
            self.s_block2 = UpConv3DBlock(in_channels=level_channels[2], res_channels=level_channels[1])

            if not self.separate_decoders:
                self.s_block1 = UpConv3DBlock(in_channels=level_channels[1], res_channels=level_channels[0],
                                              out_channels=num_classes + 4, last_layer=True)
            else:
                self.s_block1 = UpConv3DBlock(in_channels=level_channels[1], res_channels=level_channels[0],
                                              out_channels=num_classes, last_layer=True)

        self.oc_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_channels[2])
        self.oc_block2 = UpConv3DBlock(in_channels=level_channels[2], res_channels=level_channels[1])
        if heatmap_loss_weight <= 0:
            self.oc_block1 = UpConv3DBlock(in_channels=level_channels[1], res_channels=level_channels[0],
                                            out_channels=3, last_layer=True)
        else:
            self.oc_block1 = UpConv3DBlock(in_channels=level_channels[1], res_channels=level_channels[0],
                                            out_channels=4, last_layer=True)

        self._init_parameters()

    def forward(self, input):
        # Analysis Pathway
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)
        if self.aspp:
            out = self.aspp(out)
        
        if self.seg_loss_weight > 0:
            # Semantic Decoding Pathway
            sem_decoder_out = self.s_block3(out, residual_level3)
            sem_decoder_out = self.s_block2(sem_decoder_out, residual_level2)
            sem_decoder_out = self.s_block1(sem_decoder_out, residual_level1)
        
            if not self.separate_decoders:
                semantic_out = sem_decoder_out[:,:2]
                center_prediction_out = sem_decoder_out[:,2:3]
                offsets_out = sem_decoder_out[:,3:] * self.scale_offsets
            else:
                oc_decoder_out = self.oc_block3(out, residual_level3)
                oc_decoder_out = self.oc_block2(oc_decoder_out, residual_level2)
                oc_decoder_out = self.oc_block1(oc_decoder_out, residual_level1)
        
                semantic_out = sem_decoder_out
                center_prediction_out = oc_decoder_out[:, :1]
                offsets_out = oc_decoder_out[:, 1:] * self.scale_offsets

            return semantic_out, center_prediction_out, offsets_out
        
        oc_decoder_out = self.oc_block3(out, residual_level3)
        oc_decoder_out = self.oc_block2(oc_decoder_out, residual_level2)
        oc_decoder_out = self.oc_block1(oc_decoder_out, residual_level1)
        s = list(oc_decoder_out.shape)
        s[1] = self.num_classes
        ss = tuple(s)
        
        device = oc_decoder_out.device
        if self.heatmap_loss_weight <= 0:
            s[1] = 1
            sc = tuple(s)
            return torch.zeros(ss).to(device), torch.zeros(sc).to(device), oc_decoder_out * self.scale_offsets

        return torch.zeros(ss).to(device), oc_decoder_out[:, :1], oc_decoder_out[:,1:] * self.scale_offsets


    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


def get_pretrained_model(model_path, in_channels, num_classes=2):
    model = UNet3D(in_channels=1, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    if in_channels == 1:
        return model
        
    sd_layer1 = model.a_block1.conv1.state_dict()
    
    duplicated_weight = sd_layer1["weight"].repeat(1,in_channels,1,1,1)
    bias = sd_layer1["bias"] 
    
    out_channels = model.a_block1.conv1.out_channels
    new_layer = nn.Conv3d(in_channels = in_channels, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
    new_layer.load_state_dict({"weight": duplicated_weight, "bias": bias})
    model.a_block1.conv1 = new_layer
    
    return model


def freeze_pretrained_layers(model):
    for param in model.parameters():
        if param == "":
            param.requires_grad = False


if __name__ == '__main__':
    #model= get_pretrained_model("3dunet_ms-shift.pth", in_channels = 2)
    #torch.save(model.state_dict(), "UNet3D_2channel_input.pth")

    model = PanopticDeepLab3D(in_channels=1, num_classes=2).cuda()
    a = torch.rand(2,1,96,96,96).cuda()
    output = model(a)
    breakpoint()

