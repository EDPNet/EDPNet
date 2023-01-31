import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models
import model.xception as models

# from model.xception import xception


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()#torch.size([2,2048,11,11])
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)



class EDPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=9,zoom_factor=16,use_ppm=True,pretrained=False,criterion=nn.CrossEntropyLoss(ignore_index=255)):
        super(EDPNet, self).__init__()
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [2,4,8,16]
        self.zoom_factor = zoom_factor
        # print(self.zoom_factor)
        self.use_ppm = use_ppm
        self.criterion = criterion
        if layers == 50:
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            xception=models.xception(zoom_factor=zoom_factor, pretrained=pretrained)
            # self.layers = xception(downsample_factor=downsample_factor, pretrained=pretrained)
        self.layer0 = nn.Sequential(xception.conv1, xception.bn1, xception.relu, xception.conv2, xception.bn2,xception.relu)
        self.layer1, self.layer2 =xception.block1,xception.block2
        # low_featrue_layer = self.layer2.hook_layer

        self.layer3=xception.block3
        # rate = 16 // zoom_factor
        self.layer4,self.layer5= xception.block4,xception.block5
        self.layer6, self.layer7,self.layer8,self.layer9,self.layer10=xception.block6,xception.block7,xception.block8,xception.block9,xception.block10
        self.layer11, self.layer12,self.layer13,self.layer14,self.layer15=xception.block11,xception.block12,xception.block13,xception.block14,xception.block15
        self.layer16, self.layer17,self.layer18,self.layer19,self.layer20=xception.block16,xception.block17,xception.block18,xception.block19,xception.block20
        self.layer21=nn.Sequential(xception.conv3,xception.conv4,xception.conv5)


        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        x_size = x.size()#torch.size([2,3,321,321])
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)#h=321  self.zoom_factor=8
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        # H, W = x.size(2), x.size(3)


        x = self.layer0(x)#torch.size([2,64,161,161])
        x = self.layer1(x)#torch.size([2,128,81,81])
        x = self.layer2(x)#torch.size([2,256,41,41])
        # low_featrue_layer = self.layer2.hook_layer
        x = self.layer3(x)#torch.size([2,728,21,21])
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)#torch.size([2,728,21,21])
        x = self.layer19(x)#torch.size([2,728,21,21])
        x_tmp= self.layer20(x)#torch.size([2,1024,11,11])
        x = self.layer21(x_tmp)#torch.size([2,2048,11,11])



        if self.use_ppm:
            x = self.ppm(x)#torch.size([2,4096,11,11])
        x=self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)#torch.size([2,9,321,321])

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)#torch.size([2,9,321,321])

            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    input = torch.rand(4, 3, 473, 473).cuda()
    model = EDPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=9, use_ppm=True,zoom_factor=8, pretrained=True).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('EDPNet', output.size())
