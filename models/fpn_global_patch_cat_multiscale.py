from .resnet import resnet50, resnet101
import torch.nn as nn
import torch.nn.functional as F
import torch

class fpn_module_global(nn.Module):
    def __init__(self, numClass):
        super(fpn_module_global, self).__init__()
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) # Reduce channels

        # Smooth layers
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Classify layers
        self.classify = nn.Conv2d(128*4, numClass, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), mode='bilinear')
        p4 = F.interpolate(p4, size=(H, W), mode='bilinear')
        p3 = F.interpolate(p3, size=(H, W), mode='bilinear')
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, c2, c3, c4, c5):
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        smooths = {'p5': self.smooth1_1(p5), 'p4': self.smooth2_1(p4), 'p3': self.smooth3_1(p3), 'p2': self.smooth4_1(p2)}
        p5 = self.smooth1_2(smooths['p5'])
        p4 = self.smooth2_2(smooths['p4'])
        p3 = self.smooth3_2(smooths['p3'])
        p2 = self.smooth4_2(smooths['p2'])
        # Classify
        output = self.classify(self._concatenate(p5, p4, p3, p2))

        return output, smooths


class fpn_module_local(nn.Module):
    def __init__(self, numClass):
        super(fpn_module_local, self).__init__()
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # self.smooth1_2 = nn.Conv2d(256 * 5, 128, kernel_size=3, stride=1, padding=1)
        # self.smooth2_2 = nn.Conv2d(256 * 5, 128, kernel_size=3, stride=1, padding=1)
        # self.smooth3_2 = nn.Conv2d(256 * 5, 128, kernel_size=3, stride=1, padding=1)
        # self.smooth4_2 = nn.Conv2d(256 * 5, 128, kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(256 * 5, 64 * 5, kernel_size=3, stride=1, padding=1, groups=5)
        self.smooth2_2 = nn.Conv2d(256 * 5, 64 * 5, kernel_size=3, stride=1, padding=1, groups=5)
        self.smooth3_2 = nn.Conv2d(256 * 5, 64 * 5, kernel_size=3, stride=1, padding=1, groups=5)
        self.smooth4_2 = nn.Conv2d(256 * 5, 64 * 5, kernel_size=3, stride=1, padding=1, groups=5)

        # Classify layers
        # self.classify = nn.Conv2d(128*4, numClass, kernel_size=3, stride=1, padding=1)
        self.classify = nn.Conv2d(64*5*4, numClass, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), mode='bilinear')
        p4 = F.interpolate(p4, size=(H, W), mode='bilinear')
        p3 = F.interpolate(p3, size=(H, W), mode='bilinear')
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y
    
    def _get_global_patch_ms(self, f_local, f_global, top_left, ratio):
        '''
        simultaneously fuse p2~p5 together to f_local
        f_local: c, h, w
        f_global: dict of {'p5', 'p4', 'p3', 'p2'}: 1, c, h', w'
        '''
        c, Hl, Wl = f_local.size()
        
        _, _, Hg, Wg = f_global['p5'].size()
        top, left = int(top_left[0] * Hg), int(top_left[1] * Wg)
        h, w = int(Hg * ratio), int(Wg * ratio)
        gp5 = F.interpolate(f_global['p5'][0:1, :, top:top+h, left:left+w], size=(Hl, Wl), mode='bilinear')
        
        _, _, Hg, Wg = f_global['p4'].size()
        top, left = int(top_left[0] * Hg), int(top_left[1] * Wg)
        h, w = int(Hg * ratio), int(Wg * ratio)
        gp4 = F.interpolate(f_global['p4'][0:1, :, top:top+h, left:left+w], size=(Hl, Wl), mode='bilinear')
        
        _, _, Hg, Wg = f_global['p3'].size()
        top, left = int(top_left[0] * Hg), int(top_left[1] * Wg)
        h, w = int(Hg * ratio), int(Wg * ratio)
        gp3 = F.interpolate(f_global['p3'][0:1, :, top:top+h, left:left+w], size=(Hl, Wl), mode='bilinear')
        
        _, _, Hg, Wg = f_global['p2'].size()
        top, left = int(top_left[0] * Hg), int(top_left[1] * Wg)
        h, w = int(Hg * ratio), int(Wg * ratio)
        gp2 = F.interpolate(f_global['p2'][0:1, :, top:top+h, left:left+w], size=(Hl, Wl), mode='bilinear')
        
        # concatenate along channels
        return torch.cat((f_local, gp5[0], gp4[0], gp3[0], gp2[0]), dim=0)

    def forward(self, c2, c3, c4, c5, smooths_global, top_lefts, ratio):
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        
        # Smooth
        smooth5 = self.smooth1_1(p5) # b, c, h, w, each one in batch cat with one partial of global
        smooth4 = self.smooth2_1(p4) # b, c, h, w, each one in batch cat with one partial of global
        smooth3 = self.smooth3_1(p3) # b, c, h, w, each one in batch cat with one partial of global
        smooth2 = self.smooth4_1(p2) # b, c, h, w, each one in batch cat with one partial of global
        b, _, _, _ = smooth5.size()
        smooth5_fuse = []
        smooth4_fuse = []
        smooth3_fuse = []
        smooth2_fuse = []
        for i in range(b):
            smooth5_fuse.append(self._get_global_patch_ms(smooth5[i], smooths_global, top_lefts[i], ratio))
            smooth4_fuse.append(self._get_global_patch_ms(smooth4[i], smooths_global, top_lefts[i], ratio))
            smooth3_fuse.append(self._get_global_patch_ms(smooth3[i], smooths_global, top_lefts[i], ratio))
            smooth2_fuse.append(self._get_global_patch_ms(smooth2[i], smooths_global, top_lefts[i], ratio))
        smooth5_fuse = torch.stack(smooth5_fuse, dim=0)
        smooth4_fuse = torch.stack(smooth4_fuse, dim=0)
        smooth3_fuse = torch.stack(smooth3_fuse, dim=0)
        smooth2_fuse = torch.stack(smooth2_fuse, dim=0)
        
        p5 = self.smooth1_2(smooth5_fuse)
        p4 = self.smooth2_2(smooth4_fuse)
        p3 = self.smooth3_2(smooth3_fuse)
        p2 = self.smooth4_2(smooth2_fuse)
        # Classify
        output = self.classify(self._concatenate(p5, p4, p3, p2))

        return output



class fpn(nn.Module):
    def __init__(self, numClass):
        super(fpn, self).__init__()
        # Res net
        self.resnet = resnet50(True)
        self.resnet_local = resnet50(True)

        # fpn module
        self.fpn = fpn_module_global(numClass)
        self.fpn_local = fpn_module_local(numClass)

        # init fpn
        for m in self.fpn.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)
        for m in self.fpn_local.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, image_global, patches, top_lefts, ratio):
        # Top-down
        c2_global, c3_global, c4_global, c5_global = self.resnet.forward(image_global)
        c2_patches, c3_patches, c4_patches, c5_patches = self.resnet_local.forward(patches)
        
        output_global, smooths_global = self.fpn.forward(c2_global, c3_global, c4_global, c5_global)
        output_patches = self.fpn_local.forward(c2_patches, c3_patches, c4_patches, c5_patches, smooths_global, top_lefts, ratio)
        
        return output_global, output_patches
