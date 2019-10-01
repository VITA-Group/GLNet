from .resnet import resnet50
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class fpn_module_global(nn.Module):
    def __init__(self, numClass):
        super(fpn_module_global, self).__init__()
        self._up_kwargs = {'mode': 'bilinear'}
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
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # Classify layers
        self.classify = nn.Conv2d(128*4, numClass, kernel_size=3, stride=1, padding=1)

        # Local2Global: double #channels ####################################
        # Top layer
        self.toplayer_ext = nn.Conv2d(2048*2, 256, kernel_size=1, stride=1, padding=0) # Reduce channels
        # Lateral layers
        self.latlayer1_ext = nn.Conv2d(1024*2, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_ext = nn.Conv2d(512*2, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_ext = nn.Conv2d(256*2, 256, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.smooth1_1_ext = nn.Conv2d(256*2, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1_ext = nn.Conv2d(256*2, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1_ext = nn.Conv2d(256*2, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1_ext = nn.Conv2d(256*2, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_2_ext = nn.Conv2d(256*2, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2_ext = nn.Conv2d(256*2, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2_ext = nn.Conv2d(256*2, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2_ext = nn.Conv2d(256*2, 128, kernel_size=3, stride=1, padding=1)
        self.smooth = nn.Conv2d(128*4*2, 128*4, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), **self._up_kwargs) + y

    def forward(self, c2, c3, c4, c5, c2_ext=None, c3_ext=None, c4_ext=None, c5_ext=None, ps0_ext=None, ps1_ext=None, ps2_ext=None):

        # Top-down
        if c5_ext is None:
            p5 = self.toplayer(c5)
            p4 = self._upsample_add(p5, self.latlayer1(c4))
            p3 = self._upsample_add(p4, self.latlayer2(c3))
            p2 = self._upsample_add(p3, self.latlayer3(c2))
        else:
            p5 = self.toplayer_ext(torch.cat((c5, c5_ext), dim=1))
            p4 = self._upsample_add(p5, self.latlayer1_ext(torch.cat((c4, c4_ext), dim=1)))
            p3 = self._upsample_add(p4, self.latlayer2_ext(torch.cat((c3, c3_ext), dim=1)))
            p2 = self._upsample_add(p3, self.latlayer3_ext(torch.cat((c2, c2_ext), dim=1)))
        ps0 = [p5, p4, p3, p2]
        
        # Smooth
        if ps0_ext is None:
            p5 = self.smooth1_1(p5)
            p4 = self.smooth2_1(p4)
            p3 = self.smooth3_1(p3)
            p2 = self.smooth4_1(p2)
        else:
            p5 = self.smooth1_1_ext(torch.cat((p5, ps0_ext[0]), dim=1))
            p4 = self.smooth2_1_ext(torch.cat((p4, ps0_ext[1]), dim=1))
            p3 = self.smooth3_1_ext(torch.cat((p3, ps0_ext[2]), dim=1))
            p2 = self.smooth4_1_ext(torch.cat((p2, ps0_ext[3]), dim=1))
        ps1 = [p5, p4, p3, p2]
        
        if ps1_ext is None:
            p5 = self.smooth1_2(p5)
            p4 = self.smooth2_2(p4)
            p3 = self.smooth3_2(p3)
            p2 = self.smooth4_2(p2)
        else:
            p5 = self.smooth1_2_ext(torch.cat((p5, ps1_ext[0]), dim=1))
            p4 = self.smooth2_2_ext(torch.cat((p4, ps1_ext[1]), dim=1))
            p3 = self.smooth3_2_ext(torch.cat((p3, ps1_ext[2]), dim=1))
            p2 = self.smooth4_2_ext(torch.cat((p2, ps1_ext[3]), dim=1))
        ps2 = [p5, p4, p3, p2]

        # Classify
        if ps2_ext is None:
            ps3 = self._concatenate(p5, p4, p3, p2)
            output = self.classify(ps3)
        else:
            p = self._concatenate(
                    torch.cat((p5, ps2_ext[0]), dim=1), 
                    torch.cat((p4, ps2_ext[1]), dim=1), 
                    torch.cat((p3, ps2_ext[2]), dim=1), 
                    torch.cat((p2, ps2_ext[3]), dim=1)
                )
            ps3 = self.smooth(p)
            output = self.classify(ps3)

        return output, ps0, ps1, ps2, ps3


class fpn_module_local(nn.Module):
    def __init__(self, numClass):
        super(fpn_module_local, self).__init__()
        self._up_kwargs = {'mode': 'bilinear'}
        # Top layer
        fold = 2
        self.toplayer = nn.Conv2d(2048 * fold, 256, kernel_size=1, stride=1, padding=0) # Reduce channels
        # Lateral layers [C]
        self.latlayer1 = nn.Conv2d(1024 * fold, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512 * fold, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256 * fold, 256, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        # ps0
        self.smooth1_1 = nn.Conv2d(256 * fold, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256 * fold, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256 * fold, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256 * fold, 256, kernel_size=3, stride=1, padding=1)
        # ps1
        self.smooth1_2 = nn.Conv2d(256 * fold, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256 * fold, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256 * fold, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256 * fold, 128, kernel_size=3, stride=1, padding=1)
        # ps2 is concatenation
        # Classify layers
        self.smooth = nn.Conv2d(128*4*fold, 128*4, kernel_size=3, stride=1, padding=1)
        self.classify = nn.Conv2d(128*4, numClass, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), **self._up_kwargs) + y

    def forward(self, c2, c3, c4, c5, c2_ext, c3_ext, c4_ext, c5_ext, ps0_ext, ps1_ext, ps2_ext):

        # Top-down
        p5 = self.toplayer(torch.cat([c5] + [F.interpolate(c5_ext[0], size=c5.size()[2:], **self._up_kwargs)], dim=1))
        p4 = self._upsample_add(p5, self.latlayer1(torch.cat([c4] + [F.interpolate(c4_ext[0], size=c4.size()[2:], **self._up_kwargs)], dim=1)))
        p3 = self._upsample_add(p4, self.latlayer2(torch.cat([c3] + [F.interpolate(c3_ext[0], size=c3.size()[2:], **self._up_kwargs)], dim=1)))
        p2 = self._upsample_add(p3, self.latlayer3(torch.cat([c2] + [F.interpolate(c2_ext[0], size=c2.size()[2:], **self._up_kwargs)], dim=1)))
        ps0 = [p5, p4, p3, p2]
        
        # Smooth
        p5 = self.smooth1_1(torch.cat([p5] + [F.interpolate(ps0_ext[0][0], size=p5.size()[2:], **self._up_kwargs)], dim=1))
        p4 = self.smooth2_1(torch.cat([p4] + [F.interpolate(ps0_ext[1][0], size=p4.size()[2:], **self._up_kwargs)], dim=1))
        p3 = self.smooth3_1(torch.cat([p3] + [F.interpolate(ps0_ext[2][0], size=p3.size()[2:], **self._up_kwargs)], dim=1))
        p2 = self.smooth4_1(torch.cat([p2] + [F.interpolate(ps0_ext[3][0], size=p2.size()[2:], **self._up_kwargs)], dim=1))
        ps1 = [p5, p4, p3, p2]
        
        p5 = self.smooth1_2(torch.cat([p5] + [F.interpolate(ps1_ext[0][0], size=p5.size()[2:], **self._up_kwargs)], dim=1))
        p4 = self.smooth2_2(torch.cat([p4] + [F.interpolate(ps1_ext[1][0], size=p4.size()[2:], **self._up_kwargs)], dim=1))
        p3 = self.smooth3_2(torch.cat([p3] + [F.interpolate(ps1_ext[2][0], size=p3.size()[2:], **self._up_kwargs)], dim=1))
        p2 = self.smooth4_2(torch.cat([p2] + [F.interpolate(ps1_ext[3][0], size=p2.size()[2:], **self._up_kwargs)], dim=1))
        ps2 = [p5, p4, p3, p2]

        # Classify
        # use ps2_ext
        ps3 = self._concatenate(
                torch.cat([p5] + [F.interpolate(ps2_ext[0][0], size=p5.size()[2:], **self._up_kwargs)], dim=1), 
                torch.cat([p4] + [F.interpolate(ps2_ext[1][0], size=p4.size()[2:], **self._up_kwargs)], dim=1), 
                torch.cat([p3] + [F.interpolate(ps2_ext[2][0], size=p3.size()[2:], **self._up_kwargs)], dim=1), 
                torch.cat([p2] + [F.interpolate(ps2_ext[3][0], size=p2.size()[2:], **self._up_kwargs)], dim=1)
            )
        ps3 = self.smooth(ps3)
        output = self.classify(ps3)

        return output, ps0, ps1, ps2, ps3


class fpn(nn.Module):
    def __init__(self, numClass):
        super(fpn, self).__init__()
        self._up_kwargs = {'mode': 'bilinear'}
        # Res net
        self.resnet_global = resnet50(True)
        self.resnet_local = resnet50(True)

        # fpn module
        self.fpn_global = fpn_module_global(numClass)
        self.fpn_local = fpn_module_local(numClass)

        self.c2_g = None; self.c3_g = None; self.c4_g = None; self.c5_g = None; self.output_g = None
        self.ps0_g = None; self.ps1_g = None; self.ps2_g = None; self.ps3_g = None

        self.c2_l = []; self.c3_l = []; self.c4_l = []; self.c5_l = [];
        self.ps00_l = []; self.ps01_l = []; self.ps02_l = []; self.ps03_l = [];
        self.ps10_l = []; self.ps11_l = []; self.ps12_l = []; self.ps13_l = [];
        self.ps20_l = []; self.ps21_l = []; self.ps22_l = []; self.ps23_l = [];
        self.ps0_l = None; self.ps1_l = None; self.ps2_l = None
        self.ps3_l = []#; self.output_l = []

        self.c2_b = None; self.c3_b = None; self.c4_b = None; self.c5_b = None;
        self.ps00_b = None; self.ps01_b = None; self.ps02_b = None; self.ps03_b = None;
        self.ps10_b = None; self.ps11_b = None; self.ps12_b = None; self.ps13_b = None;
        self.ps20_b = None; self.ps21_b = None; self.ps22_b = None; self.ps23_b = None;
        self.ps3_b = []#; self.output_b = []

        self.patch_n = 0

        self.mse = nn.MSELoss()

        self.ensemble_conv = nn.Conv2d(128*4 * 2, numClass, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.ensemble_conv.weight, mean=0, std=0.01)

        # init fpn
        for m in self.fpn_global.children():
            if hasattr(m, 'weight'): nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): nn.init.constant_(m.bias, 0)
        for m in self.fpn_local.children():
            if hasattr(m, 'weight'): nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): nn.init.constant_(m.bias, 0)

    def clear_cache(self):
        self.c2_g = None; self.c3_g = None; self.c4_g = None; self.c5_g = None; self.output_g = None
        self.ps0_g = None; self.ps1_g = None; self.ps2_g = None; self.ps3_g = None

        self.c2_l = []; self.c3_l = []; self.c4_l = []; self.c5_l = [];
        self.ps00_l = []; self.ps01_l = []; self.ps02_l = []; self.ps03_l = [];
        self.ps10_l = []; self.ps11_l = []; self.ps12_l = []; self.ps13_l = [];
        self.ps20_l = []; self.ps21_l = []; self.ps22_l = []; self.ps23_l = [];
        self.ps0_l = None; self.ps1_l = None; self.ps2_l = None
        self.ps3_l = []; self.output_l = []

        self.c2_b = None; self.c3_b = None; self.c4_b = None; self.c5_b = None;
        self.ps00_b = None; self.ps01_b = None; self.ps02_b = None; self.ps03_b = None;
        self.ps10_b = None; self.ps11_b = None; self.ps12_b = None; self.ps13_b = None;
        self.ps20_b = None; self.ps21_b = None; self.ps22_b = None; self.ps23_b = None;
        self.ps3_b = []; self.output_b = []

        self.patch_n = 0


    def _sample_grid(self, fm, bbox, sampleSize):
        """
        :param fm: tensor(b,c,h,w) the global feature map
        :param bbox: list [b* nparray(x1, y1, x2, y2)] the (x1,y1) is the left_top of bbox, (x2, y2) is the right_bottom of bbox
        there are in range [0, 1]. x is corresponding to width dimension and y is corresponding to height dimension
        :param sampleSize: (oH, oW) the point to sample in height dimension and width dimension
        :return: tensor(b, c, oH, oW) sampled tensor
        """
        b, c, h, w = fm.shape
        b_bbox = len(bbox)
        bbox = [x*2 - 1 for x in bbox] # range transform
        if b != b_bbox and b == 1:
            fm = torch.cat([fm,]*b_bbox, dim=0)
        grid = np.zeros((b_bbox,) + sampleSize + (2,), dtype=np.float32)
        gridMap = np.array([[(cnt_w/(sampleSize[1]-1), cnt_h/(sampleSize[0]-1)) for cnt_w in range(sampleSize[1])] for cnt_h in range(sampleSize[0])])
        for cnt_b in range(b_bbox):
            grid[cnt_b, :, :, 0] = bbox[cnt_b][0] + (bbox[cnt_b][2] - bbox[cnt_b][0])*gridMap[:, :, 0]
            grid[cnt_b, :, :, 1] = bbox[cnt_b][1] + (bbox[cnt_b][3] - bbox[cnt_b][1])*gridMap[:, :, 1]
        grid = torch.from_numpy(grid).cuda()
        return F.grid_sample(fm, grid)

    def _crop_global(self, f_global, top_lefts, ratio):
        '''
        top_lefts: [(top, left)] * b
        '''
        _, c, H, W = f_global.size()
        b = len(top_lefts)
        h, w = int(np.round(H * ratio[0])), int(np.round(W * ratio[1]))

        # bbox = [ np.array([left, top, left + ratio, top + ratio]) for (top, left) in top_lefts ]
        # crop = self._sample_grid(f_global, bbox, (H, W))

        crop = []
        for i in range(b):
            top, left = int(np.round(top_lefts[i][0] * H)), int(np.round(top_lefts[i][1] * W))
            # # global's sub-region & upsample
            # f_global_patch = F.interpolate(f_global[0:1, :, top:top+h, left:left+w], size=(h, w), mode='bilinear')
            f_global_patch = f_global[0:1, :, top:top+h, left:left+w]
            crop.append(f_global_patch[0])
        crop = torch.stack(crop, dim=0) # stack into mini-batch
        return [crop] # return as a list for easy to torch.cat

    def _merge_local(self, f_local, merge, f_global, top_lefts, oped, ratio, template):
        '''
        merge feature maps from local patches, and finally to a whole image's feature map (on cuda)
        f_local: a sub_batch_size of patch's feature map
        oped: [start, end)
        '''
        b, _, _, _ = f_local.size()
        _, c, H, W = f_global.size() # match global feature size
        if merge is None:
            merge = torch.zeros((1, c, H, W)).cuda()
        h, w = int(np.round(H * ratio[0])), int(np.round(W * ratio[1]))
        for i in range(b):
            index = oped[0] + i
            top, left = int(np.round(H * top_lefts[index][0])), int(np.round(W * top_lefts[index][1]))
            merge[:, :, top:top+h, left:left+w] += F.interpolate(f_local[i:i+1], size=(h, w), **self._up_kwargs)
        if oped[1] >= len(top_lefts):
            template = F.interpolate(template, size=(H, W), **self._up_kwargs)
            template = template.expand_as(merge)
            # template = Variable(template).cuda()
            merge /= template
        return merge

    def ensemble(self, f_local, f_global):
        return self.ensemble_conv(torch.cat((f_local, f_global), dim=1))

    def collect_local_fm(self, image_global, patches, ratio, top_lefts, oped, batch_size, global_model=None, template=None, n_patch_all=None):
        '''
        patches: 1 patch
        top_lefts: all top-left
        oped: [start, end)
        '''
        with torch.no_grad():
            if self.patch_n == 0:
                self.c2_g, self.c3_g, self.c4_g, self.c5_g = global_model.module.resnet_global.forward(image_global)
                self.output_g, self.ps0_g, self.ps1_g, self.ps2_g, self.ps3_g = global_model.module.fpn_global.forward(self.c2_g, self.c3_g, self.c4_g, self.c5_g)
                # self.output_g = F.interpolate(self.output_g, image_global.size()[2:], mode='nearest')
            self.patch_n += patches.size()[0]
            self.patch_n %= n_patch_all

            self.resnet_local.eval()
            self.fpn_local.eval()
            c2, c3, c4, c5 = self.resnet_local.forward(patches)
            # global's 1x patch cat
            output, ps0, ps1, ps2, ps3 = self.fpn_local.forward(
                c2, c3, c4, c5,
                self._crop_global(self.c2_g, top_lefts[oped[0]:oped[1]], ratio),
                c3_ext=self._crop_global(self.c3_g, top_lefts[oped[0]:oped[1]], ratio),
                c4_ext=self._crop_global(self.c4_g, top_lefts[oped[0]:oped[1]], ratio),
                c5_ext=self._crop_global(self.c5_g, top_lefts[oped[0]:oped[1]], ratio),
                ps0_ext=[ self._crop_global(f, top_lefts[oped[0]:oped[1]], ratio) for f in self.ps0_g ],
                ps1_ext=[ self._crop_global(f, top_lefts[oped[0]:oped[1]], ratio) for f in self.ps1_g ],
                ps2_ext=[ self._crop_global(f, top_lefts[oped[0]:oped[1]], ratio) for f in self.ps2_g ]
            )
            # output = F.interpolate(output, patches.size()[2:], mode='nearest')

            self.c2_b = self._merge_local(c2, self.c2_b, self.c2_g, top_lefts, oped, ratio, template)
            self.c3_b = self._merge_local(c3, self.c3_b, self.c3_g, top_lefts, oped, ratio, template)
            self.c4_b = self._merge_local(c4, self.c4_b, self.c4_g, top_lefts, oped, ratio, template)
            self.c5_b = self._merge_local(c5, self.c5_b, self.c5_g, top_lefts, oped, ratio, template)

            self.ps00_b = self._merge_local(ps0[0], self.ps00_b, self.ps0_g[0], top_lefts, oped, ratio, template)
            self.ps01_b = self._merge_local(ps0[1], self.ps01_b, self.ps0_g[1], top_lefts, oped, ratio, template)
            self.ps02_b = self._merge_local(ps0[2], self.ps02_b, self.ps0_g[2], top_lefts, oped, ratio, template)
            self.ps03_b = self._merge_local(ps0[3], self.ps03_b, self.ps0_g[3], top_lefts, oped, ratio, template)
            self.ps10_b = self._merge_local(ps1[0], self.ps10_b, self.ps1_g[0], top_lefts, oped, ratio, template)
            self.ps11_b = self._merge_local(ps1[1], self.ps11_b, self.ps1_g[1], top_lefts, oped, ratio, template)
            self.ps12_b = self._merge_local(ps1[2], self.ps12_b, self.ps1_g[2], top_lefts, oped, ratio, template)
            self.ps13_b = self._merge_local(ps1[3], self.ps13_b, self.ps1_g[3], top_lefts, oped, ratio, template)
            self.ps20_b = self._merge_local(ps2[0], self.ps20_b, self.ps2_g[0], top_lefts, oped, ratio, template)
            self.ps21_b = self._merge_local(ps2[1], self.ps21_b, self.ps2_g[1], top_lefts, oped, ratio, template)
            self.ps22_b = self._merge_local(ps2[2], self.ps22_b, self.ps2_g[2], top_lefts, oped, ratio, template)
            self.ps23_b = self._merge_local(ps2[3], self.ps23_b, self.ps2_g[3], top_lefts, oped, ratio, template)

            self.ps3_b.append(ps3.cpu())
            # self.output_b.append(output.cpu()) # each output is 1, 7, h, w

            if self.patch_n == 0:
                # merged all patches into an image
                self.c2_l.append(self.c2_b); self.c3_l.append(self.c3_b); self.c4_l.append(self.c4_b); self.c5_l.append(self.c5_b);
                self.ps00_l.append(self.ps00_b); self.ps01_l.append(self.ps01_b); self.ps02_l.append(self.ps02_b); self.ps03_l.append(self.ps03_b)
                self.ps10_l.append(self.ps10_b); self.ps11_l.append(self.ps11_b); self.ps12_l.append(self.ps12_b); self.ps13_l.append(self.ps13_b)
                self.ps20_l.append(self.ps20_b); self.ps21_l.append(self.ps21_b); self.ps22_l.append(self.ps22_b); self.ps23_l.append(self.ps23_b)

                # collected all ps3 and output of patches as a (b) tensor, append into list
                self.ps3_l.append(torch.cat(self.ps3_b, dim=0)); # a list of tensors
                # self.output_l.append(torch.cat(self.output_b, dim=0)) # a list of 36, 7, h, w tensors

                self.c2_b = None; self.c3_b = None; self.c4_b = None; self.c5_b = None;
                self.ps00_b = None; self.ps01_b = None; self.ps02_b = None; self.ps03_b = None;
                self.ps10_b = None; self.ps11_b = None; self.ps12_b = None; self.ps13_b = None;
                self.ps20_b = None; self.ps21_b = None; self.ps22_b = None; self.ps23_b = None;
                self.ps3_b = []# ; self.output_b = []
            if len(self.c2_l) == batch_size:
                self.c2_l = torch.cat(self.c2_l, dim=0)# .cuda()
                self.c3_l = torch.cat(self.c3_l, dim=0)# .cuda()
                self.c4_l = torch.cat(self.c4_l, dim=0)# .cuda()
                self.c5_l = torch.cat(self.c5_l, dim=0)# .cuda()
                self.ps00_l = torch.cat(self.ps00_l, dim=0)# .cuda()
                self.ps01_l = torch.cat(self.ps01_l, dim=0)# .cuda()
                self.ps02_l = torch.cat(self.ps02_l, dim=0)# .cuda()
                self.ps03_l = torch.cat(self.ps03_l, dim=0)# .cuda()
                self.ps10_l = torch.cat(self.ps10_l, dim=0)# .cuda()
                self.ps11_l = torch.cat(self.ps11_l, dim=0)# .cuda()
                self.ps12_l = torch.cat(self.ps12_l, dim=0)# .cuda()
                self.ps13_l = torch.cat(self.ps13_l, dim=0)# .cuda()
                self.ps20_l = torch.cat(self.ps20_l, dim=0)# .cuda()
                self.ps21_l = torch.cat(self.ps21_l, dim=0)# .cuda()
                self.ps22_l = torch.cat(self.ps22_l, dim=0)# .cuda()
                self.ps23_l = torch.cat(self.ps23_l, dim=0)# .cuda()
                self.ps0_l = [self.ps00_l, self.ps01_l, self.ps02_l, self.ps03_l]
                self.ps1_l = [self.ps10_l, self.ps11_l, self.ps12_l, self.ps13_l]
                self.ps2_l = [self.ps20_l, self.ps21_l, self.ps22_l, self.ps23_l]
                # self.ps3_l = torch.cat(self.ps3_l, dim=0)# .cuda()
            return self.ps3_l, output# self.output_l


    def forward(self, image_global, patches, top_lefts, ratio, mode=1, global_model=None, n_patch=None):
        if mode == 1:
            # train global model
            c2_g, c3_g, c4_g, c5_g = self.resnet_global.forward(image_global)
            output_g, ps0_g, ps1_g, ps2_g, ps3_g = self.fpn_global.forward(c2_g, c3_g, c4_g, c5_g)
            # imsize = image_global.size()[2:]
            # output_g = F.interpolate(output_g, imsize, mode='nearest')
            return output_g, None
        elif mode == 2:
            # train global2local model
            with torch.no_grad():
                if self.patch_n == 0:
                    # calculate global images only if patches belong to a new set of global images (when self.patch_n % n_patch == 0)
                    self.c2_g, self.c3_g, self.c4_g, self.c5_g = self.resnet_global.forward(image_global)
                    self.output_g, self.ps0_g, self.ps1_g, self.ps2_g, self.ps3_g = self.fpn_global.forward(self.c2_g, self.c3_g, self.c4_g, self.c5_g)
                    # imsize_glb = image_global.size()[2:]
                    # self.output_g = F.interpolate(self.output_g, imsize_glb, mode='nearest')
                self.patch_n += patches.size()[0]
                self.patch_n %= n_patch

            # train local model #######################################
            c2_l, c3_l, c4_l, c5_l = self.resnet_local.forward(patches)
            # global's 1x patch cat
            output_l, ps0_l, ps1_l, ps2_l, ps3_l = self.fpn_local.forward(c2_l, c3_l, c4_l, c5_l,
                self._crop_global(self.c2_g, top_lefts, ratio),
                self._crop_global(self.c3_g, top_lefts, ratio),
                self._crop_global(self.c4_g, top_lefts, ratio),
                self._crop_global(self.c5_g, top_lefts, ratio),
                [ self._crop_global(f, top_lefts, ratio) for f in self.ps0_g ],
                [ self._crop_global(f, top_lefts, ratio) for f in self.ps1_g ],
                [ self._crop_global(f, top_lefts, ratio) for f in self.ps2_g ]
            )
            # imsize = patches.size()[2:]
            # output_l = F.interpolate(output_l, imsize, mode='nearest')
            ps3_g2l = self._crop_global(self.ps3_g, top_lefts, ratio)[0] # only calculate loss on 1x
            ps3_g2l = F.interpolate(ps3_g2l, size=ps3_l.size()[2:], **self._up_kwargs)

            output = self.ensemble(ps3_l, ps3_g2l)
            # output = F.interpolate(output, imsize, mode='nearest')
            return output, self.output_g, output_l, self.mse(ps3_l, ps3_g2l)
        else:
            # train local2global model
            c2_g, c3_g, c4_g, c5_g = self.resnet_global.forward(image_global)
            # local patch cat into global
            output_g, ps0_g, ps1_g, ps2_g, ps3_g = self.fpn_global.forward(c2_g, c3_g, c4_g, c5_g, c2_ext=self.c2_l, c3_ext=self.c3_l, c4_ext=self.c4_l, c5_ext=self.c5_l, ps0_ext=self.ps0_l, ps1_ext=self.ps1_l, ps2_ext=self.ps2_l)
            # imsize = image_global.size()[2:]
            # output_g = F.interpolate(output_g, imsize, mode='nearest')
            self.clear_cache()
            return output_g, ps3_g