"""
 If you use this code, please cite the following paper:
 Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith Punnappurath, and Michael S Brown.
 CIE XYZ Net: Unprocessing Images for Low-Level Computer Vision Tasks. arXiv preprint, 2020.
"""

__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]


from .local_net import *
from .global_net import *

class CIEXYZNet(nn.Module):
    def __init__(self, device='cuda', localdepth=16, local_convdepth=32, globaldepth=5, global_convdepth=64,
                 global_in=128, scale=0.25):
        super(CIEXYZNet, self).__init__()
        self.localdepth = localdepth
        self.local_convdepth = local_convdepth
        self.globaldepth = globaldepth
        self.global_convdepth = global_convdepth
        self.global_in = global_in
        self.scale = scale
        self.device = device
        self.srgb2xyz_local_net = localSubNet(blockDepth=self.localdepth, convDepth=self.local_convdepth, scale=self.scale)
        self.xyz2srgb_local_net = localSubNet(blockDepth=self.localdepth, convDepth=self.local_convdepth, scale=self.scale)
        self.srgb2xyz_globa_net = globalSubNet(blockDepth=self.globaldepth, convDepth=self.global_convdepth,
                                               in_img_sz=self.global_in, device=self.device)
        self.xyz2srgb_globa_net = globalSubNet(blockDepth=self.globaldepth, convDepth=self.global_convdepth,
                                               in_img_sz=self.global_in, device=self.device)

    def forward_local(self, x, target):
        if target == "xyz":
            localLayer = self.srgb2xyz_local_net(x)
        elif target == 'srgb':
            localLayer = self.xyz2srgb_local_net(x)
        else:
            raise Exception("Wrong target. It is expected to be srgb or xyz, but the input target is %s\n" % target)
        return localLayer

    def forward_global(self, x, target):
        if target == "xyz":
            m_v = self.srgb2xyz_globa_net(x)
        elif target == "srgb":
            m_v = self.xyz2srgb_globa_net(x)
        else:
            raise Exception("Wrong target. It is expected to be srgb or xyz, but the input target is %s\n" % target)
        m = torch.reshape(m_v, (x.size(0), 6, 3))
        # multiply
        y = x.clone()
        for i in range(m.size(0)):
            temp = torch.mm(self.kernel(torch.reshape(torch.squeeze(x[i, :, :, :]), (-1, 3))),
                            torch.squeeze(m[i, :, :]))
            y[i, :, :, :] = torch.reshape(temp, (x.size(1), x.size(2), x.size(3)))

        return y

    def forward_srgb2xyz(self, srgb):
        l_xyz = srgb - self.forward_local(srgb, target='xyz')
        xyz = self.forward_global(l_xyz, target='xyz')
        return xyz

    def forward_xyz2srgb(self, xyz):
        g_srgb = self.forward_global(xyz, target='srgb')
        srgb = g_srgb + self.forward_local(g_srgb, target='srgb')
        return srgb

    def forward(self, x):
        xyz = self.forward_srgb2xyz(x)
        srgb = self.forward_xyz2srgb(xyz)
        return xyz, srgb

    @staticmethod
    def kernel(x):
        return torch.cat((x, x * x), dim=1)
