class EnvironmentSettings:
    def __init__(self):
        self.cityscape_dir = '/home/zhiyu/data/VFM2Event/Cityscapes'
        self.waymo_dir = '/home/zhiyu/data/VFM2Event/Waymo'
        self.kitti_dir = '/home/zhiyu/data/VFM2Event/KITTI'
        self.decd_dir = '/home/zhiyu/data/VFM2Event/DECD'
        self.dsec_dir = '/home/zhiyu/data/VFM2Event/DSEC'
        self.gopro_dir = '/home/zhiyu/data/VFM2Event/GoPro'
        self.m3ed_dir = '/home/zhiyu/data/VFM2Event/M3ED'
        self.mvsec_dir = '/home/zhiyu/data/VFM2Event/MVSEC'
        self.reseg_dir = '/home/zhiyu/data/VFM2Event/RESEG'
        self.ddd17_dir = '/home/zhiyu/data/VFM2Event/DDD17'
        self.ddd17_seg_dir = '/home/zhiyu/data/VFM2Event/DDD17_SEG_704x416'  #DDD17_SEG_384x224;DDD17_SEG_480x288;DDD17_SEG_544x320;DDD17_SEG_688x400;DDD17_SEG_704x416,DDD17_SEG_800x480,
        self.dsec_seg_dir = '/home/zhiyu/data/VFM2Event/DSEC_SEG_10000'


def env_settings():
    return EnvironmentSettings()