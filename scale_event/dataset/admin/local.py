class EnvironmentSettings:
    def __init__(self):
        self.cityscape_dir = '.../Cityscapes'  # Needs modification
        self.waymo_dir = '.../Waymo'       # Needs modification
        self.kitti_dir = '.../KITTI'   # Needs modification
        self.decd_dir = '.../DECD'    # Needs modification
        self.dsec_dir = '.../DSEC'    # Needs modification
        self.gopro_dir = '.../GoPro'   # Needs modification
        self.m3ed_dir = '.../M3ED'    # Needs modification
        self.mvsec_dir = '.../MVSEC'  # Needs modification
        self.reseg_dir = '.../RESEG'  # Needs modification
        self.ddd17_dir = '.../DDD17'  # Needs modification


def env_settings():
    return EnvironmentSettings()