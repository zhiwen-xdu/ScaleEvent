from lib.models.vipt import build_siamtrack_dropmae
from thop import profile
import torch


cfg_file = "/media/kw20193/sdc1/Work_4/SIAMTrack/experiments/vipt/droptrack_rgbe.yaml"
config_module = importlib.import_module("lib.config.vipt.config")
cfg = config_module.cfg
config_module.update_config_from_file(cfg_file)

net = build_siamtrack_dropmae(cfg)
device = torch.device("cuda:0")
net.to(device)
template = torch.randn(1,6,192,192).to(device)
search = torch.randn(1,6,384,384).to(device)

macs, params = profile(net.to(device), inputs=(template,search))
print('Params: ', params/(1e+6))
print('GFLOPs: ', macs*2/(1e+9))   # 一般来讲，FLOPs是macs的两倍


