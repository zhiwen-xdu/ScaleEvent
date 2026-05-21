import torch
import random
import sys
sys.path.append("/home/zhiyu/projects/DINOv3")

from scale_event.dataset import Cityscape,Decd,Kitti,Waymo,Dsec,GoPro,M3ED,MVSEC,RESEG,DDD17


class DistillDataset(torch.utils.data.Dataset):
    def __init__(self,W=640,H=480,S=1,samples_per_epoch=50000,p_datasets=None):
        self.datasets = []
        self.datasets.append(Cityscape(W=W,H=H,S=S))
        self.datasets.append(Decd(W=W,H=H,S=S))
        self.datasets.append(Kitti(W=W,H=H,S=S))
        self.datasets.append(Waymo(W=W,H=H,S=S))
        self.datasets.append(Dsec(W=W,H=H,S=S))
        self.datasets.append(GoPro(W=W,H=H,S=S))
        self.datasets.append(M3ED(W=W, H=H, S=S))
        self.datasets.append(MVSEC(W=W, H=H, S=S))
        self.datasets.append(RESEG(W=W, H=H, S=S))
        self.datasets.append(DDD17(W=W, H=H, S=S))

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]
        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        seq_id = random.randint(0, dataset._get_sequences_num() - 1)
        frame_id = random.randint(0, dataset.sequence_lens[seq_id] - 1)
        images,events,masks = dataset._get_frame(seq_id, frame_id)
        return images,events,masks  # [3,H,W],[3,H,W],[1,H/P,W/P]


#  数据增强方法待定
if __name__ == "__main__":
    dataset = DistillDataset()
    dataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    for i,(images,events,masks) in enumerate(dataLoader):
        print(images.shape,events.shape,masks.shape,torch.mean(masks))

