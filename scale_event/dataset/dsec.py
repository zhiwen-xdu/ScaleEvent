import os
import os.path
import torch
import random

import sys
sys.path.append("/home/zhiyu/projects/DINOv3")

from scale_event.dataset.admin import env_settings
from scale_event.dataset.event_utils import prepare_data,prepare_mask,to_tensor


class Dsec(torch.utils.data.Dataset):
    def __init__(self,W=640,H=480,S=1):
        self.root = env_settings().dsec_dir
        self.W = W
        self.H = H
        self.S = S

        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_specs', 'dsec.txt')
        with open(file_path) as f:
            seq_names = [line.strip() for line in f.readlines()]
        self.sequence_list = [i for i in seq_names]

        len_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_specs', 'dsec_seqlen.txt')
        with open(len_file_path) as f:
            seq_lens = [line.strip() for line in f.readlines()]
        self.sequence_lens = [int(i) for i in seq_lens]

    def _get_sequences_num(self):
        return len(self.sequence_list)

    def _get_sequences_len(self,):
        return len(self.sequence_list)

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name)

    def _get_data_path(self, seq_id, frame_id):
        ''' return img+event path '''
        seq_path = self._get_sequence_path(seq_id)
        img_path = os.path.join(seq_path, 'image','{:05}.jpg'.format(frame_id+1))
        event_path = os.path.join(seq_path, 'voxel','{:05}.jpg'.format(frame_id+1))
        evimg_path = os.path.join(seq_path, 'evimg', '{:05}.jpg'.format(frame_id+1))
        return img_path, event_path, evimg_path

    def _get_frame(self, seq_id, frame_id):
        ''' Return:img+event frame '''
        img_path, event_path, evimg_path = self._get_data_path(seq_id, frame_id)
        # [H,W,6]:[H,W,3]+[H,W,3]
        images = prepare_data(img_path, W=self.W*self.S, H=self.H*self.S)
        events = prepare_data(event_path, W=self.W*self.S, H=self.H*self.S)
        masks = prepare_mask(evimg_path, W=self.W*self.S, H=self.H*self.S)
        # [3,H,W]-Tensor-[0~1]-Normalized
        images = to_tensor(images)
        events = to_tensor(events)

        return images, events, masks

    def __len__(self):
        return sum(self.sequence_lens)  # 95533

    def __getitem__(self, index):
        # args: index (int): Index (Ignored since we sample randomly)
        seq_id = random.randint(0, self._get_sequences_num() - 1)
        frame_id = random.randint(0, self.sequence_lens[seq_id] - 1)
        images, events, masks = self._get_frame(seq_id, frame_id)
        return images, events, masks  # [3,H,W],[3,H,W],[1,H/P,W/P]


if __name__ == "__main__":
    dataset = Dsec()
    dataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    for images,events,masks in dataLoader:
        print(images.shape,events.shape,masks.shape,torch.mean(masks))

