import os
import os.path
import torch
import random
import cv2
import numpy as np

from scale_event.dataset.admin import env_settings
from scale_event.dataset.event_utils import prepare_data,to_tensor


class DDD17_SEG_TRAIN(torch.utils.data.Dataset):
    def __init__(self,split="train",W=704,H=416):
        self.root = env_settings().ddd17_seg_dir
        self.split = split
        self.W = W
        self.H = H

        if self.split == "train":
            file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_specs', 'ddd17_seg_train.txt')
            with open(file_path) as f:
                seq_names = [line.strip() for line in f.readlines()]
            self.sequence_list = [i for i in seq_names]

            len_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_specs', 'ddd17_seg_train_seqlen.txt')
            with open(len_file_path) as f:
                seq_lens = [line.strip() for line in f.readlines()]
            self.sequence_lens = [int(i) for i in seq_lens]

        elif self.split == "test":
            file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_specs', 'ddd17_seg_test_v2.txt')
            with open(file_path) as f:
                seq_names = [line.strip() for line in f.readlines()]
            self.sequence_list = [i for i in seq_names]

            len_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_specs','ddd17_seg_test_seqlen_v2.txt')
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
        # voxel_50ms, voxel_adaptive_256, voxel_adaptive_320, voxel_adaptive_360
        event_path = os.path.join(seq_path, 'voxel_adaptive_256','{:05}.jpg'.format(frame_id+1))
        mask_path = os.path.join(seq_path, 'mask', '{:05}.png'.format(frame_id + 1))
        return event_path,mask_path

    def _get_data(self, seq_id, frame_id):
        ''' Return:event frame '''
        event_path,mask_path = self._get_data_path(seq_id, frame_id)
        # [H,W,3]
        event = prepare_data(event_path, W=self.W, H=self.H)
        # [3,H,W]-Tensor-[0~1]-Normalized
        event = to_tensor(event)

        # 6 classes: [0,1,2,3,4,5]
        seg_mask = cv2.imread(mask_path, 0)
        seg_mask = np.array(seg_mask)
        # [H,W]
        seg_mask = torch.from_numpy(seg_mask).long()

        return event,seg_mask


    def __len__(self):
        return sum(self.sequence_lens)*5  #

    def __getitem__(self, index):
        # args: index (int): Index (Ignored since we sample randomly)
        seq_id = random.randint(0, self._get_sequences_num() - 1)
        frame_id = random.randint(0, self.sequence_lens[seq_id] - 1)
        event,seg_mask = self._get_data(seq_id, frame_id)
        return event,seg_mask  # [3,H,W],[H,W]


if __name__ == "__main__":
    dataset = DDD17_SEG_TRAIN(split="train")
    dataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    for events,seg_masks in dataLoader:
        print(events.shape,seg_masks.shape,seg_masks)
