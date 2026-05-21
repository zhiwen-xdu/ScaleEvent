import os
import os.path
import torch
import random
import cv2
import numpy as np

from scale_event.dataset.admin import env_settings
from scale_event.dataset.event_utils import prepare_data,prepare_semantic_label,to_tensor

from PIL import Image
from torchvision import tv_tensors


class DSEC_SEG_EOMT_TEST(torch.utils.data.Dataset):
    def __init__(self,split="train",W=640,H=448):
        self.root = env_settings().dsec_seg_dir
        self.split = split
        self.W = W
        self.H = H

        if self.split == "train":
            file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_specs', 'dsec_seg_train_v2.txt')
            with open(file_path) as f:
                seq_names = [line.strip() for line in f.readlines()]
            self.sequence_list = [i for i in seq_names]

            len_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_specs', 'dsec_seg_train_seqlen_v2.txt')
            with open(len_file_path) as f:
                seq_lens = [line.strip() for line in f.readlines()]
            self.sequence_lens = [int(i) for i in seq_lens]

        elif self.split == "test":
            file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_specs', 'dsec_seg_test_v2.txt')
            with open(file_path) as f:
                seq_names = [line.strip() for line in f.readlines()]
            self.sequence_list = [i for i in seq_names]

            len_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_specs','dsec_seg_test_seqlen_v2.txt')
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
        event_path = os.path.join(seq_path, 'voxel','{:05}.jpg'.format(frame_id+1))
        mask_path = os.path.join(seq_path, 'mask', '{:05}.png'.format(frame_id+1))
        return event_path,mask_path

    def _get_data(self, seq_id, frame_id):
        ''' Return:event frame '''
        event_path,mask_path = self._get_data_path(seq_id, frame_id)
        # [H,W,3]
        event = prepare_data(event_path, W=self.W, H=self.H)
        # [3,H,W]-Tensor-[0~1]-Normalized
        event = to_tensor(event)

        # 11 classes: [0,1,2,3,4,5,6,7,8,9,10,255]
        seg_masks = prepare_semantic_label(mask_path,W=self.W,H=self.H)
        seg_masks = np.array(seg_masks)
        # [H,W]
        seg_masks = torch.from_numpy(seg_masks).long()

        # [1,H,W]
        semantic_masks = Image.open(mask_path)
        semantic_masks = semantic_masks.resize((self.W,self.H), Image.NEAREST)
        target = tv_tensors.Mask(semantic_masks, dtype=torch.long)
        masks, labels = self.target_parser(target)

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),  # [C,H,W]: False/True, C表示样本中有的类别数，而不是全部类别
            "labels": torch.tensor(labels),                # [C]: [0,1,2,3,4,5,6,7,8,9,10]
            "seg_masks": seg_masks
        }

        return event,target


    def target_parser(self,target):
        masks, labels = [], []
        # for label_id in target[0].unique():
        for label_id in range(11):
            masks.append(target[0] == label_id)
            labels.append(label_id)

        return masks, labels


    def __len__(self):
        return sum(self.sequence_lens)  # sum(self.sequence_lens)*5

    def __getitem__(self, index):
        # args: index (int): Index (Ignored since we sample randomly)
        seq_id = random.randint(0, self._get_sequences_num() - 1)
        frame_id = random.randint(0, self.sequence_lens[seq_id] - 1)
        event,target = self._get_data(seq_id, frame_id)
        return event,target  # [3,H,W],[H,W]


if __name__ == "__main__":
    dataset = DSEC_SEG_EOMT_TEST(split="train")
    dataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    for events,targets in dataLoader:
        print(events.shape,targets)
