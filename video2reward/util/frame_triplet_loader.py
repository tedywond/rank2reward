import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class Frameloader(Dataset):
    ''' Dataset class for loading expert trajectories.
    '''
    def __init__(self, 
                 root_dir,
                 test=False,
                 normalize_trajectory=True,
                 transforms=None,
                 subset_len=4,
                 randomized=False,
                 min_frames_between=1):
        
        dataset = 'test' if test else 'train'
        self.path = os.path.join(root_dir, f'*/{dataset}/*')
        self.ds = glob.glob(self.path)
        self.normalize_trajectory = normalize_trajectory
        self.transforms = transforms
        self.subset_len = subset_len
        self.expert_traj_lens = {i: len(glob.glob(self.ds[i] + '/*.png')) for i in range(len(self.ds))} 
        self.randomized = randomized
        self.min_frames_between = min_frames_between

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        # Randomly select start and end frames of the trajectory
        if self.randomized:
            init_frame = np.random.randint(0, self.expert_traj_lens[index] - self.min_frames_between, self.subset_len)
            goal_frame = np.random.randint(init_frame + self.min_frames_between, self.expert_traj_lens[index], self.subset_len)
            mid_frame = np.random.randint(init_frame, goal_frame + 1)
        # Fix start and end frames of the trajectory as the first and last frames the dataset
        else:
            init_frame = np.array([0 for _ in range(self.subset_len)])
            goal_frame = np.array([self.expert_traj_lens[index] - 1 for _ in range(self.subset_len)])
            mid_frame = np.random.randint(init_frame, goal_frame + 1)

        # Length of the trajectory
        delta_goal_init =  goal_frame - init_frame   
        # Relative position of the mid frame in the trajectory from 0 - 1
        relative_position = (mid_frame - init_frame) / delta_goal_init
        
        # Get batch of frames, applying transforms if given
        all_frames = []
        for i in range(self.subset_len):
            if self.transforms:
                cur = torch.vstack(
                    (self.transforms(read_image(os.path.join(self.ds[index], str(init_frame[i]) + '.png'))),
                     self.transforms(read_image(os.path.join(self.ds[index], str(mid_frame[i]) + '.png'))),
                     self.transforms(read_image(os.path.join(self.ds[index], str(goal_frame[i]) + '.png')))))
            else:
                cur = torch.vstack(
                    (read_image(os.path.join(self.ds[index], str(init_frame[i]) + '.png')),
                     read_image(os.path.join(self.ds[index], str(mid_frame[i]) + '.png')),
                     read_image(os.path.join(self.ds[index], str(goal_frame[i]) + '.png'))))
            all_frames.append(cur)       
        
        # Return the frames and their relative position in the trajectory
        if self.normalize_trajectory:
            return torch.stack(all_frames, axis=0), relative_position, delta_goal_init
        # Return the frames and their absolute position in the trajectory
        return torch.stack(all_frames, axis=0), np.round(relative_position * delta_goal_init), np.ones(self.subset_len)
