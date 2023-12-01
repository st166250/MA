import os
from pathlib import Path
from glob import glob
import torch
import torchio as tio
import numpy as np
from tqdm import tqdm


class SimCLR2DDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, path_to_train_or_val_keys, transforms, validation=False):
        super().__init__()
        all_subjects = np.load(path_to_train_or_val_keys, allow_pickle=True)[()]
        self.slices, self.slices_path = SimCLR2DDataset._create_slices_list(Path(root_path), all_subjects)
        self.validation = validation
        self.transform = transforms
    
    @staticmethod
    def _create_slices_list(root_path, all_subjects):
        slices = []
        slices_paths = []
        for subject in tqdm(all_subjects):
            path = root_path/subject.name/"slices"
            #slices_paths.append(path) ToDo: change back!!!
            slices_paths.append('None')
            try:
                slices.extend(list(path.glob("*.npy")))
            except FileNotFoundError:
                continue
        return slices, slices_paths

        
    def __len__(self):
        return len(self.slices)
    
    def load(self, idx):
        slice =  np.load(self.slices[idx]).astype(np.float32)
        slice = np.pad(slice, ((0,0), (28,28)))
        return slice
    
    def __getitem__(self ,idx):
        
        slice = self.load(idx)
        if self.validation:
            return self.transform(slice)
        
        xi, xj, _ = self.transform(slice)
        return (xi, xj, _)


class NakoIQADataset(torch.utils.data.Dataset):
    def __init__(self, root_path, transforms,  suffix="fb_W_COMPOSED"):
        super().__init__()
        root_path = Path(root_path)
        all_subjects = list(root_path.glob("*"))
        self.subjects, self.subjects_paths = NakoIQADataset._create_subject_list(root_path, all_subjects, suffix)
        self.transform = transforms
    
    @staticmethod
    def _create_subject_list(root_path, all_subjects,suffix):
        subjects = []
        subjects_paths = []
        for subject in tqdm(all_subjects):
            path = root_path/subject/"dixon"/"dicom_3D_GRE_TRA"
            path = str(path)+"_"+suffix+".nii"
            
            subjects_paths.append(path)
            try:
                tio_subject = tio.Subject(data=tio.ScalarImage(path))
                subjects.append(tio_subject)
            except FileNotFoundError as e:
                print(e)
                continue
        return subjects, subjects_paths

        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self ,idx):

        return self.transform(self.subjects[idx])


class NRUDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, transforms,  suffix="01_T1w", infix='_acq-mpragepmcoff_rec-wore_run-'):
        super().__init__()
        all_subjects = glob(f'{root_path}/sub-*')
        self.subjects, self.subjects_paths = NRUDataset._create_subject_list(root_path, all_subjects, infix, suffix)
        self.transform = transforms
    
    @staticmethod
    def _create_subject_list(root_path, all_subjects, infix, suffix):
        subjects = []
        subjects_paths = []
        for subject in tqdm(all_subjects):
            path = f'{subject}/anat/{subject[-6:]}'
            path = str(path)+ infix + suffix + ".nii"
            
            subjects_paths.append(path)
            try:
                tio_subject = tio.Subject(data=tio.ScalarImage(path))
                subjects.append(tio_subject)
            except FileNotFoundError as e:
                print(e)
                continue
        return subjects, subjects_paths

        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self ,idx):

        return self.transform(self.subjects[idx])
    
        

class SimCLR3DDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, path_to_train_or_val_keys, transforms, validation=False):
        super().__init__()
        all_subjects = np.load(path_to_train_or_val_keys, allow_pickle=True)[()]
        self.subjects, self.subjects_paths = SimCLR3DDataset._create_subject_list(Path(root_path), all_subjects)
        self.validation = validation
        self.transform = transforms
    
    @staticmethod
    def _create_subject_list(root_path, all_subjects):
        subjects = []
        subjects_paths = []
        for subject in tqdm(all_subjects):
            path = root_path/subject.name/"wat.nii.gz"
            subjects_paths.append(path)
            try:
                tio_subject = tio.Subject(data=tio.ScalarImage(path))
                subjects.append(tio_subject)
            except FileNotFoundError:
                continue
        return subjects, subjects_paths

        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self ,idx):

        if self.validation:
            return self.transform(self.subjects[idx])
        
        xi, xj, _ = self.transform(self.subjects[idx])
        return xi["data"]["data"], xj["data"]["data"], _