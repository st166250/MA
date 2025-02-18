import os
from pathlib import Path
from glob import glob
import torch
import torchio as tio
import numpy as np
from tqdm import tqdm


class SimCLR2DDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, path_to_train_or_val_keys, transforms, validation=False, small_dataset = False):
        super().__init__()
        all_subjects = np.load(path_to_train_or_val_keys, allow_pickle=True)[()]
        self.slices, self.slices_path = SimCLR2DDataset._create_slices_list(Path(root_path), all_subjects, small_dataset)
        self.validation = validation
        self.transform = transforms
        print(len(self.slices))
        print(self.slices)
    
    @staticmethod
    def _create_slices_list(root_path, all_subjects, small_dataset):
        slices = []
        slices_paths = []
        count = 0
        for subject in tqdm(all_subjects):
            count += 1
            path = root_path/subject.name/"slices"
            slices_paths.append(path) #ToDo: change back!!!  ---DONE!!
            #slices_paths.append('None')
            try:
                slices.extend(list(path.glob("*.npy")))
            except FileNotFoundError:
                continue
            if count == 50 and small_dataset == True:
                return slices, slices_paths
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
            if str(subject) != "/mnt/qdata/rawdata/NAKO_IQA/NAKO_IQA_nifti/Q3" or suffix != "fb_deep_W_COMPOSED":  #dicom_3D_GRE_TRA_fb_deep_W_COMPOSED.nii under Q3 seems corrupted and leads to an error
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
    def __init__(self, root_path, path_to_train_or_val_keys, transforms, validation=False, small_dataset=False):
        super().__init__()
        all_subjects = np.load(path_to_train_or_val_keys, allow_pickle=True)[()]
        self.subjects, self.subjects_paths = SimCLR3DDataset._create_subject_list(Path(root_path), all_subjects, small_dataset)
        self.validation = validation
        self.transform = transforms
        self.small_dataset = small_dataset
    
    @staticmethod
    def _create_subject_list(root_path, all_subjects, small_dataset):
        count = 0
        subjects = []
        subjects_paths = []
        for subject in tqdm(all_subjects):
            count += 1
            path = root_path/subject.name/"wat.nii.gz"
            subjects_paths.append(path)
            try:
                tio_subject = tio.Subject(data=tio.ScalarImage(path))
                subjects.append(tio_subject)
                #print("tio_subject[data][data].shape: {}".format(tio_subject["data"]["data"].shape))
            except FileNotFoundError:
                continue
            if count == 50 and small_dataset == True:
                return subjects, subjects_paths

        return subjects, subjects_paths

        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self ,idx):

        if self.validation:
            return self.transform(self.subjects[idx])
        
        xi, xj, _ = self.transform(self.subjects[idx])
        return xi["data"]["data"], xj["data"]["data"], _
    

class ISMRM_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, transform, suffix):
        super().__init__()
        self.transform = transform
        path = Path(root_path)
        all_subjects = list(path.glob("*"))
        self.subjects, self.subjects_list = ISMRM_Dataset._create_subject_list(path ,all_subjects, suffix)
    
    @staticmethod
    def _create_subject_list(root_path, all_subjects, suffix):
        subjects = []
        subjects_paths = []
        for subject in tqdm(all_subjects):
            suffix_fin = str(subject.name) + "_t1_vibe_dixon_tra_nativ_" + suffix + ".nii.gz"
            path = subject/suffix_fin
            subjects_paths.append(path)

            try: 
                tio_subject = tio.Subject(data=tio.ScalarImage(path))
                subjects.append(tio_subject)
            except FileNotFoundError:
                print(path, "not found")
                continue

        return subjects, subjects_paths

    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        return self.transform(self.subjects[idx])

class SimCLR3DDatasetTo2D(torch.utils.data.Dataset):
    def __init__(self, root_path, path_to_train_or_val_keys, transforms, validation=False, small_dataset=False):
        super().__init__()
        all_subjects = np.load(path_to_train_or_val_keys, allow_pickle=True)[()]
        self.subjects, self.subjects_paths = SimCLR3DDatasetTo2D._create_subject_list(Path(root_path), all_subjects, small_dataset, validation)
        self.validation = validation
        self.transform = transforms
        self.small_dataset = small_dataset

    @staticmethod
    def _create_subject_list(root_path, all_subjects, small_dataset, validation):
        count = 0
        subjects = []
        subjects_paths = []
        for subject in tqdm(all_subjects):
            count += 1
            path = root_path/subject.name/"wat.nii.gz"
            subjects_paths.append(path)
            try:
                tio_subject = tio.Subject(data=tio.ScalarImage(path))
                subjects.append(tio_subject)
                #print("tio_subject[data][data].shape: {}".format(tio_subject["data"]["data"].shape))
            except FileNotFoundError:
                continue

            if count == 768 and small_dataset == True and validation == True:
                return subjects, subjects_paths
            if count == 768 and small_dataset == True:
                return subjects, subjects_paths

        return subjects, subjects_paths

        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self ,idx):
        subject = self.subjects[idx]
        volume = subject["data"]["data"][0].numpy()


        slice_idx = np.random.randint(volume.shape[2])
        #slice_idx = 100
        slice_img = volume[:,:,slice_idx]

        #print("idx: {}, slice_idx: {}".format(idx, slice_idx))

        if self.validation:
            return self.transform(slice_img)
        
        xi, xj, _ = self.transform(slice_img)
        return (xi, xj, _)

class SimCLR3DDataset_ForMotion(torch.utils.data.Dataset):
    def __init__(self, root_path, path_to_train_or_val_keys, transforms, validation=False, small_dataset=False, datatyp="wat.nii.gz"):
        super().__init__()
        all_subjects = np.load(path_to_train_or_val_keys, allow_pickle=True)[()]
        self.subjects, self.subjects_paths = SimCLR3DDataset_ForMotion._create_subject_list(Path(root_path), all_subjects, small_dataset, validation, datatyp)
        self.validation = validation
        self.transform = transforms
        self.small_dataset = small_dataset
        self.datatyp = datatyp


    @staticmethod
    def _create_subject_list(root_path, all_subjects, small_dataset, validation, datatyp):
        count = 0
        subjects = []
        subjects_paths = []
        for subject in tqdm(all_subjects):
            count += 1
            path = root_path/subject.name/datatyp
            subjects_paths.append(path)
            try:
                tio_subject = tio.Subject(data=tio.ScalarImage(path))
                subjects.append(tio_subject)
                #print("tio_subject[data][data].shape: {}".format(tio_subject["data"]["data"].shape))
            except FileNotFoundError:
                continue

            if count == 1024 and small_dataset == True and validation == True: #576
                return subjects, subjects_paths
            if count == 7680 and small_dataset == True:                       #1984
                return subjects, subjects_paths

        return subjects, subjects_paths

        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self ,idx):
        subject = self.subjects[idx]
        volume = subject["data"]["data"][0].numpy()

        if self.validation:
            return self.transform(volume, idx)
        
        xi, xj, x_z = self.transform(volume, idx)

        return xi, xj, x_z