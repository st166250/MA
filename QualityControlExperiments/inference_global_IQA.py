import sys
sys.path.insert(0, "/home/students/studhoene1/imagequality/")

from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision
import torchio as tio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import nibabel as nib
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
print(torch.__version__)

from torch import Tensor, nn
import math
from simclr.dataset import SimCLR3DDatasetTo2D, SimCLR2DDataset
from torch.utils.data import DataLoader


from simclr.simclr_module2 import SimCLR
#from simclr.simclr_module import SimCLR

from simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform
from simclr.dataset import SimCLR3DDataset, NakoIQADataset, NRUDataset

preprocessings = tio.transforms.Compose([
            tio.transforms.ZNormalization(),
            #tio.RescaleIntensity((0, 1))
        ])
preprocessings2D = tio.transforms.Compose([
            tio.RescaleIntensity((0.0, 1.0))
        ])
# UKB
val_dataset = SimCLR3DDataset('/mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw/', "/home/students/studhoene1/imagequality/ukb_abdominal_val_keys.npy",
                    preprocessings, True)

# NAKO
nako_iqa_dataset = NakoIQADataset("/mnt/qdata/rawdata/NAKO_IQA/NAKO_IQA_nifti/", preprocessings, suffix="bh_W_COMPOSED")
nako_iqa_dataset_deep = NakoIQADataset("/mnt/qdata/rawdata/NAKO_IQA/NAKO_IQA_nifti/", preprocessings, suffix="fb_deep_W_COMPOSED")

# NRU
# infixes: T1 STIR: _acq-t1tirmpmcoff_rec-wore_run-, T2 TSE: _acq-t2tsepmcoff_rec-wore_run-, 
# T1 MPRAGE: _acq-mpragepmcoff_rec-wore_run-, FLAIR: _acq-flairpmcoff_rec-wore_run-, 
# T2 STAR: _acq-t2starpmcoff_rec-wore_run-
nru_dataset_still = NRUDataset("/home/raecker1/data/NRU/", preprocessings, suffix="01_T1w", infix='_acq-mpragepmcoff_rec-wore_run-')
nru_dataset_nod = NRUDataset("/home/raecker1/data/NRU/", preprocessings, suffix="02_T1w", infix='_acq-mpragepmcoff_rec-wore_run-')
nru_dataset_shake = NRUDataset("/home/raecker1/data/NRU/", preprocessings, suffix="03_T1w", infix = '_acq-mpragepmcoff_rec-wore_run-')


def normalize_lists(lst1, lst2):
    min1 = min(lst1)
    max1 = max(lst1)
    min2 = min(lst2)
    max2 = max(lst2)
    min_ges = min(min1, min2)
    max_ges = max(max1, max2)

    return [(x - min_ges) / (max_ges - min_ges) for x in lst1], [(x - min_ges) / (max_ges - min_ges) for x in lst2]

def load_slice(subj_idx, idx, noise=None):
    if noise:
        slc = noise(val_dataset[subj_idx])["data"]["data"][0,:,:,idx]
    else:
        slc = val_dataset[subj_idx]["data"]["data"][0,:,:,idx]
    slc = preprocessings2D(slc.unsqueeze(0).unsqueeze(-1))
    #slc = slc.squeeze(0).squeeze(-1)

    #slc = np.pad(slc, ((0,0), (28,28)))
    #slc = torch.from_numpy(slc)
    return slc.squeeze(-1)

def load_slice_nako(subj_idx, idx, noise=None):
    if noise:
        slc = noise(nako_iqa_dataset[subj_idx])["data"]["data"][0,20:-20,:224,idx].unsqueeze(0)
    else:
        slc = nako_iqa_dataset[subj_idx]["data"]["data"][0,20:-20,:224,idx].unsqueeze(0)
    slc = preprocessings2D(slc.unsqueeze(-1))
    slc = slc.squeeze(-1)
    # #slc = torchvision.transforms.functional.resize(slc, [224, 224])
    # slc = np.pad(slc, ((0,0), (28,28)), mode='edge')
    # slc = torch.from_numpy(slc).unsqueeze(0)
    # #slice = torch.from_numpy(slice)
    return slc

def load_slice_nako_deep(subj_idx, idx, noise=None):
    if noise:
        slc = noise(nako_iqa_dataset_deep[subj_idx])["data"]["data"][0,20:-20,:224,idx].unsqueeze(0)
    else:
        slc = nako_iqa_dataset_deep[subj_idx]["data"]["data"][0,20:-20,:224,idx].unsqueeze(0)
    slc = preprocessings2D(slc.unsqueeze(-1))
    slc = slc.squeeze(-1)
    # #slc = torchvision.transforms.functional.resize(slc, [224, 224])
    # slc = np.pad(slc, ((0,0), (28,28)), mode='edge')
    # slc = torch.from_numpy(slc).unsqueeze(0)
    return slc

def load_slice_nru_still(subj_idx, idx=None, pos=None, noise=None):
    if noise:
        if pos == 'middle':
            print('using middle slice')
            data = noise(nru_dataset_still[subj_idx])["data"]["data"][0,:,:,:]
            slc = data[:,:,data.shape[2]//2].unsqueeze(0)
        elif idx:
            print(f'using id {idx}')
            slc = noise(nru_dataset_still[subj_idx])["data"]["data"][0,:,:,idx].unsqueeze(0)
        else:
            raise ValueError("Please specify idx or pos")
    else:
        if pos == 'middle':
            print('using middle slice')
            data = nru_dataset_still[subj_idx]["data"]["data"][0,:,:,:]
            slc = data[:,:,data.shape[2]//2].unsqueeze(0)
        elif idx:
            print(f'using id {idx}')
            slc = nru_dataset_still[subj_idx]["data"]["data"][0,:,:,idx].unsqueeze(0)
        else:
            raise ValueError("Please specify idx or pos")
    return slc


def load_slice_nru_shake(subj_idx, idx=None, pos=None, noise=None):
    if noise:
        if pos == 'middle':
            print('using middle slice')
            data = noise(nru_dataset_shake[subj_idx])["data"]["data"][0,:,:,:]
            slc = data[:,:,data.shape[2]//2].unsqueeze(0)
        elif idx:
            print(f'using id {idx}')
            slc = noise(nru_dataset_shake[subj_idx])["data"]["data"][0,:,:,idx].unsqueeze(0)
        else:
            raise ValueError("Please specify idx [int] or pos ['middle']")
    else:
        if pos == 'middle':
            print('using middle slice')
            data = nru_dataset_shake[subj_idx]["data"]["data"][0,:,:,:]
            slc = data[:,:,data.shape[2]//2].unsqueeze(0)
        elif idx:
            print(f'using id {idx}')
            slc = nru_dataset_shake[subj_idx]["data"]["data"][0,:,:,idx].unsqueeze(0)
        else:
            raise ValueError("Please specify idx or pos")
    return slc


def load_slice_nru_nod(subj_idx, idx=None, pos=None, noise=None):
    if noise:
        if pos == 'middle':
            print('using middle slice')
            data = noise(nru_dataset_nod[subj_idx])["data"]["data"][0,:,:,:]
            slc = data[:,:,data.shape[2]//2].unsqueeze(0)
        elif idx:
            print(f'using id {idx}')
            slc = noise(nru_dataset_nod[subj_idx])["data"]["data"][0,:,:,idx].unsqueeze(0)
        else:
            raise ValueError("Please specify idx or pos")
    else:
        if pos == 'middle':
            print('using middle slice')
            data = nru_dataset_nod[subj_idx]["data"]["data"][0,:,:,:]
            slc = data[:,:,data.shape[2]//2].unsqueeze(0)
        elif idx:
            print(f'using id {idx}')
            slc = nru_dataset_nod[subj_idx]["data"]["data"][0,:,:,idx].unsqueeze(0)
        else:
            raise ValueError("Please specify idx or pos")
    return slc

#Experiment UKB (20 reference scans, similar height+weight)
def exp_UKB(): 
    test_id = 25
    slc = 250
    reference_keys = pd.read_csv('/home/students/studhoene1/imagequality/QualityControlExperiments/ref_keys.csv')
    root_path = '/mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw/'
    ref_idx = []
    for subject in reference_keys['eid']:
        path = Path(root_path)/str(subject)/"wat.nii.gz"
        ref_idx.append(val_dataset.subjects_paths.index(path))
    #noise = tio.Compose([tio.transforms.RandomNoise(mean=0, std=(0, 0.25))])
    noise = tio.Motion(degrees=np.array([[1.5, 1.5, 1.5]]), translation=np.array([[1.5, 1.5, 1.5]]), times=np.array([0.5]), image_interpolation='linear')

    sim_HQ = []
    sim_LQ = []
    torch.set_printoptions(profile="full", precision=6)
    for idx in ref_idx:
        slice_test = load_slice(test_id, slc)
        slice_HQ = load_slice(idx, slc)
        slice_LQ = load_slice(idx, slc, noise)

        #plt.imshow(np.rot90(slice_LQ[0], 1), cmap='gray')
        #plt.title(f'ID Noise: {test_id}, slice: {slc}')
        #plt.savefig(f'/home/students/studhoene1/imagequality/QualityControlExperiments/noise{test_id}_slc{slc}.png')
        #plt.close()
        #print("mean input shape test: {}".format(torch.mean(slice_HQ.unsqueeze(0))))
        #print("mean input shape lq: {}".format(torch.mean(slice_LQ.unsqueeze(0))))
        feat_test = model(slice_test.unsqueeze(0).to(device))

        feat_HQ = model(slice_HQ.unsqueeze(0).to(device))
        feat_LQ = model(slice_LQ.unsqueeze(0).to(device))
        #print("mean output shape test: {}".format(torch.mean(feat_HQ)))
        #print("mean output shape lq : {}".format(torch.mean(feat_LQ)))
        
        sim_HQ.append(F.cosine_similarity(feat_test, feat_HQ))
        print("HQ similarity: {}".format(F.cosine_similarity(feat_test, feat_HQ)))
        print("LQ similarity: {}".format(F.cosine_similarity(feat_test, feat_LQ)))

        sim_LQ.append(F.cosine_similarity(feat_test, feat_LQ))


    #sim_HQ, sim_LQ = normalize_lists(sim_HQ, sim_LQ)

    print("\nSIM HQ: {}".format(sim_HQ))
    print("SIM_LQ: {}\n".format(sim_LQ))

    sim_HQ = torch.stack(sim_HQ).mean()
    sim_LQ = torch.stack(sim_LQ).mean()
    print(f'Mean similarity to HQ references: {sim_HQ}')
    print(f'Mean similarity to LQ reference: {sim_LQ}')
    plt.imshow(np.rot90(slice_test[0], 1), cmap='gray')
    plt.title(f'ID Noise: {test_id}, slice: {slc}, mean sim to HQ ref: {round(float(sim_HQ), 3)}/ LQ ref: {round(float(sim_LQ), 3)}')
    plt.savefig(f'/home/students/studhoene1/imagequality/QualityControlExperiments/results/3SLices_NTXENT/exp13_UKB_tioNoise_id{test_id}_slc{slc}.png')
    plt.close()

def warp_2D(img, flow):
    flow = flow.astype('float32')
    height, width = np.shape(img)[0], np.shape(img)[1]
    posx, posy = np.mgrid[:height, :width]
    # flow=np.reshape(flow, [-1, 3])
    vx = flow[:, :, 1]  # to make it consistent as in matlab, ux in python is uy in matlab
    vy = flow[:, :, 0]
    coord_x = posx + vx
    coord_y = posy + vy
    coords = np.array([coord_x, coord_y])
    if img.dtype == np.complex128:
        img_real = np.real(img).astype('float32')
        img_imag = np.imag(img).astype('float32')
        warped_real = warp(img_real, coords, order=1)
        warped_imag = warp(img_imag, coords, order=1)
        warped = warped_real + 1j*warped_imag
    else:
        img = img.astype('float32')
        warped = warp(img, coords, order=1)  # order=1 for bi-linear

    return warped
    
#Experiment NAKO IQA (5 reference images, with and without image registraion of reference scans to test scan)
def exp_NAKO_IQA(): 
    test_id = 16
    slc = 100 #100
    registration = False

    ref_idx = list(range(5)) #5

    sim_HQ = []
    sim_LQ = []
    for idx in ref_idx:
        # load test slice
        slc_test = load_slice_nako(test_id, slc)
        #slc_test = load_slice_nako_deep(test_id, slc)
    
        # load reference slices
        noise = tio.Compose([tio.transforms.RandomNoise(mean=0, std=(0, 0.7))])
        motion = tio.Motion(degrees=np.array([[1.5, 1.5, 1.5]]), translation=np.array([[1.5, 1.5, 1.5]]), times=np.array([0.5]), image_interpolation='linear')
        #motion = tio.Compose([tio.transforms.RandomMotion(num_transforms=1, )])

        slc_HQ = load_slice_nako(idx, slc)
        #slc_LQ = load_slice_nako(idx, slc, motion)
        slc_LQ = load_slice_nako_deep(idx, slc)
        if registration:
            slc_test = np.squeeze(slc_test.numpy())
            slc_HQ = np.squeeze(slc_HQ.numpy())
            slc_LQ = np.squeeze(slc_LQ.numpy())
        
            # flow calculation HQ (mov) to test (fix)
        
            ux, uy = optical_flow_tvl1(slc_test, slc_HQ)
            flows = np.stack([ux, uy], axis=-1)
            slc_HQ_warped = warp_2D(slc_HQ, flows)
            slc_HQ = torch.from_numpy(slc_HQ_warped).unsqueeze(0)

            # flow calculation LQ (mov) to test (fix)
            ux, uy = optical_flow_tvl1(slc_test, slc_LQ)
            flows = np.stack([ux, uy], axis=-1)
            slc_LQ_warped = warp_2D(slc_LQ, flows)
            slc_LQ = torch.from_numpy(slc_LQ_warped).unsqueeze(0)

            slc_test = torch.from_numpy(slc_test).unsqueeze(0)
    
        # compute feature representations
        feat_test = model(slc_test.unsqueeze(0).to(device))
        feat_HQ = model(slc_HQ.unsqueeze(0).to(device))
        feat_LQ = model(slc_LQ.unsqueeze(0).to(device))

        # compute cosine similarity
        sim_HQ.append(F.cosine_similarity(feat_test, feat_HQ))
        sim_LQ.append(F.cosine_similarity(feat_test, feat_LQ))
    print(sim_HQ)
    print(sim_LQ)
    sim_HQ = torch.stack(sim_HQ).mean()
    sim_LQ = torch.stack(sim_LQ).mean()
    print(f'Mean similarity to HQ references: {sim_HQ}')
    print(f'Mean similarity to LQ reference: {sim_LQ}')
    plt.imshow(np.rot90(slc_test[0], 1), cmap='gray')
    plt.title(f'NAKO IQA BH ID: {test_id}, slice: {slc}, mean sim to HQ ref: {round(float(sim_HQ), 3)}/ LQ ref: {round(float(sim_LQ), 3)}')
    plt.show()
    plt.savefig(f'/home/students/studhoene1/imagequality/QualityControlExperiments/results/fig_meansim_exp23origprep_nakoiqa_fb_id{test_id}_slc{slc}.png')
    plt.close()

#Experiment: NRU Brain Data
def exp_NRU_brain_data(): 
    patients = range(5, 22)
    slc = 0
    ref_idx = list(range(5))
    sim_scores = {'LQ': [], 'HQ': []}
    for test_id in patients:
        sim_HQ = []
        sim_LQ = []
        for idx in ref_idx:
            slice_test = load_slice_nru_still(test_id, idx=150)
            slice_HQ = load_slice_nru_still(idx, idx=150)
            slice_LQ = load_slice_nru_nod(idx, idx=150)
            feat_test = model(slice_test.unsqueeze(0).to(device))
            feat_HQ = model(slice_HQ.unsqueeze(0).to(device))
            feat_LQ = model(slice_LQ.unsqueeze(0).to(device))
            sim_HQ.append(F.cosine_similarity(feat_test, feat_HQ))
            sim_LQ.append(F.cosine_similarity(feat_test, feat_LQ))
        sim_HQ = torch.stack(sim_HQ).mean()
        sim_LQ = torch.stack(sim_LQ).mean()
        sim_HQ = sim_HQ.cpu().detach().numpy()
        sim_LQ = sim_LQ.cpu().detach().numpy()
        sim_scores['HQ'].append(float(sim_HQ))
        sim_scores['LQ'].append(float(sim_LQ))
        print(f'Mean similarity to HQ references: {sim_HQ}')
        print(f'Mean similarity to LQ reference: {sim_LQ}')
        plt.imshow(np.rot90(slice_test[0], 1), cmap='gray')
        plt.title(f'NRU t1stir nod still ID: {test_id}, slice: {slc}, mean sim to HQ ref: {round(float(sim_HQ), 3)}/ LQ ref: {round(float(sim_LQ), 3)}')
        plt.show()
        plt.close()
    print(sim_scores)

#Experiment: Increasing Motion
def exp_incr_motion():  
    test_id = 64
    reference_keys = pd.read_csv('/home/students/studhoene1/imagequality/QualityControlExperiments/ref_keys.csv')
    root_path = '/mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw/'
    ref_idx = []
    for subject in reference_keys['eid']:
        path = Path(root_path)/str(subject)/"wat.nii.gz"
        ref_idx.append(val_dataset.subjects_paths.index(path))
    slc = 265

    noise = tio.Compose([tio.transforms.RandomNoise(mean=0, std=(0, 0.5))])
    #noise = tio.Compose([tio.transforms.RandomNoise(mean=0, std=(0, 0.5))])
    #motion_lq = tio.Compose([tio.transforms.RandomMotion(num_transforms=1)])


    motion_1 = tio.Motion(degrees=np.array([[0.0, 0.0, 0.0]]), translation=np.array([[0.0, 0.0, 0.0]]),
                          times=np.array([0.5]), image_interpolation='linear')
    motion_2 = tio.Motion(degrees=np.array([[0.5, 0.5, 0.5]]), translation=np.array([[0.5, 0.5, 0.5]]),
                          times=np.array([0.5]), image_interpolation='linear')
    motion_3 = tio.Motion(degrees=np.array([[1.0, 1.0, 1.0]]), translation=np.array([[1.0, 1.0, 1.0]]),
                          times=np.array([0.5]), image_interpolation='linear')
    motion_4 = tio.Motion(degrees=np.array([[1.5, 1.5, 1.5]]), translation=np.array([[1.5, 1.5, 1.5]]),
                          times=np.array([0.5]), image_interpolation='linear')
    motion_5 = tio.Motion(degrees=np.array([[2.0, 2.0, 2.0]]), translation=np.array([[2.0, 2.0, 2.0]]),
                          times=np.array([0.5]), image_interpolation='linear')
    motion_6 = tio.Motion(degrees=np.array([[2.5, 2.5, 2.5]]), translation=np.array([[2.5, 2.5, 2.5]]),
                          times=np.array([0.5]), image_interpolation='linear')
    motion_7 = tio.Motion(degrees=np.array([[3.0, 3.0, 3.0]]), translation=np.array([[3.0, 3.0, 3.0]]),
                          times=np.array([0.5]), image_interpolation='linear')

    motion_list = [motion_1, motion_2, motion_3, motion_4, motion_5, motion_6, motion_7]

    fig, axs = plt.subplots(1, len(motion_list), figsize=(35, 5))
    for i, motion in enumerate(motion_list):
        sim_HQ = []
        sim_LQ = []
        for idx in tqdm(ref_idx):
            slice_test = load_slice(test_id, slc, motion)
            slice_HQ = load_slice(idx, slc)
            slice_LQ = load_slice(idx, slc, noise)
            feat_test = model(slice_test.unsqueeze(0).to(device))
            feat_HQ = model(slice_HQ.unsqueeze(0).to(device))
            feat_LQ = model(slice_LQ.unsqueeze(0).to(device))
            sim_HQ.append(F.cosine_similarity(feat_test, feat_HQ))
            sim_LQ.append(F.cosine_similarity(feat_test, feat_LQ))
            
        sim_HQ = torch.stack(sim_HQ).mean()
        sim_LQ = torch.stack(sim_LQ).mean()
        print(f'Mean similarity to HQ references: {sim_HQ}')
        print(f'Mean similarity to LQ reference: {sim_LQ}')
        axs[i].imshow(np.rot90(slice_test[0], 1), cmap='gray')
        axs[i].set_title(f'Motion {i}: sim HQ: {round(float(sim_HQ), 3)}/ sim LQ: {round(float(sim_LQ), 3)}')
        axs[i].axis('off')

    plt.savefig('/home/students/studhoene1/imagequality/QualityControlExperiments/results/3SLices_NTXENT/exp31_setting_motion_sim.png')

#Experiment: Increasing noise 
def exp_incr_noise():
    test_id = 64
    ref_idc = [1, 2, 3, 4, 5]

    noise_test = tio.Noise(mean=0.0, std=0.3, seed=29)
    noise_1 = tio.Noise(mean=0.0, std=0.1, seed=29)
    noise_2 = tio.Noise(mean=0.0, std=0.2, seed=29)
    noise_3 = tio.Noise(mean=0.0, std=0.3, seed=29)
    noise_4 = tio.Noise(mean=0.0, std=0.4, seed=29)
                      

    #ref_motion_list = ['_', motion_1, motion_2, motion_3, motion_4]
    ref_noise_list = ['_', noise_1, noise_2, noise_3, noise_4]

    slc = 100
    slice_test = load_slice(test_id, slc, noise_test)
    feat_test = model(slice_test.unsqueeze(0).to(device))
    sim = {}
    for Q, noise in enumerate(ref_noise_list):
        sim[f'Q{Q+1}'] = []
        for ref_id in ref_idc:
            if noise == '_':
                slice_ref = load_slice(ref_id, slc)
                slice_ref_sub = torch.sub(slice_ref, slice_ref)
            else:
                slice_ref = load_slice(ref_id, slc)
                slice_ref_noise = load_slice(ref_id, slc, noise)
                slice_ref_sub = torch.sub(slice_ref_noise, slice_ref)
            feat_ref = model(slice_ref_sub.unsqueeze(0).to(device))
            sim[f'Q{Q+1}'].append(F.cosine_similarity(feat_test, feat_ref))
        sim[f'Q{Q+1}'] = torch.stack(sim[f'Q{Q+1}']).mean()

    plt.show()
    plt.imshow(np.rot90(slice_test[0], 1))
    plt.title(f'UKB ID sub noise Q1 groups: {test_id}, slice: {slc}, sim to Q1: {round(float(sim["Q1"]), 3)}, Q2: {round(float(sim["Q2"]), 3)}, Q3: {round(float(sim["Q3"]), 3)}, Q4: {round(float(sim["Q4"]), 3)}, Q5: {round(float(sim["Q5"]), 3)}')
    plt.show()    
    plt.savefig(f'/home/students/studhoene1/imagequality/QualityControlExperiments/results/test_classification_{test_id}_slc{slc}.png')
    plt.close()


model = SimCLR(arch="resnet50")
model.load_state_dict(torch.load('/home/students/studhoene1/imagequality/QualityControlExperiments/checkpoints/exp31_noPad__NoSimMotion_1000Epochs/simclr3Slices_changePosNeg_randomMotion_epoch500.0_loss_0.023298370504849834.pth'))
#model = SimCLR.load_from_checkpoint("/home/raecker1/3DSSL/weights/first_wat_crop_02TO1/checkpoints/epoch=372-step=5287275.ckpt")
#model = SimCLR.load_from_checkpoint("/home/students/studhoene1/test_imqual/imagequality/checkpoints/from_scratch_small_datasetepoch=05_train_loss=0.06.ckpt")
#model = SimCLR.load_from_checkpoint("/home/raecker1/3DSSL/weights/version_4/checkpoints/epoch=892-step=12658275.ckpt")
#model = SimCLR.load_from_checkpoint("/home/raecker1/3DSSL/weights/version_5/checkpoints/epoch=217-step=3090150.ckpt")
#model = SimCLR.load_from_checkpoint("/home/raecker1/3DSSL/weights/version_8/checkpoints/epoch=179-step=5103000.ckpt")

device = torch.device("cuda:0")
model.to(device)
model.eval()

###Experiments
print("EXP UKB")
exp_UKB()
print("EXP NAKO:")
exp_NAKO_IQA()
#exp_NRU_brain_data()
print("EXP Motion Increase:")
exp_incr_motion()
#exp_incr_noise()