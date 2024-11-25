import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchio as tio
import torchvision
import numpy as np
from skimage.transform import warp
import h5py
from skimage.registration import optical_flow_tvl1
import matplotlib.pyplot as plt

sys.path.append("/home/raecker1/3DSSL/")
from selfsupervised2d.simclr.simclr_module import SimCLR
from selfsupervised2d.simclr.dataset import SimCLR3DDataset, NakoIQADataset, NRUDataset



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


def load_slice(val_dataset, subj_idx, idx, noise=None):
    if noise:
        slice = noise(val_dataset[subj_idx])["data"]["data"][0,:,:,idx]
    else:
        slice = val_dataset[subj_idx]["data"]["data"][0,:,:,idx]
    #slice = np.pad(slice, ((0,0), (28,28)))
    #slice = torch.from_numpy(slice)
    return slice.unsqueeze(0)


def load_slice_nako(val_dataset, subj_idx, idx, noise=None):
    if noise:
        slice = noise(val_dataset[subj_idx])["data"]["data"][0, 20:-20, :224, idx].unsqueeze(0)
    else:
        slice = val_dataset[subj_idx]["data"]["data"][0, 20:-20, :224, idx].unsqueeze(0)

    print(slice.shape)
    slice = torchvision.transforms.functional.resize(slice, [224, 224])
    # slice = np.pad(slice, ((0,0), (28,28)))
    # slice = torch.from_numpy(slice)
    return slice


def load_slice_nru_still(val_dataset, subj_idx, idx=None, pos=None, noise=None):
    if noise:
        if pos == 'middle':
            print('using middle slice')
            data = noise(val_dataset[subj_idx])["data"]["data"][0,:,:,:]
            slc = data[:,:,data.shape[2]//2].unsqueeze(0)
        elif idx:
            print(f'using id {idx}')
            slc = noise(val_dataset[subj_idx])["data"]["data"][0,:,:,idx].unsqueeze(0)
        else:
            raise ValueError("Please specify idx or pos")
    else:
        if pos == 'middle':
            print('using middle slice')
            data = val_dataset[subj_idx]["data"]["data"][0,:,:,:]
            slc = data[:,:,data.shape[2]//2].unsqueeze(0)
        elif idx:
            print(f'using id {idx}')
            slc = val_dataset[subj_idx]["data"]["data"][0,:,:,idx].unsqueeze(0)
        else:
            raise ValueError("Please specify idx or pos")
    return slc



def load_slice_nru_nod(val_dataset, subj_idx, idx=None, pos=None, noise=None):
    if noise:
        if pos == 'middle':
            print('using middle slice')
            data = noise(val_dataset[subj_idx])["data"]["data"][0,:,:,:]
            slc = data[:,:,data.shape[2]//2].unsqueeze(0)
        elif idx:
            print(f'using id {idx}')
            slc = noise(val_dataset[subj_idx])["data"]["data"][0,:,:,idx].unsqueeze(0)
        else:
            raise ValueError("Please specify idx or pos")
    else:
        if pos == 'middle':
            print('using middle slice')
            data = val_dataset[subj_idx]["data"]["data"][0,:,:,:]
            slc = data[:,:,data.shape[2]//2].unsqueeze(0)
        elif idx:
            print(f'using id {idx}')
            slc = val_dataset[subj_idx]["data"]["data"][0,:,:,idx].unsqueeze(0)
        else:
            raise ValueError("Please specify idx or pos")
    return slc


def global_iqa_inference(img, slc, ref_slc, model_path, ref_data_path, img_reg, region):
    # reference data set
    preprocessings = tio.transforms.Compose([tio.transforms.ZNormalization(), tio.RescaleIntensity((-1, 1))])
    #val_dataset = SimCLR3DDataset(ref_data_path['data'], ref_data_path['keys'], preprocessings, True)
    if region == 'brain':
        val_dataset_hq = NRUDataset(ref_data_path, preprocessings, suffix="01_T1w", infix='_acq-t1tirmpmcoff_rec-wore_run-')    # mprage: _acq-mpragepmcoff_rec-wore_run-
        val_dataset_lq = NRUDataset(ref_data_path, preprocessings, suffix="02_T1w", infix='_acq-t1tirmpmcoff_rec-wore_run-')
        
    else:
        val_dataset_hq = NakoIQADataset(ref_data_path, preprocessings, suffix='bh_W_COMPOSED')
        val_dataset_lq = NakoIQADataset(ref_data_path, preprocessings, suffix='fb_deep_W_COMPOSED')


    # load model
    print('loading model ...')
    model = SimCLR.load_from_checkpoint(model_path)
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    print('model loaded')

    ref_idx = [0,1,2,3,4]

    # for slc in tqdm(range(0, 200, 10)):
    sim_hq = []
    sim_lq = []
    for idx in tqdm(ref_idx):
        # new test slice
        if region == 'abdominal':
            slc_test = torch.from_numpy(img[:,:,slc].copy()).float()
            slc_test = slc_test[-224:, 20:-20].unsqueeze(0)
            slc_test = torchvision.transforms.functional.resize(slc_test, [224, 224])
            plt.imshow(slc_test.squeeze(), cmap='gray')
            plt.savefig(f'dummy_test.png')
            plt.close()

        # load reference slices
        #noise = tio.Compose([tio.transforms.RandomNoise(mean=0, std=(0, 0.25))])
        noise = tio.Compose([tio.transforms.RandomNoise(mean=0, std=(0, 0.7))])

        #hq_img = val_dataset_hq[idx]['data']['data']
        if region == 'brain':
            slc_test = img
            plt.imshow(slc_test.squeeze(), cmap='gray')
            plt.savefig(f'dummy_test.png')
            plt.close()
            slc_HQ = load_slice_nru_still(val_dataset_hq, idx, idx=ref_slc)
            slc_LQ = load_slice_nru_nod(val_dataset_lq, idx, idx=ref_slc)

        else:
            hq_img = val_dataset_hq[idx]['data']['data']
            slc_HQ = hq_img[0, 20:-20, :224, ref_slc].unsqueeze(0)
            slc_HQ = torchvision.transforms.functional.resize(slc_HQ, [224, 224])
            slc_HQ = torch.rot90(slc_HQ, dims=(1, 2))
            #lq_img = noise(val_dataset_hq[idx])['data']['data']
            lq_img = val_dataset_lq[idx]['data']['data']
            slc_LQ = lq_img[0, 20:-20, :224, ref_slc].unsqueeze(0)
            slc_LQ = torchvision.transforms.functional.resize(slc_LQ, [224, 224])
            #slc_LQ = np.rot90(slc_LQ, axes=(1, 2))
            slc_LQ = torch.rot90(slc_LQ, dims=(1, 2))

        plt.imshow(slc_HQ.squeeze(), cmap='gray')
        plt.savefig(f'dummy_hqref_{idx}.png')
        plt.close()

        plt.imshow(slc_LQ.squeeze(), cmap='gray')
        plt.savefig(f'dummy_lqref_{idx}.png')
        plt.close()

        if img_reg:
            slc_test = np.squeeze(slc_test.numpy())
            slc_HQ = np.squeeze(slc_HQ)
            slc_LQ = np.squeeze(slc_LQ)

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

        feat_new = model(slc_test.unsqueeze(0).to(device))
        feat_hq = model(slc_HQ.unsqueeze(0).to(device))
        feat_lq = model(slc_LQ.unsqueeze(0).to(device))
        sim_hq.append(F.cosine_similarity(feat_new, feat_hq))
        sim_lq.append(F.cosine_similarity(feat_new, feat_lq))
    sim_hq = torch.stack(sim_hq).mean()
    sim_lq = torch.stack(sim_lq).mean()
    print(f'Mean similarity to HQ references: {sim_hq}')
    print(f'Mean similarity to LQ reference: {sim_lq}')
    if sim_hq > sim_lq:
        q_class = 1     # HQ
        q_score = sim_hq
    else:
        q_class = 0     # LQ
        q_score = 1 - sim_lq
    return q_class, q_score.detach().cpu().numpy()


if __name__ == '__main__':
    fhandle = h5py.File('/home/fire/FIRE_IQA/test_img.h5', 'r')
    img = fhandle['img']
    ref_data_path = {'data': '/home/fire/FIRE_IQA/data/NAKO_IQA_nifti'}
    model_path = '/home/fire/FIRE_IQA/global_iqa/weights/first_wat_crop_02TO1/checkpoints/epoch=372-step=5287275.ckpt'
    slc = img.shape[0]//2
    image_registration = False
    global_iqa_inference(img, slc, model_path, ref_data_path, image_registration)