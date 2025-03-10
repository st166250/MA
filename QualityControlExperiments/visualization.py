import sys
sys.path.insert(0, "/home/students/studhoene1/imagequality/")

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torchio as tio
import torchvision
from torchsummary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm
from simclr.simclr_module2 import SimCLR
from simclr.dataset import SimCLR3DDataset, NakoIQADataset, ISMRM_Dataset
from backbone.ssl_head import SSLHead

####Parameters
degrees = 1.0
translation = 1.0
times = 0.5

#print("Plot with Parameters: Degrees: {}, Translation: {}, Times: {}. Datasets: UKB, NAKO[320,260] with and without simulated motion, ISMRM[320,260], direkt resizing ".format(degrees, translation, times))

preprocessings = tio.transforms.Compose([
            tio.transforms.ZNormalization(masking_method=None),
            #tio.RescaleIntensity((0.0, 1.0))
        ])

preprocessings2D = tio.transforms.Compose([
            tio.RescaleIntensity((0.0, 1.0))
        ])
# tsne_dataset_ISMRM_bh = ISMRM_Dataset("/mnt/qdata/share/raecker1/ISMRM_2025/ISMRM_nakoiqa_2/", preprocessings, suffix="bh_nofire_w")
# tsne_dataset_ISMRM_deep = ISMRM_Dataset("/mnt/qdata/share/raecker1/ISMRM_2025/ISMRM_nakoiqa_2/", preprocessings, suffix="deep_fb_nofire_w")
# tsne_loader_ISMRM_deep = DataLoader(tsne_dataset_ISMRM_deep, batch_size=1, shuffle=True, num_workers=4)
# tsne_loader_ISMRM_bh = DataLoader(tsne_dataset_ISMRM_bh, batch_size=1, shuffle=True, num_workers=4)


# UKB
tsne_dataset_ukb = SimCLR3DDataset('/mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw/', "/home/students/studhoene1/imagequality/ukb_abdominal_val_keys.npy",
                    preprocessings, validation=True, small_dataset=True)
tsne_loader_ukb = DataLoader(tsne_dataset_ukb, batch_size=1,
                                          shuffle=True, num_workers=4)

tsne_nako_iqa_dataset = NakoIQADataset("/mnt/qdata/rawdata/NAKO_IQA/NAKO_IQA_nifti/", preprocessings, suffix="bh_W_COMPOSED")
tsne_nako_iqa_dataset_deep = NakoIQADataset("/mnt/qdata/rawdata/NAKO_IQA/NAKO_IQA_nifti/", preprocessings, suffix="fb_deep_W_COMPOSED")
tsne_loader_nako = DataLoader(tsne_nako_iqa_dataset, batch_size=1, shuffle=True, num_workers=4)
tsne_loader_nako_deep = DataLoader(tsne_nako_iqa_dataset_deep, batch_size=1, shuffle=True, num_workers=4)


#model = SimCLR.load_from_checkpoint('/home/students/studhoene1/imagequality/QualityControlExperiments/checkpoints/simclr_epoch10_loss_6.282579112052917.pth')
#model = SimCLR.load_from_checkpoint("/home/students/studhoene1/imagequality/checkpoints/simclr2_NTXENT2_epoch=64_train_loss=6.13.ckpt")
#model = SimCLR(arch="resnet50")
model = SSLHead()

#model.load_state_dict(torch.load('/home/students/studhoene1/imagequality/QualityControlExperiments/checkpoints/exp34_noPad__NoSimMotion_BS128_posDenom/simclr3Slices_changePosNeg_randomMotion_epoch500.0_loss_0.7203241523943449.pth'))
checkpoint = torch.load('/home/students/studhoene1/imagequality/QualityControlExperiments/checkpoints_2gpu/ViT_meanPoolingNoMotion/simclr3Slices1000.0_loss_1.6866214275360107.pth')
# Remove the "module." prefix if present
new_state_dict = {}
for key, value in checkpoint.items():
    new_key = key.replace("module.", "")  # Remove 'module.' from keys
    new_state_dict[new_key] = value

# Load into the model
model.load_state_dict(new_state_dict)

device = torch.device("cuda:0")
model.to(device)
model.eval()
#print(model)
summary(model, (1,224,224))
#summary(model, (1,224,168))

#motion = tio.Compose([tio.transforms.RandomNoise(mean=0, std=(0, 0.25))])
#noise = tio.Compose([tio.transforms.RandomBlur(std=(0.4, 1.3))])
motion = tio.Motion(degrees=np.array([[degrees, degrees, degrees]]), translation=np.array([[translation, translation, translation]]), times=np.array([times]), image_interpolation='linear')
#motion = tio.Compose([tio.transforms.RandomMotion(num_transforms=2, degrees=(-7.5,7.5), translation=(-7.5,7.5))])


feature_list = []
label_list = []
slice_ids = []


with torch.no_grad():
    #for i, images in tqdm(enumerate(tsne_loader_ISMRM_deep)):
    #    for j in range(len(images["data"]["data"][0,0,0,0,:])):
    #        slice_id = f"Image_NAKO-{i}_Slice-{j}"
    #        slice_ids.append(slice_id)  # Append the identifier for HQ
 
    #        image = images["data"]["data"][0,0,:,:,j].unsqueeze(0)
    #        image = torchvision.transforms.functional.resize(image, [224, 168])

    #        feat_bh = model(image.unsqueeze(0).to(device))
            
    #        feature_list.append(feat_bh)
    #        label_list.append(torch.tensor([6]))

    #for i, images in tqdm(enumerate(tsne_loader_ISMRM_bh)):
    #    for j in range(len(images["data"]["data"][0,0,0,0,:])):
    #        slice_id = f"Image_NAKO-{i}_Slice-{j}"
    #        slice_ids.append(slice_id)  # Append the identifier for HQ

    #        image = images["data"]["data"][0,0,:,:,j].unsqueeze(0)
    #        image = torchvision.transforms.functional.resize(image, [224, 168])

    #        feat_bh = model(image.unsqueeze(0).to(device))
           
    #        feature_list.append(feat_bh)
    #        label_list.append(torch.tensor([5]))

    for i, images in tqdm(enumerate(tsne_loader_nako_deep)):
        for j in range(len(images["data"]["data"][0,0,0,0,:])):
            slice_id = f"Image_NAKO-{i}_Slice-{j}"
            slice_ids.append(slice_id)  # Append the identifier for HQ

            image = images["data"]["data"][0,0,:,:,j].unsqueeze(0)
            image = preprocessings2D(images["data"]["data"][0,0,:,:,j].unsqueeze(0).unsqueeze(-1))
            image = image.squeeze(-1)
            image = torchvision.transforms.functional.resize(image, [224, 224])

            #image = images["data"]["data"][0,0,:,:,j].unsqueeze(0)
            #image = np.pad(image.squeeze(0), ((0,0), (28,28)), mode='edge')
            #image = preprocessings2D(image.unsqueeze(-1))
            #image = image.squeeze(-1)

            feat_deep = model(image.unsqueeze(0).to(device))
         
            feature_list.append(feat_deep)
            label_list.append(torch.tensor([4]))

    for i, images in tqdm(enumerate(tsne_loader_nako)):
        image_noise3D = motion(images["data"]["data"][0,0,:,:,:].unsqueeze(0))

        for j in range(len(images["data"]["data"][0,0,0,0,:])):
            slice_id = f"Image_NAKO-{i}_Slice-{j}"
            slice_ids.append(slice_id)  # Append the identifier for HQ
            slice_ids.append(slice_id)  # Append the identifier for LQ

            image = images["data"]["data"][0,0,:,:,j].unsqueeze(0)
            image = preprocessings2D(images["data"]["data"][0,0,:,:,j].unsqueeze(0).unsqueeze(-1))
            image = image.squeeze(-1)
            image = torchvision.transforms.functional.resize(image, [224, 224])

            #image = images["data"]["data"][0,0,:,:,j].unsqueeze(0)
            #image = np.pad(image.squeeze(0), ((0,0), (28,28)), mode='edge')
            #image = preprocessings2D(image.unsqueeze(-1))
            #image = image.squeeze(-1)

            image_noise = preprocessings2D(image_noise3D[0,:,:,j].squeeze(-1).unsqueeze(-1).unsqueeze(0))
            image_noise = image_noise.squeeze(-1)
            image_noise = torchvision.transforms.functional.resize(image_noise, [224, 224])

            #image_noise = image_noise3D[0,:,:,j].squeeze(-1).unsqueeze(0)
            #image_noise = np.pad(image_noise.squeeze(0), ((0,0), (28,28)), mode='edge')
            #image_noise = preprocessings2D(image_noise.unsqueeze(-1))
            #image_noise = image_noise.squeeze(-1)
             # if i%10==0 and j%100==0:
             #      plt.imshow(image_noise.numpy(), cmap='gray')  # Use a colormap like 'gray' for grayscale images
             #      plt.colorbar()  # Optional: Add a color bar
             #      plt.savefig("imNAKO{}_{}_motion2_75_75.png".format(i, j), bbox_inches='tight')
             #      plt.show()
             #      plt.close()
             #      plt.imshow(image.squeeze(0).numpy(), cmap='gray')  # Use a colormap like 'gray' for grayscale images
             #      plt.colorbar()  # Optional: Add a color bar
             #      plt.savefig("imNAKO{}_{}_NOmotion2_75_75.png".format(i,j), bbox_inches='tight')
             #      plt.show()
             #      plt.close()

            feat_bh_hq = model(image.unsqueeze(0).to(device))
            feat_bh_lq = model(image_noise.unsqueeze(0).to(device))
           
            feature_list.append(feat_bh_hq)
            label_list.append(torch.tensor([3]))
            feature_list.append(feat_bh_lq)
            label_list.append(torch.tensor([2]))

    for i, images in tqdm(enumerate(tsne_loader_ukb)):
        image_noise3D = motion(images["data"]["data"][0,0,:,:,:].unsqueeze(0))
        for j in range(363):
            slice_id = f"Image_UKB-{i}_Slice-{j}"
            slice_ids.append(slice_id)  # Append the identifier for HQ
            slice_ids.append(slice_id)  # Append the same identifier for LQ

            image = images["data"]["data"][0,0,:,:,j].unsqueeze(0)
            image = np.pad(image.squeeze(0), ((0,0), (28,28)), mode='edge')
            image = preprocessings2D(torch.from_numpy(image).unsqueeze(-1).unsqueeze(0))
            image = image.squeeze(-1)

            #image_noise = preprocessings2D(image_noise3D[0,:,:,j].squeeze(-1).unsqueeze(-1).unsqueeze(0))
            #image_noise = image_noise.squeeze(-1)
            image_noise = image_noise3D[0,:,:,j].squeeze(-1).unsqueeze(0)
            image_noise = np.pad(image_noise.squeeze(0), ((0,0), (28,28)), mode='edge')
            image_noise = preprocessings2D(torch.from_numpy(image_noise).unsqueeze(-1).unsqueeze(0))
            image_noise = image_noise.squeeze(-1)

            #print(image_noise.shape)
            #plt.imshow(image_noise.squeeze(0).squeeze(-1).numpy(), cmap='gray')  # Use a colormap like 'gray' for grayscale images
            #plt.colorbar()  # Optional: Add a color bar
            #plt.savefig("AAAAAAimage.png")
            #plt.show()
            #aaaa

            #print(np.max(image_noise.numpy()))

            # if i%10==0 and j%100==0:
            #     plt.imshow(image_noise.numpy(), cmap='gray')  # Use a colormap like 'gray' for grayscale images
            #     plt.colorbar()  # Optional: Add a color bar
            #     plt.savefig("im{}_{}_motion_15_15_4.png".format(i, j), bbox_inches='tight')
            #     plt.show()
            #     plt.close()
            #     plt.imshow(image.squeeze(0).numpy(), cmap='gray')  # Use a colormap like 'gray' for grayscale images
            #     plt.colorbar()  # Optional: Add a color bar
            #     plt.savefig("im{}_{}_NOmotion_15_15_4.png".format(i,j), bbox_inches='tight')
            #     plt.show()
            #     plt.close()
            # torch.set_printoptions(profile="full", precision=2)

            # print("mean input shape hq: {}".format(torch.mean(image.unsqueeze(0))))
            # print("mean input shape lq: {}".format(torch.mean(image_noise.unsqueeze(0).unsqueeze(0))))

            feat_hq = model(image.unsqueeze(0).to(device))
            feat_lq = model(image_noise.unsqueeze(0).to(device))
            
            # print("mean output shape hq: {}".format(torch.mean(feat_hq)))
            # print("mean output shape lq : {}".format(torch.mean(feat_lq)))
            feature_list.append(feat_hq)
            label_list.append(torch.tensor([1]))
            feature_list.append(feat_lq)
            label_list.append(torch.tensor([0]))
    

feature_list = torch.cat(feature_list).cpu().numpy()
label_list = torch.cat(label_list).cpu().numpy()

unique_labels = np.unique(label_list)

word_labels = {0: "UKB_LQ", 1: "UKB_HQ", 2: "NAKO_LQ", 3: "NAKO_HQ", 4: "NAKO_deep", 5: "ISMRM_bh", 6: "ISMRM_deep"}


colors = plt.cm.jet(np.linspace(0,1,len(unique_labels)))

tsne = TSNE(n_components=2, random_state=20)
tsne_results = tsne.fit_transform(feature_list)

for idx, (x, y) in enumerate(tsne_results):
    print(f"Point at ({x}, {y}) corresponds to {slice_ids[idx]} with label {label_list[idx]}")

 
plt.figure(figsize=(20,20))
scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c = label_list, cmap='jet', alpha=0.5)
handles = [plt.Line2D([0], [0], marker='o', color='w', label=word_labels[labels], markerfacecolor=colors[i], markersize=10) for i, labels in enumerate(unique_labels)]

plt.legend(handles=handles, title='classes')
plt.title("NTXENT with 3 Slices: Random Motion NAKO, degrees={}, translation={}, time={}".format(degrees, translation, times))
plt.colorbar=scatter
plt.savefig('/home/students/studhoene1/imagequality/tsne_figs_3Slices_Ntxent/ViT1_meanPooling_noMotion.png')
plt.show()