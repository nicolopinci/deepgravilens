import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder

class CombinedDataset(Dataset):
    def __init__(self, images, lightcurves, labels, transform=None):
        self.images = images
        self.lightcurves = lightcurves
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        lightcurve = self.lightcurves[idx]
        label = np.array(self.labels[idx])
        
        sample = {'lightcurve': lightcurve, 'image': image, 'label': label}
        
        if self.transform:
            sample['image'] = self.transform(torch.from_numpy(image).float())
        else:
            sample['image'] = torch.from_numpy(image).float()
            
        tct = ToCombinedTensor()
        sample = tct(sample)
        
        return sample
    
class ToCombinedTensor(object):
    def __call__(self, sample):
        lightcurve, image, label = sample['lightcurve'], sample['image'], sample['label']
        
        return {'lightcurve': torch.from_numpy(lightcurve).float(),
                'image': image,
                'label': torch.from_numpy(label)}


def train_val_test_split(data, labels, train_ratio = 0.70, validation_ratio = 0.15, test_ratio = 0.15, seed = 42):
    assert train_ratio + validation_ratio + test_ratio == 1.0
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=1 - train_ratio, stratify = labels, random_state = seed)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), stratify = y_test, random_state = seed) 
    return x_train, x_val, x_test, y_train, y_val, y_test

    
def check_data_leakage(train_data, val_data, test_data, modality_name):
    print("Checking the presence of data leakage")
    leakage = False
    
    # Repeated data in train
    u, c = np.unique(train_data, axis=0, return_counts=True)
    dup_tr = u[c > 1]
    print("[", modality_name, "] There are", len(dup_tr), "repetitions in the train set")
    if len(dup_tr)>0:
        leakage = True
    
    # Repeated data in validation
    u, c = np.unique(val_data, axis=0, return_counts=True)
    dup_val = u[c > 1]
    print("[", modality_name, "] There are", len(dup_val), "repetitions in the validation set")
    if len(dup_val)>0:
        leakage = True
        
    # Repeated data in test
    u, c = np.unique(test_data, axis=0, return_counts=True)
    dup_test = u[c > 1]
    print("[", modality_name, "] There are", len(dup_test), "repetitions in the test set")
    if len(dup_test)>0:
        leakage = True
        
    # Repeated data across all sets
    all_data = np.concatenate((train_data, val_data, test_data))
    u, c = np.unique(all_data, axis=0, return_counts=True)
    dup = u[c > 1]
    print("[", modality_name, "] There are", len(dup), "repetitions across different sets")
    if len(dup)>0:
        leakage = True
        
    return leakage

def make_train_test_datasets(directory: str, class_names: list, suffix: int, transform=ToCombinedTensor(), label_map={}):
    images, lightcurves, labels, metadata = [], [], [], []
    
    print("Reading the simulated data")
    # Ingest and label data
    for label, class_name in enumerate(class_names):
        print("- Reading", class_name)
        # Lightcurves
        lc_file = f'../dataset/{directory}/{class_name}_lcs_{suffix}.npy'
        lc = np.load(lc_file)
        lightcurves.append(lc)
        
        # Images
        im_file = f'../dataset/{directory}/{class_name}_ims_{suffix}.npy'
        im = np.load(im_file)
        images.append(im.reshape((len(im), 4, 45, 45)))
        
        # Labels
        class_label = label_map[class_name] if class_name in label_map else label
        labels.extend([class_label]*len(im))
        
        # Metadata
        md_file = f'../dataset/{directory}/{class_name}_mds_{suffix}.npy'
        metadata.append(np.load(md_file, allow_pickle=True).item())
        
    # Split and save metadata
    full_md = []
    for md in metadata:
        for v in md.values():
            full_md.append(v)

    
    # Shuffle and split data
    X_lc = np.concatenate(lightcurves)
    X_im = np.concatenate(images)

    y = np.array(labels, dtype=int)
    train_lightcurves, val_lightcurves, test_lightcurves, train_labels, val_labels, test_labels = train_val_test_split(X_lc, y)
    train_images, val_images, test_images, train_labels2, val_labels2, test_labels2 = train_val_test_split(X_im, y)
    train_md, val_md, test_md, train_labels3, val_labels3, test_labels3 = train_val_test_split(full_md, y)
    
    assert np.alltrue(train_labels == train_labels2)
    assert np.alltrue(train_labels == train_labels3)
    assert np.alltrue(val_labels == val_labels2)
    assert np.alltrue(val_labels == val_labels3)
    assert np.alltrue(test_labels == test_labels2)
    assert np.alltrue(test_labels == test_labels3)
    
    leakage_lc = check_data_leakage(train_lightcurves, val_lightcurves, test_lightcurves, "time series")
    leakage_im = check_data_leakage(train_images, val_images, test_images, "image")
    
    if leakage_lc is False and leakage_im is False:
        print("No data leakage or repetition")
    else:
        print("There is data leakage")
    
    print("Train set proportion:", len(train_labels)/len(y))
    print("Validation set proportion:", len(val_labels)/len(y))
    print("Test set proportion:", len(test_labels)/len(y))

    train_md = {idx: train_md[idx] for idx in range(len(train_md))}
    val_md = {idx: val_md[idx] for idx in range(len(val_md))}
    test_md = {idx: test_md[idx] for idx in range(len(test_md))}
    
    print("Saving the train-validation-test split")
    np.save(f"../dataset/{directory}/{directory}_train_md_{suffix}.npy", train_md, allow_pickle=True)
    np.save(f"../dataset/{directory}/{directory}_validation_md_{suffix}.npy", val_md, allow_pickle=True)
    np.save(f"../dataset/{directory}/{directory}_test_md_{suffix}.npy", test_md, allow_pickle=True)
    
    transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=360),  transforms.RandomVerticalFlip(p=0.5)])
    
    # Create a PyTorch Dataset
    return (CombinedDataset(train_images, train_lightcurves, train_labels, transform=transform),
            CombinedDataset(val_images, val_lightcurves, val_labels, transform=None),
            CombinedDataset(test_images, test_lightcurves, test_labels, transform=None))
        

def make_dataloader(dataset, batch_size=128, shuffle=True, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)