import sys
import os

import data_utils
import utils
import numpy as np

from matplotlib import pyplot as plt
from astropy.visualization import make_lupton_rgb

import networks

import utils

import time
import torch

from joblib import dump, load
import sklearn.metrics
from astropy.visualization import make_lupton_rgb

system_ID = sys.argv[1]

colours = ['g', 'r', 'b', '0.5']
classes_names = ["No Lens", "Lens", "LSNIa", "LSNCC"]
    
def view_image_rgb(images, Q=2.0, stretch=4.0, **imshow_kwargs):
    """
    Merge images into a single RGB image. This function assumes the image array
    is ordered [g, r, i].
    Args:
        images (List[np.array]): a list of at least 3 2-dimensional arrays of pixel values corresponding to different photometric bandpasses
        imshow_kwargs (dict): dictionary of keyword arguments and their values to pass to matplotlib.pyplot.imshow
    """

    assert len(images) > 2, "3 images are needed to generate an RGB image"

    rgb = make_lupton_rgb(images[2],
                          images[1],
                          images[0],
                          Q=Q, stretch=stretch)

    return rgb

def compute_labels(network, device, image_path, time_series_path, name):
    # Open the image .npy
    image = np.load(image_path)
    image = np.expand_dims(image, axis=0)
    image = (image - image.min())/(image.max()-image.min())
    image = torch.from_numpy(image)
    image = image.type(torch.cuda.FloatTensor)
    image = image.to(device)

    # Open the time series .npy
    lightcurve = np.load(time_series_path)
    lightcurve = np.swapaxes(lightcurve, 0,1)
    lightcurve = np.expand_dims(lightcurve, axis=0)
    lightcurve = (lightcurve - lightcurve.min())/(lightcurve.max() - lightcurve.min())
    lightcurve = torch.from_numpy(lightcurve)
    lightcurve = lightcurve.type(torch.cuda.FloatTensor)
    lightcurve = lightcurve.to(device)
    
    return torch.exp(network(lightcurve, image, return_features = False)), lightcurve[0], image[0]



def produce_pdf(image, ts, name, pred_class):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    coadded_img = view_image_rgb(image, Q=90.0, stretch=0.001, stretch_func=np.log10)
    ax[0].imshow(coadded_img)
    ax[0].set_axis_off()

    x_values = np.linspace(1, suffix, suffix)

    labels = ["g", "r", "i", "z"]

    for k in range(0,4):
        ax[1].plot(x_values, ts[:, k], label=labels[k], c = colours[k], linestyle = 'solid', marker = 'o')

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].set_xlabel("Time step")
    ax[1].set_ylabel("Scaled Brightness")
    fig.show()

    plt.title("Prediction: %s" %(pred_class))
    plt.savefig("../results/real_des_deep/%s.pdf" %(name), bbox_inches='tight', pad_inches = 0, dpi = 300)
    
    


with torch.no_grad():
    groups = ['GROUP_1', 'GROUP_2', 'GROUP_3', 'GROUP_4']
    label_map = {}
    
    dataset_name = "des_deep_data"
    suffix = 14
    
    start = time.time()
    
    data_path = "../dataset/real_des_deep/"
    system_name = "DES-" + system_ID
    
    if not os.path.exists(data_path + system_name + "_ts") or not os.path.exists(data_path + system_name + "_img"):
        raise Exception("There are no data for the given ID")
    else:
        print("Reading data")
    
    num_ts = len([name for name in os.listdir(data_path + system_name + "_ts/") if ".npy" in name])
    num_img = len([name for name in os.listdir(data_path + system_name + "_img/") if ".npy" in name])
        
    assert num_ts == num_img, "The number of images is different than the number of timeseries"
    num = num_ts
    print("Processing",num,"samples")
    
    indices = range(0, num)
    predictions = []
    
    for idx in indices:
        image_path = data_path + system_name + "_img/" + str(idx) + ".npy"
        ts_path = data_path + system_name + "_ts/" + str(idx) + ".npy"
        name = system_name + "_" + str(idx)
    
        lonet = networks.LoNet(in_channels = 4, input_size = 4, num_classes = 4, ts_length = suffix, informed = True, ds_name = dataset_name)
        lonet.load_state_dict(torch.load(f"../models/" + dataset_name + "/" + dataset_name + "_LoNet_informed_network.pt"))
        lonet.eval()
        test_lonet, _, _ = compute_labels(lonet, "cpu", image_path, ts_path, name)

        glonet = networks.GloNet(num_classes=4, ts_length=suffix, image_side=45, num_channels=4, informed=True, ds_name = dataset_name)
        glonet.load_state_dict(torch.load(f"../models/" + dataset_name + "/" + dataset_name + "_GloNet_informed_network.pt"))
        glonet.eval()
        test_glonet, _, _ = compute_labels(glonet, "cpu", image_path, ts_path, name)

        munet = networks.MuNet(in_channels = 4, num_classes = 4, image_side = 45, ts_length = suffix, informed = True, ds_name = dataset_name)
        munet.load_state_dict(torch.load(f"../models/" + dataset_name + "/" + dataset_name + "_MuNet_informed_network.pt"))
        munet.eval()
        test_munet, lightcurve, image = compute_labels(munet, "cpu", image_path, ts_path, name)

        x_test = torch.cat((test_lonet, test_glonet, test_munet), dim=1).cpu().data.numpy()
        svc_classifier = load('../models/' + dataset_name + "/" + dataset_name + '_svc.joblib') 

        test_clf_pred = svc_classifier.predict(x_test)[0]
        predictions.append(test_clf_pred)
        print("[sample", idx, "] Prediction:", test_clf_pred)
        
        produce_pdf(image, lightcurve, name, classes_names[int(test_clf_pred)])



    end = time.time()
    
print("Time/sample [s]:", (end-start)/num)