import pandas as pd
import sys
import os
import networks, data_utils
import torch
import numpy as np
import sklearn.metrics
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from joblib import dump, load

datasets = ["lsst_data", "full_data", "high_cad_data", "des_deep_data"]


def compute_labels(network, dataloader, num_classes, device, is_evidential = False):
    output = torch.empty((0, num_classes)).to(device)                                                                                   
    labels = np.array([])
        
    for batch, sample_batched in enumerate(dataloader):  
        partial_output = None
        if is_evidential:
            partial_output, _ = network(sample_batched['lightcurve'].to(device),
                               sample_batched['image'].to(device), return_features = False)
        else:
            partial_output = network(sample_batched['lightcurve'].to(device),
                               sample_batched['image'].to(device), return_features = False)
            
        output = torch.cat((output, partial_output))

        partial_labels = sample_batched['label'].to(device)
        labels = np.concatenate((labels, partial_labels.cpu().data.numpy()))
        
    output = torch.exp(output)
    return output, labels



df = pd.DataFrame(columns = ["Dataset", "Network", "Test F1", "Test Accuracy"])


for ds_name in datasets:
    print("========================== Evaluation of", ds_name, "==========================")
    correct_indexes = dict()
    ts_length = 14
    if ds_name == "full_data":
        ts_length = 7

    groups = ['GROUP_1', 'GROUP_2', 'GROUP_3', 'GROUP_4']
    train_dataset, val_dataset, test_dataset = data_utils.make_train_test_datasets(ds_name, groups, ts_length)
    test_dataloader = data_utils.make_dataloader(test_dataset, shuffle=False)
    train_dataloader = data_utils.make_dataloader(train_dataset, shuffle=False)
    val_dataloader = data_utils.make_dataloader(val_dataset, shuffle=False)

    in_channels = 4
    input_size = 4
    num_classes = 4
    informed = True
    image_side = 45
    device = "cuda"
    
    informed = False

    lonet = networks.LoNet(in_channels, input_size, num_classes, ts_length, informed, ds_name)
    glonet = networks.GloNet(num_classes, ts_length, image_side, in_channels, informed, ds_name)
    munet = networks.MuNet(in_channels, num_classes, image_side, ts_length, informed, ds_name)
    evidential_lonet = networks.EvidentialLoNet(in_channels, input_size, num_classes, ts_length, False, ds_name)
    evidential_munet = networks.EvidentialMuNet(in_channels, input_size, num_classes, ts_length, False, ds_name)

    lonet_model_path = '../models/' + ds_name + '/' + ds_name + '_LoNet_informed_network.pt'
    glonet_model_path = '../models/' + ds_name + '/' + ds_name + '_GloNet_informed_network.pt'
    munet_model_path = '../models/' + ds_name + '/' + ds_name + '_MuNet_informed_network.pt'
    evidential_lonet_model_path = '../models/' + ds_name + '/' + ds_name + '_EvidentialLoNet_noninformed_network.pt'
    evidential_munet_model_path = '../models/' + ds_name + '/' + ds_name + '_EvidentialMuNet_noninformed_network.pt'

    lonet.to(device)
    glonet.to(device)
    munet.to(device)
    evidential_lonet.to(device)
    evidential_munet.to(device)

    lonet.load_state_dict(torch.load(lonet_model_path))
    glonet.load_state_dict(torch.load(glonet_model_path))
    munet.load_state_dict(torch.load(munet_model_path))
    evidential_lonet.load_state_dict(torch.load(evidential_lonet_model_path))
    evidential_munet.load_state_dict(torch.load(evidential_munet_model_path))

    lonet.eval()
    glonet.eval()
    munet.eval()
    evidential_lonet.eval()
    evidential_munet.eval()

    with torch.no_grad():
        lonet_output, lonet_labels = compute_labels(lonet, test_dataloader, 4, device)
        glonet_output, glonet_labels = compute_labels(glonet, test_dataloader, 4, device)
        munet_output, munet_labels = compute_labels(munet, test_dataloader, 4, device)
        evidential_lonet_output, evidential_lonet_labels = compute_labels(evidential_lonet, test_dataloader, 4, device, True)
        evidential_munet_output, evidential_munet_labels = compute_labels(evidential_munet, test_dataloader, 4, device, True)

        tlonet_output, tlonet_labels = compute_labels(lonet, train_dataloader, 4, device)
        tglonet_output, tglonet_labels = compute_labels(glonet, train_dataloader, 4, device)
        tmunet_output, tmunet_labels = compute_labels(munet, train_dataloader, 4, device)
       
        vlonet_output, vlonet_labels = compute_labels(lonet, val_dataloader, 4, device)
        vglonet_output, vglonet_labels = compute_labels(glonet, val_dataloader, 4, device)
        vmunet_output, vmunet_labels = compute_labels(munet, val_dataloader, 4, device)
        
        # The best combination is formed by the results of the three networks
        x_test = torch.cat((lonet_output, glonet_output, munet_output), dim=1).cpu().data.numpy()


        svc_classifier = load('../models/' + ds_name + '/' + ds_name + '_svc.joblib') 
        test_clf_pred = svc_classifier.predict(x_test)

        test_accuracy = sklearn.metrics.accuracy_score(lonet_labels, test_clf_pred)
        test_cm = sklearn.metrics.confusion_matrix(lonet_labels, test_clf_pred)
        print("Test accuracy (DeepGraviLens):", test_accuracy)
        
        test_accuracy = sklearn.metrics.accuracy_score(lonet_labels, np.argmax(lonet_output.cpu().data.numpy(), axis = 1))
        print("Test accuracy (LoNet):", test_accuracy)
        
        test_accuracy = sklearn.metrics.accuracy_score(glonet_labels, np.argmax(glonet_output.cpu().data.numpy(), axis = 1))
        print("Test accuracy (GloNet):", test_accuracy)
        
        test_accuracy = sklearn.metrics.accuracy_score(munet_labels, np.argmax(munet_output.cpu().data.numpy(), axis = 1))
        print("Test accuracy (MuNet):", test_accuracy)
        
        test_accuracy = sklearn.metrics.accuracy_score(munet_labels, np.argmax(evidential_lonet_output.cpu().data.numpy(), axis = 1))
        print("Test accuracy (Evidential LoNet):", test_accuracy)
        
        test_accuracy = sklearn.metrics.accuracy_score(munet_labels, np.argmax(evidential_munet_output.cpu().data.numpy(), axis = 1))
        print("Test accuracy (Evidential MuNet):", test_accuracy)