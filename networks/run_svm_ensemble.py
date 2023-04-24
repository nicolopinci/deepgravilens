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


def compute_labels(network, dataloader, num_classes, device):
    output = torch.empty((0, num_classes)).to(device)                                                                                   
    labels = np.array([])
        
    for batch, sample_batched in enumerate(dataloader):                            
        partial_output = network(sample_batched['lightcurve'].to(device),
                               sample_batched['image'].to(device), return_features = False)
        output = torch.cat((output, partial_output))

        partial_labels = sample_batched['label'].to(device)
        labels = np.concatenate((labels, partial_labels.cpu().data.numpy()))
        
    output = torch.exp(output)
    return output, labels


df = pd.DataFrame(columns = ["Dataset", "Network", "Test F1", "Test Accuracy"])


for ds_name in datasets:
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
        
        lonet = networks.LoNet(in_channels, input_size, num_classes, ts_length, informed, ds_name)
        glonet = networks.GloNet(num_classes, ts_length, image_side, in_channels, informed, ds_name)
        munet = networks.MuNet(in_channels, num_classes, image_side, ts_length, informed, ds_name)

        lonet_model_path = '../models/' + ds_name + '/' + ds_name + '_LoNet_informed_network.pt'
        glonet_model_path = '../models/' + ds_name + '/' + ds_name + '_GloNet_informed_network.pt'
        munet_model_path = '../models/' + ds_name + '/' + ds_name + '_MuNet_informed_network.pt'

        lonet.to(device)
        glonet.to(device)
        munet.to(device)

        lonet.load_state_dict(torch.load(lonet_model_path))
        glonet.load_state_dict(torch.load(glonet_model_path))
        munet.load_state_dict(torch.load(munet_model_path))

        lonet.eval()
        glonet.eval()
        munet.eval()

        with torch.no_grad():
            lonet_output, lonet_labels = compute_labels(lonet, test_dataloader, 4, device)
            glonet_output, glonet_labels = compute_labels(glonet, test_dataloader, 4, device)
            munet_output, munet_labels = compute_labels(munet, test_dataloader, 4, device)
                        
            tlonet_output, tlonet_labels = compute_labels(lonet, train_dataloader, 4, device)
            tglonet_output, tglonet_labels = compute_labels(glonet, train_dataloader, 4, device)
            tmunet_output, tmunet_labels = compute_labels(munet, train_dataloader, 4, device)
            
            vlonet_output, vlonet_labels = compute_labels(lonet, val_dataloader, 4, device)
            vglonet_output, vglonet_labels = compute_labels(glonet, val_dataloader, 4, device)
            vmunet_output, vmunet_labels = compute_labels(munet, val_dataloader, 4, device)
     
            
            # SVM
            best_svm_params = {'C': None, 'kernel': None}
            best_val_accuracy = 0.0
            best_clf = None
            
            for comb in [[0, 1, 2]]: #[[0, 1], [0, 2], [1, 2], [0, 1, 2]]:
                
                x = None
                x_val = None
                
                for C in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]:
                    for kernel in ['poly', 'linear', 'rbf', 'sigmoid']:

                        clf = make_pipeline(SVC(C = C, kernel = kernel, random_state = 42, break_ties = True, tol = 1e-5))
                        
                        
                        if comb == [0, 1]:
                            x = torch.cat((tlonet_output, tglonet_output), dim=1).cpu().data.numpy()
                            x_val = torch.cat((vlonet_output, vglonet_output), dim=1).cpu().data.numpy()
                        elif comb == [0, 2]:
                            x = torch.cat((tlonet_output, tmunet_output), dim=1).cpu().data.numpy()
                            x_val = torch.cat((vlonet_output, vmunet_output), dim=1).cpu().data.numpy()
                        elif comb == [1, 2]:
                            x = torch.cat((tglonet_output, tmunet_output), dim=1).cpu().data.numpy()
                            x_val = torch.cat((vglonet_output, vmunet_output), dim=1).cpu().data.numpy()
                        elif comb == [0, 1, 2]:
                            x = torch.cat((tlonet_output, tglonet_output, tmunet_output), dim=1).cpu().data.numpy()
                            x_val = torch.cat((vlonet_output, vglonet_output, vmunet_output), dim=1).cpu().data.numpy()
                        
                                
                        y = tlonet_labels

                        clf.fit(x, y)

                        val_clf_pred = clf.predict(x_val)
                        val_f1 = sklearn.metrics.f1_score(vlonet_labels, val_clf_pred, average = 'micro')
                        val_accuracy = sklearn.metrics.accuracy_score(vlonet_labels, val_clf_pred)

                        print("SVM", C, kernel, comb, val_accuracy)

                        if val_accuracy > best_val_accuracy:
                            best_val_accuracy = val_accuracy
                            best_svm_parameters = {'C': C, 'kernel': kernel, 'comb': comb}
                            best_clf = clf
                     
            dump(best_clf, '../models/' + ds_name + '/' + ds_name + '_svc.joblib') 
            
            comb = best_svm_parameters['comb']
            x_test = None
            
            if comb == [0, 1]:
                x_test = torch.cat((lonet_output, glonet_output), dim=1).cpu().data.numpy()
            elif comb == [0, 2]:
                x_test = torch.cat((lonet_output, munet_output), dim=1).cpu().data.numpy()
            elif comb == [1, 2]:
                x_test = torch.cat((glonet_output, munet_output), dim=1).cpu().data.numpy()
            elif comb == [0, 1, 2]:
                x_test = torch.cat((lonet_output, glonet_output, munet_output), dim=1).cpu().data.numpy()
                            
            test_clf_pred = best_clf.predict(x_test)
            test_f1 = sklearn.metrics.f1_score(lonet_labels, test_clf_pred, average = 'macro')
            test_accuracy = sklearn.metrics.accuracy_score(lonet_labels, test_clf_pred)
            
            new_row = {'Dataset': ds_name, 'Ensemble': 'SVM', 'Test F1': test_f1, 'Test Accuracy': test_accuracy}
            df = df.append(new_row, ignore_index=True)

            
            svc_classifier = load('../models/' + ds_name + '/' + ds_name + '_svc.joblib') 
            test_clf_pred = svc_classifier.predict(x_test)
    
            test_accuracy = sklearn.metrics.accuracy_score(lonet_labels, test_clf_pred)
            test_cm = sklearn.metrics.confusion_matrix(lonet_labels, test_clf_pred)

            print("Test accuracy:", test_accuracy)
        
df.to_csv('../results/svm_ensemble.csv')