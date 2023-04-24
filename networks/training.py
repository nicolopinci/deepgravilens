import sys
import numpy as np
import torch
import torch.nn as nn
import tqdm
import math
import sklearn.utils.class_weight as class_weight
import sklearn
import torch.nn.functional as F

def train_network(network, train_dataloader, train_dataset, validation_dataloader, validation_dataset, test_dataloader, test_dataset, monitor=False, outfile_prefix="", gpu = True, lr=None, batch_size = None):
    ntw = gpu_train_network(network, train_dataloader, train_dataset, validation_dataloader, validation_dataset, test_dataloader, test_dataset,  monitor, outfile_prefix, lr, batch_size)
        
    return ntw


def reciprocal_Loss(alpha_dict, p, c = 4):
    total_loss = 0
    
    for key, alpha in alpha_dict.items():
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        label = F.one_hot(p, num_classes=c)
        alp = E * (1 - label) + 1
        S1 = torch.sum(alp, dim=1, keepdim=True)
        reciprocal_Loss = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)+1/torch.sum(((1-label)*(
                torch.digamma(S1) -torch.digamma(alp))), dim=1, keepdim=True)
        total_loss += reciprocal_Loss
        
    loss = torch.mean(total_loss)
    return loss



def compute_labels(network, dataloader, num_classes, device):
    output = torch.empty((0, num_classes)).to(device)                                                                                   
    labels = np.array([])
        
    for batch, sample_batched in enumerate(dataloader):                            
        partial_output = network(sample_batched['lightcurve'].to(device),
                               sample_batched['image'].to(device))
        
        if len(partial_output) == 2: # evidential learning
            partial_output = partial_output[0]
            
        output = torch.cat((output, partial_output))

        partial_labels = sample_batched['label'].to(device)
        labels = np.concatenate((labels, partial_labels.cpu().data.numpy()))
        
    return output, labels
    

def gpu_train_network(network, train_dataloader, train_dataset, validation_dataloader, validation_dataset, test_dataloader, test_dataset, monitor=False, outfile_prefix="", lr=None, batch_size = None):
    num_classes = 4
    es_patience = 20
    number_of_training_epochs = 500
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Learning rate:", lr)

    network.to(device)
    number_batches = math.ceil(len(train_dataset) / batch_size)

    es_flag = False
    best_val_acc = 0.0

    train_size = len(train_dataset)
    validation_size = len(validation_dataset)
    test_size = len(test_dataset)
       
    labels = train_dataset[:]['label']
    
    # the weight are approximately equal in all the dataset because of the stratify option
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(labels), y = labels.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
     
    loss_function = nn.CrossEntropyLoss(weight = class_weights)
    optimizer = torch.optim.Adam(network.parameters(), lr = lr, weight_decay=1e-4)

    train_losses, val_losses, train_accs, validation_accs, test_accs, train_f1s, validation_f1s, test_f1s, = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    for epoch in tqdm.tqdm(range(number_of_training_epochs), total=number_of_training_epochs):
        for i_batch, sample_batched in tqdm.tqdm(enumerate(train_dataloader), total=number_batches):
            network.train()
            optimizer.zero_grad()
            
            output = network(sample_batched['lightcurve'].to(device), sample_batched['image'].to(device))
            
            loss = None
            
            if len(output) == 2: # evidential
                output, alpha = output
                loss = reciprocal_Loss(alpha, sample_batched['label'].to(device))
            else: # non-evidential
                loss = loss_function(output, sample_batched['label'].to(device))
                
            loss.backward()
            optimizer.step()

            if monitor:
                if i_batch % (number_batches - 1) == 0 and i_batch != 0:
                    network.eval()
                    with torch.no_grad():
                        train_output, train_labels = compute_labels(network, train_dataloader, num_classes, device)
                        validation_output, validation_labels = compute_labels(network, validation_dataloader, num_classes, device)
                        test_output, test_labels = compute_labels(network, test_dataloader, num_classes, device)
   
                        train_predictions = torch.max(train_output, 1)[1].cpu().data.numpy()
                        validation_predictions = torch.max(validation_output, 1)[1].cpu().data.numpy()
                        test_predictions = torch.max(test_output, 1)[1].cpu().data.numpy()
                        
                        train_f1 = sklearn.metrics.f1_score(train_labels, train_predictions, average = 'micro')
                        validation_f1 = sklearn.metrics.f1_score(validation_labels, validation_predictions, average = 'micro')
                        test_f1 = sklearn.metrics.f1_score(test_labels, test_predictions, average = 'micro')
                        
                        train_accuracy = np.sum(train_predictions == train_labels)/train_size
                        validation_accuracy = np.sum(validation_predictions == validation_labels)/validation_size
                        test_accuracy = np.sum(test_predictions == test_labels)/test_size

                        train_loss = loss_function(train_output.to(device), torch.from_numpy(train_labels).type(torch.LongTensor).to(device))
                        val_loss = loss_function(validation_output.to(device), torch.from_numpy(validation_labels).type(torch.LongTensor).to(device))
                        
                        print("\nEpoch: {0} | Training Acc.: {1:.3f} -- Validation Acc.: {2:.3f} -- Test Acc.: {3:.3f} -- Training Loss: {4:.3f} -- Validation Loss: {5:.3f}".format(epoch + 1, train_accuracy, validation_accuracy, test_accuracy, train_loss.cpu().data.numpy(), val_loss.cpu().data.numpy()))

                        es_patience_corr = min(es_patience, len(val_losses))
                        val_losses = np.append(val_losses, val_loss.cpu().data.numpy().item())
                        train_losses = np.append(train_losses, train_loss.cpu().data.numpy().item())
                        
                        train_accs = np.append(train_accs, train_accuracy)
                        validation_accs = np.append(validation_accs, validation_accuracy)
                        test_accs = np.append(test_accs, test_accuracy)

                        if es_patience_corr > 0:
                            vl = [validation_accs[-i] - best_val_acc for i in range(1, es_patience_corr + 2)]
                            vl = [v < 0 for v in vl]
                            if sum(vl) == es_patience_corr + 1:
                                print("Early stopping on the validation accuracy")
                                es_flag = True

                        # save best network
                        if validation_accuracy > best_val_acc:
                            torch.save(network.state_dict(), f"{outfile_prefix}_network.pt")
                            best_val_acc = validation_accuracy

                        if es_flag:
                            print("Early stopping - ending batch loop")
                            break

        if es_flag:
            print("Early stopping - ending epochs loop")
            break

    setattr(network, 'train_losses', train_losses)
    setattr(network, 'val_losses', val_losses)
    setattr(network, 'train_acc', train_accs)
    setattr(network, 'validation_acc', validation_accs)
    setattr(network, 'test_acc', test_accs)

    return network