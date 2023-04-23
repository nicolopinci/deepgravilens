import numpy as np
import pandas as pd
import torch
import networks
import utils

def save_performance(directory, network_name, prev_network, dataset_part, subset_info, trained_network_path, suffix, is_informed, dataset_name, num_classes):
    print(f"Saving of performances for network {network_name} - {subset_info} set started")
    # Load best network
    network, _ = utils.get_network(network_name, num_classes, is_informed, suffix, dataset_name)
    informed_string = "informed"
    if is_informed is False:
        informed_string = "noninformed"

    network.load_state_dict(torch.load(trained_network_path))
    network.eval()
    
    # Save classifications
    labels = dataset_part[:]['label'].data.numpy()
    res = network(dataset_part[:]['lightcurve'], dataset_part[:]['image']).detach().numpy()
    columns = ["No Lens", "Lens", "LSNIa", "LSNCC", "Label"]

    output = np.hstack((res, labels.reshape(len(labels), 1)))
    df = pd.DataFrame(data=output, columns=columns)
    df.to_csv(f"../results/{directory}/{directory}_{network_name}_{informed_string}_classifications_{subset_info}.csv", index=False)
    print("Classifications saving completed")

    # Save network performance
    if subset_info == "val":
        print("Monitoring saving started")
        out_data = [(a, b, c, d, e) for a, b, c, d, e in zip(prev_network.train_losses, prev_network.val_losses, prev_network.train_acc, prev_network.validation_acc, prev_network.test_acc)]
        out_columns = ["Train Loss", "Validation Loss", "Train Acc", "Validation Acc", "Test Acc"]
        df = pd.DataFrame(data=out_data, columns=out_columns)
        df.to_csv(f"../results/{directory}/{directory}_{network_name}_{informed_string}_monitoring.csv", index=False)
        print("Monitoring saving completed")
    
    print(f"Saving of performances for network {network_name} - {subset_info} set completed")