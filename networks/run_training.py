import sys
import data_utils
import networks
import save
import utils
import numpy as np
import training

# Get dataset name and other parameters
dataset_name = utils.get_dataset_name()
suffix = utils.get_suffix(dataset_name)
network_name = utils.get_network_name()
is_informed = utils.get_informed()

informed_string = None
if is_informed:
    informed_string = "informed"
else:
    informed_string = "noninformed"

num_classes = 4
network, lr = utils.get_network(network_name, num_classes, is_informed, suffix, dataset_name)


# Ingest the data
groups = ['GROUP_1', 'GROUP_2', 'GROUP_3', 'GROUP_4']
print("Loading data")
train_dataset, validation_dataset, test_dataset = data_utils.make_train_test_datasets(dataset_name, groups, suffix)

# Get dataloader
train_dataloader = data_utils.make_dataloader(train_dataset)
validation_dataloader = data_utils.make_dataloader(validation_dataset)
test_dataloader = data_utils.make_dataloader(test_dataset)

# Train network
trained_network_path = f"../models/{dataset_name}/{dataset_name}_{network_name}_{informed_string}"
print(f"Training {network_name}")

print("The model will be saved in", trained_network_path + "_network.pt")

network = training.train_network(network,
                                train_dataloader,
                                train_dataset,
                                validation_dataloader,
                                validation_dataset,
                                test_dataloader,
                                test_dataset,
                                monitor=True,
                                outfile_prefix=trained_network_path,
                                gpu = True,
                                lr = lr,
                                batch_size = 128)


# Save the performance
print("Saving results")
save.save_performance(dataset_name, network_name, network, validation_dataset, "val", trained_network_path + "_network.pt", suffix, is_informed, dataset_name, num_classes)
save.save_performance(dataset_name, network_name, network, test_dataset, "test", trained_network_path + "_network.pt", suffix, is_informed, dataset_name, num_classes)