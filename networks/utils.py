import os
import sys
from os import listdir
from os.path import isfile, join
import networks

def get_dataset_name():
    """
    Collect dataset name from command line args

    Returns:
        dataset_name as string if command line arg is valid

    Raises:
        ValueError if passed dataset name is not valid
        KeyError if no dataset name is passed
    """

    # Set allowed dataset names
    datasets = ['full_data', 'high_cad_data', 'lsst_data', 'des_deep_data']

    # Get dataset name
    try:
        dataset_name = sys.argv[1]
    except IndexError:
        raise KeyError("No dataset name was passed as a command line arg")

    # Check that the name is valid
    force = "--force" in sys.argv
    if dataset_name not in datasets:
        if not force:
            raise ValueError(f"{dataset_name} is not a valid dataset")

    return dataset_name


def get_suffix(dataset_name):
    dataset_path = f"../dataset/{dataset_name}/"
    all_files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f)) if "GROUP_1_ims" in f]
    suff = [int(f.split("GROUP_1_ims_")[1].split(".npy")[0]) for f in all_files]
    assert len(suff) == 1
    suffix = suff[0]
    return int(suffix)

def get_network_name():
    # Set allowed network names
    networks = ['DeepCNN', 'SmallImageFC', 'ShallowGRU', 'LoNet', 'EvidentialLoNet','GloNet','MuNet','EvidentialMuNet']

    # Get network name
    try:
        network_name = sys.argv[2]
    except IndexError:
        raise KeyError("No network name was passed as a command line arg")

    # Check that the name is valid
    force = "--force" in sys.argv
    if network_name not in networks:
        if not force:
            raise ValueError(f"{network_name} is not a valid network")

    return network_name



def get_informed():
    # Set allowed values
    informed = ['informed', 'noninformed']

    # Get informed settings
    try:
        informed_choice = sys.argv[3]
    except IndexError:
        raise KeyError("No settings for 'informed' was passed as a command line arg")

    # Check that the name is valid
    force = "--force" in sys.argv
    if informed_choice not in informed:
        if not force:
            raise ValueError(f"{informed_choice} is not a valid 'informed' setting")

    if informed_choice == 'informed':
        return True
    return False




def get_network(network_name, num_classes, is_informed, suffix, dataset_name):
    lr = 0.001

    if network_name == 'DeepCNN':
        network = networks.DeepCNN(ts_length = suffix, informed = is_informed, num_classes = num_classes)
    elif network_name == 'SmallImageFC':
        network = networks.SmallImageFC(ts_length = suffix, informed = is_informed, num_classes = num_classes)
    elif network_name == 'ShallowGRU':
        network = networks.ShallowGRU(ts_length = suffix, informed = is_informed, num_classes = num_classes)
    elif network_name == 'LoNet':
        network = networks.LoNet(ts_length = suffix, informed = is_informed, num_classes = num_classes, ds_name = dataset_name)
        lr = 1e-4
    elif network_name == 'EvidentialLoNet':
        network = networks.EvidentialLoNet(ts_length = suffix, informed = is_informed, num_classes = num_classes, ds_name = dataset_name)
        lr = 1e-2
    elif network_name == 'GloNet':
        network = networks.GloNet(ts_length = suffix, informed = is_informed, num_classes = num_classes, ds_name = dataset_name)
        lr = 1e-4
    elif network_name == 'MuNet':
        network = networks.MuNet(ts_length = suffix, informed = is_informed, num_classes = num_classes, ds_name = dataset_name)  
        lr = 1e-4
    elif network_name == 'EvidentialMuNet':
        network = networks.EvidentialMuNet(ts_length = suffix, informed = is_informed, num_classes = num_classes, ds_name = dataset_name)  
        lr = 1e-2
    else:
        raise Exception("Network not supported")
        
    return network, lr
