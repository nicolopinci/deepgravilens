# DeepGraviLens: a Multi-Modal Architecture for Classifying Gravitational Lensing Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7860294.svg)](https://doi.org/10.5281/zenodo.7860294)

This repository is the official implementation of [DeepGraviLens: a Multi-Modal Architecture for Classifying Gravitational Lensing Data](https://doi.org/10.1007/s00521-023-08766-9). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data sets and pretrained models
Both the data sets, containing simulated and real data, and the models described in the article, are available on Zenodo at [this link](https://zenodo.org/record/7860294) as zipped files. To use them:
* Create a ```dataset``` folder in the main directory, and put the content of ```dataset.zip``` in there
* Create a ```models``` folder in the main directory, and put the content of ```models.zip``` in there
* Create a ```results``` folder in the main directory, and create the subfolders ```lsst_data```, ```des_deep_data```, ```real_des_deep```, ```full_data```, and ```high_cad_data``` in there

The simulated data sets were obtained using deeplenstronomy, similarly to the ones presented [here](https://doi.org/10.5281/zenodo.5425726). The real data set was obtained using [NoirLab Astro Data Lab services](https://datalab.noirlab.edu/).


## Training

To train one of the unimodal or multimodal networks in the paper, run this command inside the ```networks``` folder:

```train
python run_training.py <dataset_name> <network_name> <is_informed>
```

where:
* ```dataset_name``` is the name of the data set (lsst_data for LSST-wide, des_deep_data for DES-deep, full_data for DES-wide, and high_cad_data for DESI-DOT)
* ```network_name``` is the name of the (unimodal or multimodal network) to train. Available network names are: DeepCNN (i.e., the CNN network used for LoNet and MuNet), SmallImageFC (i.e., the FC network used for MuNet), ShallowGRU (i.e., the GRU network used for LoNet and MuNet), LoNet, EvidentialLoNet, GloNet, MuNet, EvidentialMuNet
* ```is_informed``` must be ```informed``` when the mean and variance are considered, or ```noninformed``` otherwise

To train the SVM ensemble for all the data sets (LoNet, GloNet and MuNet models are required):
```train
python run_svm_ensemble.py
```

Use the following script to train all the networks sequentially:
```train
sh run_all_trainings.sh
```

## Evaluation

To evaluate the trained models on the simulated data sets, run this command inside the ```networks``` folder:

```eval
python overall_evaluation.py
```

Note that this script requires the presence of all the models implemented in the repository.

To evaluate the ensemble of LoNet, GloNet, and MuNet presented in the paper (with SVM) on the simulated data sets, run this command inside the ```networks``` folder:
```eval
python best_evaluation.py
```

To evaluate the ensemble of LoNet, GloNet, and MuNet presented in the paper (with SVM) on the **real** data, run this command inside the ```networks``` folder:
```eval
python real_data_inference.py <OBS_ID>
```
where ```<OBS_ID>``` is the ID associated with the observation as presented in the paper (691022126, 701263907, 699919273).

## Results

Our model achieves the following performance:

|                                        | **DESI-DOT** | **DES-deep** | **DES-wide** | **LSST-wide** |
|----------------------------------------|:---------------------:|:---------------------:|:---------------------:|:----------------------:|
| DeepZipper                    |          77.1         |          58.6         |          51.7         |          74.3          |
| DeepZipper II                 |          78.9         |          57.4         |          49.8         |          70.7          |
| STNet                         |          85.1         |          58.4         |          82.5         |          84.3          |
| EvidentialLoNet (Ours)       |          81.6         |          65.6         |          79.9         |          84.5          |
| EvidentialMuNet (Ours)       |          81.1         |          65.6         |          79.4         |          84.2          |
| LoNet (Ours)                 |          87.0         |          67.5         |          85.8         |          87.2          |
| GloNet (Ours)                |          77.2         |          62.3         |          76.8         |          76.8          |
| MuNet (Ours)                 |          87.9         |          67.9         |          86.5         |          88.5          |
| DeepGraviLens (Ours) |        **88.7**       |        **69.6**       |        **87.7**       |        **88.8**        |
| Improvement                   |          3.6          |          11.0         |          5.2          |           4.5          |

Please refer to the paper for additional analyses.

## Citation
If you use this code, please cite the corresponding article as follows:

> Pinciroli Vago, N.O., Fraternali, P. DeepGraviLens: a multi-modal architecture for classifying gravitational lensing data. Neural Comput & Applic (2023). https://doi.org/10.1007/s00521-023-08766-9

or using the following BibTeX entry:

```
@article{PinciroliVago2023,
  doi = {10.1007/s00521-023-08766-9},
  url = {https://doi.org/10.1007/s00521-023-08766-9},
  year = {2023},
  month = jun,
  publisher = {Springer Science and Business Media {LLC}},
  author = {Nicol{\`{o}} Oreste Pinciroli Vago and Piero Fraternali},
  title = {{DeepGraviLens}: a multi-modal architecture for classifying gravitational lensing data},
  journal = {Neural Computing and Applications}
}
```
