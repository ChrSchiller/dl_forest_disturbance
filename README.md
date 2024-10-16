# Forest Disturbance Detection in Central Europe using Transformers and Sentinel-2 Time Series

This is the code supporting the publication of Schiller et al. (2024) in Remote Sensing of Environment. 

## Citation

If you use our work, please cite the paper as follows: 

```
@ARTICLE{}

```

Link to the publication: 

## Requirements

To use the code, install Python 3.8.10 and the packages specified in requirements.txt using the following commands: 


```
sudo apt install -y python3.8 python3.8-venv python3.8-dev
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Experiments

To run the experiments as in the publication, follow these steps: 

### Dataset Preparation

Execute the R (v43.1 or higher) code in the directories 01_prepare_datasets, 02_get_timeseries and 03_prepare_ts_for_dl in the order indicated by the file numbering, e.g.

```
Rscript ./01_prepare_datasets/01_acquire_meta_forwind.R &&
Rscript ./01_prepare_datasets/01_acquire_meta_senf.R
```

and so on. Note that you need to download the raw datasets beforehand (see paper for details) and change the directories according to your needs. The code in 02_get_times directory needs to be executed for each dataset once - choose pre-training and finetuning code accordingly. While the pretraining data is openly accessible already, the finetuning data can be made available upon reasonable request.

Note: The dataset preparation code requires a Sentinel-2 datacube preprocessed using the FORCE algorithm (https://force-eo.readthedocs.io/en/latest/index.html), e.g. as available on the EO-Lab platform (https://eo-lab.org/en/; https://github.com/CODE-DE-EO-Lab/community_FORCE) for Germany. 

### Deep Learning Training and Testing



### Trained (Finetuned) Models

We provide the trained (in this case: finetuned) Transformer models of one of the three seeds of the study in `./trained_models`. 
The naming of the files is done using the following convention: "dl_forest_disturbance_" + [model setup] + "_" + [spatial holdout for testing] + ".tar". The models can be used for inference or further tuning given the code in this repository. 
Model setups are: 
- "vi_false" for the setup with only Sentinel-2 (S2) bands and no vegetation indices (VIs), called "DL base" in the paper
- "vi_true" for the setup with ten S2 bands and ten VIs, called "DL +IND" in the paper
- "vi_only" for the setup with only ten VIs, called "DL IND" in the paper

Spatial hold-outs are bb/nrw/rlp/lux/sax/thu if Brandenburg/Northrhine-Westphalia/Rhineland-Palatinate/Luxembourg/Saxony/Thuringia AOI was used for testing.

## Disclosure

This code is strongly based on the repository https://github.com/linlei1214/SITS-BERT, as we use the SITS-BERT model as backbone. All credits to the authors of this publication and repository. Thank you very much!



