# ***ClimART*** - A Benchmark Dataset for Emulating Atmospheric Radiative Transfer in Weather and Climate Models
<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8.1+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-89b8cd?style=for-the-badge&labelColor=gray"></a>
![CC BY 4.0][cc-by-image]

[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## Official PyTorch Implementation

### Using deep learning to optimise radiative transfer calculations.

Our NeurIPS 2021 Datasets Track paper: https://arxiv.org/abs/2111.14671

Abstract:   *Numerical simulations of Earth's weather and climate require substantial amounts of computation. This has led to a growing interest in replacing subroutines that explicitly compute physical processes with approximate machine learning (ML) methods that are fast at inference time. Within weather and climate models, atmospheric radiative transfer (RT) calculations are especially expensive.  This has made them a popular target for neural network-based emulators. However, prior work is hard to compare due to the lack of a comprehensive dataset and standardized best practices for ML benchmarking. To fill this gap, we build a large dataset, ClimART, with more than **10 million** samples from present, pre-industrial, and future climate conditions, based on the Canadian Earth System Model.
ClimART poses several methodological challenges for the ML community, such as multiple out-of-distribution test sets, underlying domain physics, and a trade-off between accuracy and inference speed. We also present several novel baselines that indicate shortcomings of datasets and network architectures used in prior work.*

**Contact:** Venkatesh Ramesh [(venka97 at gmail)](mailto:venka97@gmail.com) or Salva Rühling Cachay [(salvaruehling at gmail)](mailto:salvaruehling@gmail.com). <br>

## Overview:

* ``climart/``: Package with the main code, baselines and ML training logic.
* ``analysis/``: Scripts to create visualization of the results (requires logging).
* ``configs/``: Yaml configuration files for Hydra that define in a modular way (hyper-)parameters.

## Getting Started
<details><p>
    <summary><b> Requirements</b></summary>
    <p style="padding: 10px; border: 2px solid red;">
    <ul>
    <li>Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.</li>
    <li>NVIDIA GPUs with at least 8 GB of memory and system with 12 GB RAM (More RAM is required if training with --load_train_into_mem option which allows for faster training). We have done all testing and development using NVIDIA V100 GPUs.</li> 
    <li>64-bit Python >=3.7 and PyTorch >=1.8.1. See https://pytorch.org/ for PyTorch install instructions.</li> 
    <li>Python libraries mentioned in ``env.yml`` file, see Getting Started (Need to have miniconda/conda installed).</li> 
    </ul></p>
</details>

<details><p>
    <summary><b> Downloading the ClimART Dataset </b></summary>
    <p style="padding: 10px; border: 2px solid #ff0000;">
    By default, only a subset of CLimART is downloaded.
    To download the train/val/test years you want, please change the loop in ``data_download.sh.`` appropriately.
    To download the whole ClimART dataset, you can simply run 
    
    sudo bash download_climart.sh 
   </p>
</details>
       
  **Note:** If you have issues with downloading the data please let us know to help you.

    conda env create -f env.yml   # create new environment will all dependencies
    conda activate climart  # activate the environment called 'climart'
    sudo bash download_data_subset.sh  # download the dataset (or a subset of it, see above)
    python run.py trainer.gpus=0 datamodule.train_years="2000" # train a MLP emulator on 2000

## Data Structure

To avoid storage redundancy, we store one single input array for both pristine- and clear-sky conditions. The dimensions of ClimART’s input arrays are:
<ul>
<li>layers: (N, 49, D-lay) </li>
<li>levels: (N, 50, 4) </li>
<li>globals: (N, 82) </li>
</ul>

where N is the data dimension (i.e. the number of examples of a specific year, or, during training, of a batch),
 49 and 50 are the number of layers and levels in a column respectively. Dlay, 4, 82 is the number of features/channels for layers, levels, globals respectively. 

For pristine-sky Dlay = 14, while for clear-sky Dlay = 45, since it contains extra aerosol related variables. The array for pristine-sky conditions can be easily accessed by slicing the first 14 features out of the stored array, e.g.:
```      pristine_array = layers_array[:, :, : 14] ```. This is automatically done for you when you set the atmospheric
condition type via ```datamodule.exp_type=pristine``` or ```datamodule.exp_type=clear_sky```.


## Baselines

To reproduce our paper results (for seed = 7), you may choose any of our pre-defined configs in the
 [configs/model](configs/model) folder and train it as follows
 
 ```
# You can replace mlp with "graphnet", "gcn", or "cnn" to run a different ML model
# To train on the CPU, choose trainer.gpus=0
# Specify the directory where the CLimART data is saved with datamodule.data_dir="<your-data-dir>"
# Test on the OOD subsets by setting arg datamodule.{test_ood_historic, test_ood_1991, test_ood_future}=True
python run.py seed=7 model=mlp trainer.gpus=1 
```

To reproduce the exact CNN model used in the paper, you can use the following command:
```
python run.py experiment=reproduce_paper2021_cnn seed=7    # feel free to run for more/other seeds
```
Note: You can also take a look at 
[this WandB report](https://wandb.ai/salv47/ClimART-public-runs/reports/ClimART-paper-CNN-runs--VmlldzozMDUyOTUy)
which shows the results of three runs of the CNN model from the paper.

### Inference
Check out [this notebook](notebooks/2022-06-06-get-predictions-pl.ipynb) for simple code on how to extract the predictions
for each target variable from a trained model (for arbitrary years of the ClimART dataset).

## Tips

<details><p>
    <summary><b> Reproducibility & Data Generation code </b></summary>
    <p style="padding: 10px; border: 2px solid #ff0000;">
    To best reproduce our baselines and experiments and/or look into how the ClimART dataset was created/designed,
    have a look at our `research_code` branch. It operates on pure PyTorch and has a less clean interface/code 
    than our main branch -- if you have any questions, let us know!
</p></details>

<details><p>
    <summary><b> Testing on OOD data subsets </b></summary>
    <p style="padding: 10px; border: 2px solid #ff0000;">
    By default tests run on the main test dataset only (2007-14), to test on the 
    historic, future or anomaly test subsets you need to pass/change the arg
    <code>datamodule.test_ood_historic=True</code> (and/or <code>test_ood_future=True</code>, <code>test_ood_1991=True</code>),
     besides downloading those data files, e.g. via the <code>download_climart.sh</code> script.

</p></details>

<details><p>
    <summary><b> Overriding nested Hydra config groups </b></summary>
    <p style="padding: 10px; border: 2px solid #ff0000;">
    Nested config groups need to be overridden with a different notation - not with a dot, since it would be interpreted as a string otherwise.
    For example, if you want to change the optimizer in the model you want to train, you should run:
    <code>python run.py  model=graphnet  optimizer@model.optimizer=SGD</code>
    <br>
</p></details>

<details><p>
    <summary><b> Local configurations </b></summary>
    <p style="padding: 10px; border: 2px solid #ff0000;">
    You can easily use a local config file (that,e.g., overrides data paths, working dir etc.), by putting such a yaml config
    in the configs/local subdirectory (Hydra searches for & uses by default the file configs/local/default.yaml, if it exists)
</p></details>   
    
<details><p>
    <summary><b> Wandb </b></summary>
    <p style="padding: 10px; border: 2px solid #ff0000;">
    If you use Wandb, make sure to select the "Group first prefix" option in the panel settings of the web app.
    This will make it easier to browse through the logged metrics.
</p></details>

<details><p>
    <summary><b> Credits & Resources </b></summary>
    <p style="padding: 10px; border: 2px solid #ff0000;">
    The following template was extremely useful for getting started with the PL+Hydra implementation:
    [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
</p></details>



## License: 
This work is made available under [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode) license. ![CC BY 4.0][cc-by-shield]

## Development

This repository is currently under active development and you may encounter bugs with some functionality. 
Any feedback, extensions & suggestions are welcome!


## Citation
If you find ClimART or this repository helpful, feel free to cite our publication:

    @inproceedings{cachay2021climart,
        title={{ClimART}: A Benchmark Dataset for Emulating Atmospheric Radiative Transfer in Weather and Climate Models},
        author={Salva R{\"u}hling Cachay and Venkatesh Ramesh and Jason N. S. Cole and Howard Barker and David Rolnick},
        booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
        year={2021},
        url={https://arxiv.org/abs/2111.14671}
    }