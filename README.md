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
* ``notebooks/``: Notebooks for visualization of data.
* ``analysis/``: Scripts to create visualization of the results (requires logging).
* ``scripts/``: Scripts to train and evaluate models, and to download the whole ClimART dataset.

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
    
    bash scripts/download_climart.sh 
   </p>
</details>


    conda env create -f env.yml   # create new environment will all dependencies
    conda activate climart  # activate the environment called 'climart'
    bash download_data_subset.sh  # download the dataset (or a subset of it, see above)
    python run.py trainer.gpus=0 datamodule.training_years="1999"  # train a MLP emulator


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
```      pristine_array = layers_array[:, :, : 14] ```


## Baselines

To reproduce our paper results (for seed = 7), you may choose any of our pre-defined configs in the
 [configs/model](configs/model) folder (for now only mlp) and train it as follows
 
 ```
# Soon: you can replace mlp with "graphnet", "gcn", or "cnn"
# To train on the CPU, choose trainer.gpus=0
python run.py seed=7 model=mlp trainer.gpus=1  
```
 
## Tips

<details><p>
    <summary><b> Reproducibility & Data Generation code </b></summary>
    <p style="padding: 10px; border: 2px solid #ff0000;">
    To best reproduce our baselines and experiments and/or look into how the ClimART dataset was created/designed,
    have a look at our `research_code` branch. It operates on pure PyTorch and has a less clean interface/code 
    than our main branch -- if you have any questions, let us know!
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