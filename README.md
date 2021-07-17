# MLPP

Code for reproducing the method described in:

**Co-evolution based machine-learning for predicting functional interactions between human genes**

Doron Stupp, Elad Sharon, Idit Bloch, Marinka Zitnik, Or Zuk, Yuval Tabach*

*. To whom correspondence should be addressed. YuvalTab@ekmd.huji.ac.il

Citation:
```{bibtex}
will be added after publication
```

----------------

## Description

MLPP - Machine Learning based Phylogenetic Profiling, the method described in the aforementioned paper, is a method for predicting functional interactions between human genes based on their phylogenetic profile. This method leverages machine learning based clade-wise co-evolution to improve upon existing phylogenetic profiling approaches. 

The code in this repo enables one to reproduce the models and predictions presented in the paper. A small example for reproducing figure 1a and supp. figure 3 from the paper is included as well. 

## Installation:

Installation and execution was tested on Ubuntu 20.04 running miniconda. To run the full pipeline it is advised to utilize a parallalized execution using snakemake due to the high workload of predicting all gene-pairs across all lables.

### Conda Enviroment

One can install the required software to reproduce our results by using conda. Clone this repo into a directory, and using the terminal from inside the cloned repository run:

```{bash}
conda env create -f enviroment.yaml
```

This will create a conda enviroment called `MLPP`.

### Data files:

To run this example one needs to download the associated phylogenetic profiling matrices and other auxillary data which is available on zenodo at: [10.5281/zenodo.5090776](10.5281/zenodo.5090776) 

To download the data, please execute the data script in a bash terminal inside the folder of the cloned repo.

```{bash}
bash retrieve_data.sh
```

## Running:

### Small example
This example trains several machine learning models to predict functional interactions between genes. This examples reproduces figures 1a and supp. figure 3 from the paper.
For reproducing pathway types comparisons change the argument to `--which inReactome_Path`

Using the command line, run from inside the cloned repo:
```{bash}
conda activate MLPP
python comparing_classifiers.py --which in_Reactome
```

Expected output is available through the `reports/figures/01_comparing_classifiers` folder.

### Running the full pipeline
The full pipeline can be run using snakemake. This takes considerable time due to the prediction of all gene-pairs across all different labels. For the results shown in the paper, the pipeline was ran on an HPC cluster using a slurm scheduler.

First fill `config/config_smk.yaml` with the correct project dir:
```{yaml}
#proj_dir
proj_dir: XXXXXXXXXXXXXX
```

Then, to run the pipeline locally run the script `run_snakemake.sh`
This first installs the `src` directory as a python package and then runs the snakemake pipeline.

```{bash}
conda activate MLPP
bash run_snakemake.sh
```

## Citation

When using the method or predictions described in the paper please cite 
```{bibtex}
will be added after publication
```