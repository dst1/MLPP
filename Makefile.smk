import os
import json
from src.tools.PairsIndex import PairsIndex
import numpy as np
import sys

gen_params = json.load(open("config/gen_params.json"))
ind = PairsIndex.from_file(gen_params['index_path'])

n_bins = config['n_bins']
n_genes = ind.n
index_size = ind.ind_size

#set partial indexes
tmp = list(range(0, index_size, int(np.ceil(index_size/n_bins))))
partial_inds = {"{:03}".format(i-1):{"st":tmp[i-1],"en":tmp[i]} for i in range(1,len(tmp))}
partial_inds["{:03}".format(n_bins)]={"st":tmp[-1],"en":index_size}

localrules: all
MODELS=config['models']

rule all:
    input:
        # Data
        "data/clades.json",
        "data/inGmt.csv",
        # Models
        expand("models/{model}/targets",
               model=MODELS),
        #predictions
        expand("data/Predictions/{model}_{AllCV}.hdf5",
               model=MODELS, AllCV="AllCV"),

## models: trains and predicts models to data

##	models_pred: pseudo rule, runs predictions steps
rule models_pred:
    input:
        expand("data/Predictions/{model}.hdf5",
               model=MODELS)

##	models_train: pseudo rule, runs training steps
rule models_train:
    input:
        expand("models/{model}/targets",
               model=MODELS)

##	predict: predicts for a single model, in a partial indexes range
rule predict:
    input:
        targets = "models/{model}/targets"
    output:
        "data/Predictions_partial/Model_Preds_{AllCV}/{model}_{partial_index}.hdf5"
    priority: 50
    params:
        current_range = lambda wildcards: partial_inds[wildcards.partial_index],
        model_dir="models/{model}",
        all_cv=lambda wildcards: True if wildcards.AllCV=="AllCV" else False,
        proj_dir=config["proj_dir"],
        genes_list=gen_params['index_path']
    script:
        config['srcM']+"/predict/predict_model.py"

##	merge_predict: merges all predictions for a single model
rule merge_predict:
    input:
        lambda wildcards: expand("data/Predictions_partial/Model_Preds_{AllCV}/{model}_{partial_index}.hdf5",
                             partial_index=partial_inds.keys(),
                             model=wildcards.model,
                             AllCV=wildcards.AllCV)
    params:
        proj_dir=config["proj_dir"],
        ind_size=index_size,
        name="{model}"
    priority: 100
    output:
        h5_comp = "data/Predictions/{model}_{AllCV}.hdf5"
    script:
        config['srcM']+"/predict/write_h5.py"


##	models_train: trains the different PUFramework-LightGBM models
rule model_train:
    input:
        "data/inGmt.csv",
        "data/inReactome_path.csv",
        "data/inGO_path.csv",
        "data/clades.json"
    output:
        "models/{model}/targets"
    params:
        PU=config['PU'],
    log:
        "logs/TRAIN/{model}_train.log"
    shell:
        "python src/models/train/train_models.py -w {wildcards.model} --PU {params.PU}"

rule data:
    input:
        #clades
        "data/clades.json",
        # inGmt
        "data/inGmt.csv",
        "data/inGO_path.csv",
        "data/inReactome_path.csv"