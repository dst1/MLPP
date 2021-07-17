#!/bin/bash
source activate MLPP

log=logs/snakemake.log

mkdir -p logs
python setup.py install --user -f

snakemake -s Makefile.smk \
          --configfile config/config_smk.yaml \
          --rerun-incomplete \
          --latency-wait 60 2>&1 | tee $log
