#!/bin/bash

for x in 'inReactome_Path' # 'in_Reactome' 
do
    y="DEFAULT"
    p=models/$y/$x
    mkdir -p $p
    rm -rf $p/*
    #PYTHONUNBUFFERED=1 srun -o $p/out -e $p/err -J train_${x}_${y} --mem 6G -c 8 -t 04-00 \
    python src/models/train/train_models.py -w $x --tuned $y --PU None
    sleep 2s

    y="TUNED"
    p=models/$y/$x
    mkdir -p $p
    rm -rf $p/*

    for f in $(find models/DEFAULT/$x/* -maxdepth 0 -type d)
    do
        f_=$(basename $f)
        echo $f $f_
        mkdir -p $p/$f_
        cp -r models/DEFAULT/$x/$f_/splits $p/$f_/splits
    done

    #PYTHONUNBUFFERED=1 srun -o $p/out -e $p/err -J train_${x}_${y} --mem 6G -c 8 -t 04-00 \
    python src/models/train/train_models.py -w $x --tuned $y --PU None
    sleep 2s

done
