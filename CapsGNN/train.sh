#!/usr/bin/env bash
T="../dat/pkl/protein_nx2/json/"
P="../dat/pkl/protein/json/"
O=./output/protein.csv
for i in 1 0.5 0.1 0.01 0.001 0.0001 0.00001
    do
        python src/train.py --epochs 10 --batch-size 60 --train-graph-folder $T --test-graph-folder $P \
        --prediction-path $O --weight-decay 0.000001 --gcn-filters 2 --inner-attention-dimension 20 \
         --capsule-dimensions 8 --learning-rate 0.001 --lambd $i
    done

for j in 1 0.1 0.01 0.0001 0.0001 0.00001 0.000001 0.0000001 0.00000001 0.000000001
    do
        python src/train.py --epochs 10 --batch-size 60 --train-graph-folder $T --test-graph-folder $P \
        --prediction-path $O --weight-decay $j --gcn-filters 2 --inner-attention-dimension 20 \
         --capsule-dimensions 8 --learning-rate 0.001 --lambd 1
    done

for k in 1 0.1 0.01 0.0001 0.0001 0.00001 0.000001 0.0000001 0.00000001 0.000000001
    do
        python src/train.py --epochs 10 --batch-size 60 --train-graph-folder $T --test-graph-folder $P \
        --prediction-path $O --weight-decay 0.000001 --gcn-filters 2 --inner-attention-dimension 20 \
         --capsule-dimensions 8 --learning-rate $k --lambd 1
    done