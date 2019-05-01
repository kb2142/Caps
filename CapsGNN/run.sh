#!/usr/bin/env bash
T=../dat/pkl/protein_nx2/json/
P=$T
O=./output/protein.csv
python src/main.py --epochs 10 --batch-size 3 --train-graph-folder $T --test-graph-folder $P --prediction-path $O 
