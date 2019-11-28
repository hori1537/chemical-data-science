@echo off
conda create -c rdkit -n chem rdkit
call conda activate chem
call python CMC.py
