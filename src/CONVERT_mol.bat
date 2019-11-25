@echo off
REM conda create -c rdkit -n chem rdkit
call conda activate chem
call python nametosmiles_csv.py