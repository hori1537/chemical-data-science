@echo off
SET VIRTUAL_ENV_NAME="chem2"
echo Y | conda create -c rdkit -n %VIRTUAL_ENV_NAME% rdkit
call conda activate %VIRTUAL_ENV_NAME%
call python Chemical_Data_Science.py