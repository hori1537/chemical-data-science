@echo off
SET VIRTUAL_ENV_NAME="chem"
echo Y | conda create -c rdkit -n %VIRTUAL_ENV_NAME% rdkit
call conda activate %VIRTUAL_ENV_NAME%
call pip install -r requirements.txt
call python Chemical_Data_Science.py