print('importing libraries......')
from pathlib import Path
from csv import reader

from tkinter.filedialog import askopenfilename
# third party
import pandas as pd
from numpy import nan

from pubchempy import get_compounds,  get_properties

print('program start')

# paths
current_path = Path.cwd()
program_path = Path(__file__).parent.resolve()
parent_path  = program_path.parent.resolve()

data_path           = parent_path / 'data'
data_raw_path       = data_path   / 'raw'
data_interim_path   = data_path   / 'interim'

def get_csv():
    csv_filepath = askopenfilename(initialdir = data_raw_path,
                                   title = 'choose the csv',
                                   filetypes = [('csv file', '*.csv')])
    return csv_filepath

def apply_get_compounds(mol_name):
    properties = get_properties( ['IsomericSMILES'], mol_name, 'name')

    if properties != [] :
        # get_properties return dictionary in list [{'CID': 9890, 'IsomericSMILES': 'C(C(F)F)F'}]
        # properties[0] return dictionary {'CID': 9890, 'IsomericSMILES': 'C(C(F)F)F'}
        iso_smiles = properties[0]['IsomericSMILES']
        print(mol_name, ' : ', iso_smiles)
    else:
        iso_smiles =nan
        print(mol_name, ' can\'t convert')

    return iso_smiles

def convert_nametosmiels(csv_filepath):
    with open(csv_filepath) as f:
        reader_ = reader(f)
        l = [row for row in reader_]
        col_name_of_mol_name = l[0][0]


    print('col_name_of_mol_name : ', col_name_of_mol_name)

    csv_filename =str(Path(csv_filepath).name)

    df = read_csv(csv_filepath)
    df['smiles'] = df[col_name_of_mol_name].map(apply_get_compounds)
    df.to_csv(data_interim_path / (str(Path(csv_filepath).stem) +'_smiles.csv' ),  index=False)
    print('saved csv as ', (data_interim_path / (str(Path(csv_filepath).stem) +'_smiles.csv' )))

    df_dropna = df
    df_dropna = df_dropna.dropna(subset=['smiles']).dropna(how='all')
    df_dropna.to_csv(data_interim_path / (str(Path(csv_filepath).stem) +'_smiles_dropna.csv' ),  index=False)
    print('saved csv as ', (data_interim_path / (str(Path(csv_filepath).stem) +'_smiles_dropna.csv' )))

    print('finish')

def main():
    csv_filepath = get_csv()
    convert_nametosmiels(csv_filepath)

if __name__ == '__main__' :
    main()
