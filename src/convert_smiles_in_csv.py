

import tkinter
from tkinter import ttk
from tkinter import N,E,S,W

import pubchempy as pcp

from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem, Draw, PandasTools


import pandas as pd

import sascorer


# refer https://horomary.hatenablog.com/entry/2018/10/21/122025
# refer https://www.ag.kagawa-u.ac.jp/charlesy/2017/07/27/deepchem%E3%81%AB%E3%82%88%E3%82%8B%E6%BA%B6%E8%A7%A3%E5%BA%A6%E4%BA%88%E6%B8%AC-graph-convolution-%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF/
# refer https://future-chem.com/rdkit-intro/
#モジュールの読み込み
import tensorflow as tf
import deepchem as dc
import numpy as np
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel
import sys
from sys import exit

def apply_get_compounds(mol_name):
    try:
        iso_smiles = pcp.get_compounds(mol_name, 'name')[0].isomeric_smiles
    except:
        iso_smiles = ''

    return iso_smiles
file_name = 'GWP/Hodnebrog_et_al_2013_results-'


df = pd.read_csv(file_name + '.csv')
df = pd.DataFrame(df)
#print(df)
#df['smiles'] = pcp.get_compounds()

#df['smiles'] = df['Name'].applymap(pcp.get_compounds())

df['smiles'] = df['Name'].map(apply_get_compounds)

df.to_csv(file_name + '_smiles'.csv')
print(df)
exit()

molecule = pcp.get_compounds(t1.get(), 'name')

model_sol = dc.models.GraphConvTensorGraph.load_from_dir('models')
model_lip = dc.models.GraphConvTensorGraph.load_from_dir('models-Lipophilicity')


def get_all(self):
    t2.set('')
    t_sol.set('')
    t_lip.set('')
    t_sasc.set('')



    print('molecule')

    print(molecule[0])
    #print('canocical_smile', molecule[0].canonical_smiles)
    print('isomeric_smile',  molecule[0].isomeric_smiles)
    mol_canonical_smiles = molecule[0].canonical_smiles
    mol_isomeric_smiles  = molecule[0].isomeric_smiles
    t2.set(mol_isomeric_smiles)

    mol_ = Chem.MolFromSmiles(mol_isomeric_smiles)

    Draw.MolToFile(mol_, 'tmp.png')

    global image_
    image_open = Image.open('tmp.png')
    image_ = ImageTk.PhotoImage(image_open, master=frame1)

    canvas.create_image(150,75, image=image_)

    smiles = t2.get()

    df = pd.DataFrame({'name': [t1.get()], 'smiles' : [t2.get()], 'solubility': [0.00]})
    #df = pd.DataFrame([])
    df.to_csv('tmp.csv')


    graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()

    loader_p = dc.data.data_loader.CSVLoader( tasks = ['solubility'], smiles_field = "smiles", id_field = "name", featurizer = graph_featurizer )
    predictset = loader_p.featurize( 'tmp.csv' )

    prediction_sol =  model_sol.predict(predictset)
    t_sol.set(round(10**prediction_sol[0][0],3))

    prediction_lip =  model_lip.predict(predictset)
    t_lip.set(round(10**prediction_lip[0][0],3))


    PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')

    sa_score = df.ROMol.map(sascorer.calculateScore)

    t_sasc.set(round(sa_score[0],2))

    #print(df['calc_SA_score'])
    print(sa_score[0])



root = tkinter.Tk()
root.title('test-horie')

frame1  = tkinter.ttk.Frame(root)
label1  = tkinter.ttk.Label(frame1, text = 'Molecule name:')
label2  = tkinter.ttk.Label(frame1, text = 'Isomeric smiles:')

t1 = tkinter.StringVar()
t2 = tkinter.StringVar()

t_sol = tkinter.StringVar()
t_lip = tkinter.StringVar()

t_sasc = tkinter.StringVar()

t_image_dir = tkinter.StringVar()



entry1  = ttk.Entry(frame1, textvariable=t1, width = 60)
button1 = ttk.Button(frame1, text='convert', command = get_all)

entry2 = ttk.Entry(frame1, textvariable = t2, width = 60)


label_sol = ttk.Label(frame1, text = 'Water solubility g/100ml:')
entry_sol = ttk.Entry(frame1, textvariable = t_sol)

label_lip = ttk.Label(frame1, text = 'Water-Octanol Partition coefficient LogP:')
entry_lip = ttk.Entry(frame1, textvariable = t_lip)


label_sasc = ttk.Label(frame1, text = 'Synthesis difficulty 1:EASY - 10:HARD:')
entry_sasc = ttk.Entry(frame1, textvariable = t_sasc)

canvas = tkinter.Canvas(frame1, width = 300, height = 150)
canvas.grid(row=7, column = 2, sticky= W)

global image_

from PIL import ImageTk, Image
image_open = Image.open('deepchem_.png')

image_ = ImageTk.PhotoImage(image_open, master=frame1)
#image_ = image_.subsample(4)

canvas.create_image(150,75, image=image_)

#photo = tkinter.PhotoImage(file = '/home/garaken/deepchem-try/test.jpg')

#img = ImageTk.PhotoImage(file ='tmp.svg')
#canvas.create_image(0,0, image = img)


#svgfile = tkinter.PhotoImage(file = 'tmp.svg')
#cv = tkinter.Canvas(bg='red')
#cv.create_image(1,1,image=svgfile)

frame1.grid(row=0,column=0,sticky=(N,E,S,W))

label1.grid(row=1,column=1,sticky=E)
entry1.grid(row=1,column=2,sticky=W)

entry1.bind('<Return>', get_all)

button1.grid(row=2,column=2,sticky=W)

label2.grid(row=3,column=1,sticky=E)
entry2.grid(row=3, column=2,sticky =W)


label_sol.grid(row=4, column =1, sticky = E)
entry_sol.grid(row=4, column = 2, sticky = W)

label_lip.grid(row=5, column =1, sticky = E)
entry_lip.grid(row=5, column = 2, sticky = W)


label_sasc.grid(row=6, column =1, sticky = E)
entry_sasc.grid(row=6, column = 2, sticky = W)


#cv.grid(row= 4, columns=1, sticky = W)


for child in frame1.winfo_children():
    child.grid_configure(padx=5, pady=5)

root.mainloop()
