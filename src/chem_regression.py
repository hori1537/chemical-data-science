# coding: utf-8

import os
import tkinter
import tkinter.filedialog

from tkinter import ttk
from tkinter import N,E,S,W
from tkinter import font


import pubchempy as pcp

from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem, Draw, PandasTools

import sys
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


model_sol       = dc.models.GraphConvTensorGraph.load_from_dir('models/solubility')
model_lip       = dc.models.GraphConvTensorGraph.load_from_dir('models/lipophilicity')
model_LT        = dc.models.GraphConvTensorGraph.load_from_dir('models/lifetime')
model_GWP100    = dc.models.GraphConvTensorGraph.load_from_dir('models/GWP100')
model_RE        = dc.models.GraphConvTensorGraph.load_from_dir('models/RE')
#model_n        = dc.models.GraphConvTensorGraph.load_from_dir('models/refractive-index-5')


def choose_csv():
    tk_csv = tkinter.Tk()
    current_dir = os.getcwd()

    csv_file_path = tkinter.filedialog.askopenfilename(initialdir = current_dir, 
    title = 'choose the csv', filetypes = [('csv file', '*.csv')])

    t_csv.set(csv_file_path)


def predict_csv():
    graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    loader = dc.data.data_loader.CSVLoader( tasks = [t_task.get()], smiles_field = t_smiles.get(), id_field = t_id.get(), featurizer = graph_featurizer )
    predict_set = loader.featurize(t_csv.get())

    model_n1        = dc.models.GraphConvTensorGraph.load_from_dir('models/refractive-index-1')
    model_n2        = dc.models.GraphConvTensorGraph.load_from_dir('models/refractive-index-2')
    model_n3        = dc.models.GraphConvTensorGraph.load_from_dir('models/refractive-index-3')
    model_n4        = dc.models.GraphConvTensorGraph.load_from_dir('models/refractive-index-4')
    model_n5        = dc.models.GraphConvTensorGraph.load_from_dir('models/refractive-index-5')

    #for x in inspect.getmembers(predict_set, inspect.ismethod):
    #    print(x[0])

    print(dir(predict_set))
    a = predict_set.itersamples()
    print(a)
    print('predict_set.X')
    print(predict_set.X)

    #print(dir(list(a[0][0])))
    '''
    for i in a:
        print(list(i))
        sys.exit()
        #print(i.isomeric_smiles)
    '''

    smiles_list = []
    #for smiles in predict_set.X:
    #    smiles_list.append(smiles.mols.isomeric_smiles)

    print(len(smiles_list))    
    print(len(predict_set.X))

    predict_sol     = list(model_sol.predict(predict_set))
    predict_sol     = [v[0] for v in predict_sol]
    predict_lip     = model_lip.predict(predict_set)
    predict_lip     = [v[0] for v in predict_lip]
    predict_LT      = model_LT.predict(predict_set)
    predict_LT      = [v[0] for v in predict_LT]
    predict_GWP100  = model_GWP100.predict(predict_set)
    predict_GWP100  = [v[0] for v in predict_GWP100]
    predict_GWP100_raw = [10**v for v in predict_GWP100]


    predict_n1  = model_n1.predict(predict_set)
    predict_n1  = [v[0] for v in predict_n1]
    predict_n2  = model_n2.predict(predict_set)
    predict_n2  = [v[0] for v in predict_n2]
    predict_n3  = model_n3.predict(predict_set)
    predict_n3  = [v[0] for v in predict_n3]
    predict_n4  = model_n4.predict(predict_set)
    predict_n4  = [v[0] for v in predict_n4]
    predict_n5  = model_n5.predict(predict_set)
    predict_n5  = [v[0] for v in predict_n5]

    predict_RE      = model_RE.predict(predict_set)
    predict_RE     = [v[0] for v in predict_RE]

    df_save = pd.DataFrame(
        {'log(sol)':list(predict_sol),
        'log(lip)':list(predict_lip),
        'log(LT)':list(predict_LT),
        'log(GWP100)':list(predict_GWP100),
        'log(RE)':list(predict_RE),
        'GWP100':list(predict_GWP100_raw),
        'refractive-index1':list(predict_n1),
        'refractive-index2':list(predict_n2),
        'refractive-index3':list(predict_n3),
        'refractive-index4':list(predict_n4),
        'refractive-index5':list(predict_n5),
                
        })

    #Save the CSV
    savename = t_savename.get()
    df_save.to_csv('predict/' + savename + '.csv')


def convertsmiles():
    t_smiles.set('')
    t_sol.set('')
    t_lip.set('')
    t_sasc.set('')

    molecule = pcp.get_compounds(t_name.get(), 'name')
    print('molecule')

    print(molecule[0])
    #print('canocical_smile', molecule[0].canonical_smiles)
    print('isomeric_smile',  molecule[0].isomeric_smiles)
    mol_canonical_smiles = molecule[0].canonical_smiles
    mol_isomeric_smiles  = molecule[0].isomeric_smiles
    t_smiles.set(mol_isomeric_smiles)

    mol_ = Chem.MolFromSmiles(mol_isomeric_smiles)

    Draw.MolToFile(mol_, 'tmp.png')

    global image_
    image_open = Image.open('tmp.png')
    image_ = ImageTk.PhotoImage(image_open, master=frame1)

    canvas.create_image(150,75, image=image_)


def regression_gc():
    
    smiles = t_smiles.get()
    df = pd.DataFrame({'name': [t_name.get()], 'smiles' : [t_smiles.get()], 'solubility': [0.00]})
    df.to_csv('tmp.csv')

    graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()

    loader_p = dc.data.data_loader.CSVLoader( tasks = ['solubility'], smiles_field = "smiles", id_field = "name", featurizer = graph_featurizer )
    predictset = loader_p.featurize( 'tmp.csv' )

    prediction_sol =  model_sol.predict(predictset)
    t_sol.set(round(10**prediction_sol[0][0],3))

    prediction_lip =  model_lip.predict(predictset)
    t_lip.set(round(10**prediction_lip[0][0],3))

    prediction_GWP100 =  model_GWP100.predict(predictset)
    t_GWP100.set(round(10**prediction_GWP100[0][0],3))


    PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')

    sa_score = df.ROMol.map(sascorer.calculateScore)

    t_sasc.set(round(sa_score[0],2))

    print(sa_score[0])



root = tkinter.Tk()

font1 = font.Font(family='游ゴシック', size=10, weight='bold')
root.option_add("*Font", font1)
root.option_add("*Button.font", font1)

style1  = ttk.Style()
style1.configure('my.TButton', font = ('游ゴシック',10)  )


root.title('test-horie')

frame1  = tkinter.ttk.Frame(root)


t_name = tkinter.StringVar()
t_smiles = tkinter.StringVar()

t_sol = tkinter.StringVar()
t_lip = tkinter.StringVar()
t_GWP100 = tkinter.StringVar()

t_sasc = tkinter.StringVar()

t_image_dir = tkinter.StringVar()

t_csv = tkinter.StringVar()

label_name  = tkinter.ttk.Label(frame1, text = '分子名:')
entry_name  = ttk.Entry(frame1, textvariable=t_name, width = 60) 

button_tosmiles = ttk.Button(frame1, text='convert to SMILES', command = convertsmiles, style = 'my.TButton')
button_reg_gc   = ttk.Button(frame1, text='Predict!', command = regression_gc, style = 'my.TButton')


label_smiles  = tkinter.ttk.Label(frame1, text = 'Isomeric smiles:')
entry_smiles = ttk.Entry(frame1, textvariable = t_smiles, width = 60)


label_sol = ttk.Label(frame1, text = '水への溶解度 g/100ml:')
entry_sol = ttk.Entry(frame1, textvariable = t_sol)

label_lip = ttk.Label(frame1, text = '水ーオクタノール分配係数:')
entry_lip = ttk.Entry(frame1, textvariable = t_lip)

label_GWP100 = ttk.Label(frame1, text = '地球温暖化係数:')
entry_GWP100 = ttk.Entry(frame1, textvariable = t_GWP100)


label_sasc = ttk.Label(frame1, text = '合成難易度 1:EASY - 10:HARD:')
entry_sasc = ttk.Entry(frame1, textvariable = t_sasc)

canvas = tkinter.Canvas(frame1, width = 300, height = 150)
canvas.grid(row=8, column = 2, sticky= W)

global image_

from PIL import ImageTk, Image
image_open = Image.open('deepchem_.png')

image_ = ImageTk.PhotoImage(image_open, master=frame1)
#image_ = image_.subsample(4)

canvas.create_image(150,75, image=image_)

entry_csv = ttk.Entry(frame1, textvariable = t_csv, width = 60)
button_choosecsv =ttk.Button(frame1, text = 'Choose CSV file', command = choose_csv, style = 'my.TButton')

label_task_clm_name = tkinter.ttk.Label(frame1, text = '特性値の列名:')
label_smiles_clm_name = tkinter.ttk.Label(frame1, text = 'smilesの列名:')
label_id_clm_name = tkinter.ttk.Label(frame1, text = 'IDの列名:')
label_savename = tkinter.ttk.Label(frame1, text = '保存フォルダ名:')

t_task_clm = tkinter.StringVar()
t_smiles_clm = tkinter.StringVar()
t_id_clm = tkinter.StringVar()
t_savename = tkinter.StringVar()

entry_task_clm_name = ttk.Entry(frame1, textvariable=t_task_clm, width = 60) 
entry_smiles_clm_name = ttk.Entry(frame1, textvariable=t_smiles_clm, width = 60) 
entry_id_clm_name = ttk.Entry(frame1, textvariable=t_id_clm, width = 60) 
entry_savename = ttk.Entry(frame1, textvariable=t_savename, width = 60) 

button_predict_csv =ttk.Button(frame1, text = 'Predict!', command = predict_csv)


frame1.grid(row=0,column=0,sticky=(N,E,S,W))

label_name.grid(row=1,column=1,sticky=E)
entry_name.grid(row=1,column=2,sticky=W)

entry_name.bind('<Return>', convertsmiles)

button_tosmiles.grid(row=2,column=2,sticky=W)
button_reg_gc.grid(row=2,column=3,sticky=W)

label_smiles.grid(row=3,column=1,sticky=E)
entry_smiles.grid(row=3, column=2,sticky =W)


label_sol.grid(row=4, column =1, sticky = E)
entry_sol.grid(row=4, column = 2, sticky = W)

label_lip.grid(row=5, column =1, sticky = E)
entry_lip.grid(row=5, column = 2, sticky = W)

label_GWP100.grid(row=6, column =1, sticky = E)
entry_GWP100.grid(row=6, column = 2, sticky = W)


label_sasc.grid(row=7, column =1, sticky = E)
entry_sasc.grid(row=7, column = 2, sticky = W)


button_choosecsv.grid(  row=10,column=1,sticky=E)
entry_csv.grid(         row=10,column=2,sticky=W)
button_predict_csv.grid(row=16,column=2, sticky = W)


label_task_clm_name.grid(    row=12,column=1,sticky=E)
label_smiles_clm_name.grid(  row=13,column=1,sticky=E)
label_id_clm_name.grid(      row=14,column=1,sticky=E)
label_savename.grid(         row=15,column=1,sticky=E)

entry_task_clm_name.grid(    row=12,column=2,sticky=W)
entry_smiles_clm_name.grid(  row=13,column=2,sticky=W)
entry_id_clm_name.grid(      row=14,column=2,sticky=W)
entry_savename.grid(         row=15,column=2,sticky=W)


#cv.grid(row= 4, columns=1, sticky = W)


for child in frame1.winfo_children():
    child.grid_configure(padx=5, pady=5)

root.mainloop()
