
from __future__ import print_function
# CMC:chemical model create

print('import libraries')
import os
import random
from pathlib import Path

import sys
from sys import exit

import glob
import pdb

from argparse import ArgumentParser

import tkinter
import tkinter.filedialog
from tkinter import ttk
from tkinter import N, E, S, W
from tkinter import font

import csv
import pandas as pd
import numpy as np
import numpy # Necessary

import pprint
import cloudpickle

import pubchempy as pcp
from mordred import descriptors, Calculator

from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem import BRICS
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import rdMolDraw2D

#scikit-learn
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import sascorer # need fpscores.pkl.gz, sascorer.py

import matplotlib.pyplot as plt
from PIL import ImageTk, Image, ImageDraw

print('import chainer - wait a minute')
import chainer

from chainer import functions
from chainer import optimizers
from chainer import training

from chainer.datasets import split_dataset_random
from chainer.iterators import SerialIterator
from chainer.training import extensions

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.parsers import CSVFileParser
from chainer_chemistry.dataset.converters import converter_method_dict
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict

from chainer_chemistry.links.scaler.standard_scaler import StandardScaler
from chainer_chemistry.models import Regressor
from chainer_chemistry.models import MLP, NFP
from chainer_chemistry.models.prediction import set_up_predictor
from chainer_chemistry.training.extensions.auto_print_report import AutoPrintReport
from chainer.training.extensions import Evaluator

from chainer_chemistry.links.scaler.standard_scaler import StandardScaler  # NOQA
from chainer_chemistry.models.prediction import GraphConvPredictor  # NOQA
from chainer_chemistry.utils import run_train
from chainer_chemistry.utils import save_json

print('finich the importing')


# refer https://horomary.hatenablog.com/entry/2018/10/21/122025
# refer https://www.ag.kagawa-u.ac.jp/charlesy/2017/07/27/deepchem%E3%81%AB%E3%82%88%E3%82%8B%E6%BA%B6%E8%A7%A3%E5%BA%A6%E4%BA%88%E6%B8%AC-graph-convolution-%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF/
# refer https://future-chem.com/rdkit-intro/

# default setting of chainer chemistry
method_name = 'nfp'
#['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn','relgat', 'mpnn', 'gnnfilm']
epochs=50
virtual_libraly_num = 10

# paths
current_path = Path.cwd()
program_path = Path(__file__).parent   # x.pyのあるディレクトリ 相対パス
parent_path = program_path / '../'     # ディレクトリ移動
parent_path_str = str(parent_path.resolve()) #絶対パスに変換してから、文字列に変換　（C\desktop\sssss\aaaaa\） という文字列になる

data_path           = parent_path / 'data'
data_processed_path = data_path / 'processed'

#program_path = Path(__file__).parent.resolve() # 絶対パス
#parent_path = program_path.parent.resolve()　# 絶対パス


if os.name == 'posix':
    import deepchem as dc
    from deepchem.models.tensorgraph.models.graph_models import GraphConvModel
    print('起動環境はLinuxです')

elif os.name == 'nt':
    print('起動環境はWindowsです')
    print('deepchemは利用できません')
else:
    print('OSの種類を判別できません')

def chk_mkdir(theme_name):
    paths =[parent_path / 'models',
            parent_path / 'models' / theme_name,
            parent_path / 'models' / theme_name / 'chainer',
            parent_path / 'results',
            parent_path / 'results' / theme_name,
            parent_path / 'results' / theme_name / 'chainer' ,
            parent_path / 'results' / theme_name / 'chainer' / 'predict',
            parent_path / 'results' / theme_name / 'chainer' / 'search',
            parent_path / 'results' / theme_name / 'chainer' / 'virtual-search' ,
            parent_path / 'results' / theme_name / 'chainer' / 'virtual-search'  / 'png'
            ]

    for path_name in paths:
        if os.path.exists(path_name) == False:
            os.mkdir(path_name)

    return

def get_csv():
    current_dir = os.getcwd()

    csv_file_path = tkinter.filedialog.askopenfilename(initialdir = data_processed_path,
                                                        title = 'choose the csv',
                                                        filetypes = [('csv file', '*.csv')])


    t_csv_filename.set(str(Path(csv_file_path).name))
    t_csv_filepath.set(csv_file_path)

    t_theme_name.set(Path(csv_file_path).parent.name)

    with open(t_csv_filepath.get()) as f:
        reader = csv.reader(f)
        l = [row for row in reader]

        t_id.set(l[0][0])       #CSVの１列目がIDや名前
        t_task.set(l[0][1])     #CSVの２列目が目的関数（水の溶解度等）
        t_smiles.set(l[0][2])   #CSVの３列目がSMILES表記

def apply_molfromsmiles(smiles_name):
    try:
        mols = Chem.MolFromSmiles(smiles_name)

    except:
        #SMILESからMOLに変換できなかった場合、""を返す
        mols = ""
        print(smiles_name)
        print('Error')

    return mols

def parse_arguments():
    #Chainer-chemistry用のparse
    theme_name = t_theme_name.get() + '-' + method_name

    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn',
                   'relgat', 'mpnn', 'gnnfilm']
    scale_list = ['standardize', 'none']

    # Set up the argument parser.
    parser = ArgumentParser(description='Regression on own dataset')

    parser.add_argument('--datafile', '-d', type=str,
                        default='dataset_train.csv',
                        help='csv file containing the dataset')

    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default=method_name)

    parser.add_argument('--label', '-l', nargs='+',
                        default=t_task.get(),
                        help='target label for regression')

    parser.add_argument('--scale', type=str, choices=scale_list,
                        help='label scaling method', default='standardize')

    parser.add_argument('--conv-layers', '-c', type=int, default=4,
                        help='number of convolution layers')

    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='batch size')

    parser.add_argument('--device', type=str, default='-1',
        help='Device specifier. Either ChainerX device specifier or an '
             'integer. If non-negative integer, CuPy arrays with specified '
             'device id are used. If negative integer, NumPy arrays are used')

    parser.add_argument('--out', '-o', type=str, default=parent_path / 'models',
                        help='path to save the computed model to')

    parser.add_argument('--epoch', '-e', type=int, default=int(float(t_epochs.get())),
                        help='number of epochs')

    parser.add_argument('--unit-num', '-u', type=int, default=16,
                        help='number of units in one layer of the model')

    parser.add_argument('--seed', '-s', type=int, default=777,
                        help='random seed value')

    parser.add_argument('--train-data-ratio', '-r', type=float, default=0.7,
                        help='ratio of training data w.r.t the dataset')

    parser.add_argument('--protocol', type=int, default=2,
                        help='pickle protocol version')

    parser.add_argument('--model-foldername', type=str, default=os.path.join(theme_name , 'chainer'),
                        help='saved model foldername')

    parser.add_argument('--model-filename', type=str, default='regressor.pkl',
                        help='saved model filename')

    return parser.parse_args()

def rmse(x0, x1):
    return functions.sqrt(functions.mean_squared_error(x0, x1))

def fit_by_chainer_chemistry():

    def main():
        # Parse the arguments.
        args = parse_arguments()
        theme_name = t_theme_name.get() + '-' + method_name
        print(theme_name)

        if args.label:
            labels = args.label
            class_num = len(labels) if isinstance(labels, list) else 1
        else:
            raise ValueError('No target label was specified.')

        # Dataset preparation. Postprocessing is required for the regression task.
        def postprocess_label(label_list):
            return numpy.asarray(label_list, dtype=numpy.float32)

        # Apply a preprocessor to the dataset.
        print('Preprocessing dataset...')
        preprocessor = preprocess_method_dict[args.method]()
        smiles_col_name = t_smiles.get()
        print('smiles_col_name',  smiles_col_name)

        parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label,
                               labels=labels, smiles_col=smiles_col_name)

        args.datafile=t_csv_filepath.get()
        dataset = parser.parse(args.datafile)['dataset']

        # Scale the label values, if necessary.
        if args.scale == 'standardize':
            scaler = StandardScaler()
            scaler.fit(dataset.get_datasets()[-1])
        else:
            scaler = None

        # Split the dataset into training and validation.
        train_data_size = int(len(dataset) * args.train_data_ratio)
        trainset, testset = split_dataset_random(dataset, train_data_size, args.seed)



        # Set up the predictor.
        if Booleanvar_transfer_learning.get() == False:
            predictor = set_up_predictor(
                args.method, args.unit_num,
                args.conv_layers, class_num, label_scaler=scaler)

        elif Booleanvar_transfer_learning.get() == True:
            # refer https://github.com/pfnet-research/chainer-chemistry/issues/407
            with open(parent_path / 'models' / 'Lipophilicity-nfp' / 'chainer' / ('regressor-' + str(args.method) + '.pickle'),  'rb') as f:
                regressor = cloudpickle.loads(f.read())
                pre_predictor = regressor.predictor
                predictor = GraphConvPredictor(pre_predictor.graph_conv, MLP(out_dim=1, hidden_dim=16))

        # Set up the regressor.
        device = chainer.get_device(args.device)
        metrics_fun = {'mae': functions.mean_absolute_error, 'rmse': rmse}

        regressor = Regressor(predictor, lossfun=functions.mean_squared_error,
                              metrics_fun=metrics_fun, device=device)

        print('Training...')
        run_train(regressor, trainset, valid=None,
                  batch_size=args.batchsize, epoch=args.epoch,
                  out=args.out, extensions_list=None,
                  device=device, converter=concat_mols,
                  resume_path=None)

        # Save the regressor's parameters.
        args.model_foldername=t_theme_name.get()

        model_path = os.path.join(args.out, args.model_foldername, args.model_filename)
        print('Saving the trained model to {}...'.format(model_path))

        # TODO(nakago): ChainerX array cannot be sent to numpy array when internal
        # state has gradients.
        if hasattr(regressor.predictor.graph_conv, 'reset_state'):
            regressor.predictor.graph_conv.reset_state()

        #model_path = os.path.join(os.getcwd(), args.model_filename)
        print(model_path)

        with open(parent_path / 'models' / theme_name / 'chainer' / ('regressor-' + str(args.method) + '.pickle'),  'wb') as f:
            cloudpickle.dump(regressor, f)

        with open(parent_path / 'models' / theme_name / 'chainer' / ('predictor-' + str(args.method) + '.pickle'),  'wb') as f:
            cloudpickle.dump(predictor, f)


        print('Evaluating...')
        test_iterator = SerialIterator(testset, 16, repeat=False, shuffle=False)
        eval_result = Evaluator(test_iterator, regressor, converter=concat_mols,
                                device=device)()
        print('Evaluation result: ', eval_result)

        @chainer.dataset.converter()
        def extract_inputs(batch, device=None):
            return concat_mols(batch, device=device)[:-1]


        pred_train = regressor.predict(trainset, converter=extract_inputs)
        pred_train = [i[0] for i in pred_train]
        pred_test = regressor.predict(testset, converter=extract_inputs)
        pred_test = [i[0] for i in pred_test]


        y_train = [i[2][0] for i in trainset]
        y_test = [i[2][0] for i in testset]


        from PIL import Image
        plt.figure(figsize=(5,5))
        plt.scatter(y_train, pred_train, label = 'Train', c = 'blue')
        plt.title(args.label)
        plt.xlabel('Measured value')
        plt.ylabel('Predicted value')

        plt.scatter(y_test, pred_test, c = 'lightgreen', label = 'Test', alpha = 0.8)
        plt.legend(loc = 4)

        plt.savefig(parent_path / 'results' / theme_name / 'chainer' / 'scatter.png')
        plt.close()

        global image_score
        image_score_open = Image.open(parent_path / 'results' / theme_name / 'chainer' / 'scatter.png')
        image_score = ImageTk.PhotoImage(image_score_open, master=frame1)

        canvas.create_image(200,200, image=image_score)

    main()

def predict_by_chainer_chemistry():
    def main():
        # Parse the arguments.
        args = parse_arguments()
        theme_name = t_theme_name.get() + '-' + method_name

        if args.label:
            labels = args.label
        else:
            raise ValueError('No target label was specified.')

        # Dataset preparation.
        def postprocess_label(label_list):
            return numpy.asarray(label_list, dtype=numpy.float32)

        smiles_col_name = t_smiles.get()
        print('Preprocessing dataset...')
        preprocessor = preprocess_method_dict[args.method]()
        parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label,
                               labels=labels, smiles_col=t_smiles.get())
        args.datafile=parent_path / 'results' /  theme_name / 'chainer' / 'virtual-search'  / 'virtual.csv'
        virtualset = parser.parse(args.datafile)['dataset']

        test = virtualset

        @chainer.dataset.converter()
        def extract_inputs(batch, device=None):
            return concat_mols(batch, device=device)[:-1]

        print('Predicting...')
        # Set up the regressor.
        device = chainer.get_device(args.device)
        model_path = os.path.join(args.out, args.model_foldername, args.model_filename)

        with open(parent_path / 'models' / theme_name / 'chainer' / ('regressor-' + str(args.method) + '.pickle'), 'rb') as f:
            regressor = cloudpickle.loads(f.read())

        # Perform the prediction.
        print('Evaluating...')
        converter = converter_method_dict[args.method]
        test_iterator = SerialIterator(test, 16, repeat=False, shuffle=False)
        eval_result = Evaluator(test_iterator, regressor, converter=converter,
                                device=device)()
        print('Evaluation result: ', eval_result)

        pred_virtual = regressor.predict(virtualset, converter=extract_inputs)
        pred_virtual = [i[0] for i in pred_virtual]
        df_virtual = pd.read_csv(parent_path / 'results' /  theme_name / 'chainer' / 'virtual-search'  / 'virtual.csv')
        df_pred_virtual = df_virtual
        df_pred_virtual[t_task.get()] = pred_virtual

        #print(df_pred_virtual)
        df_pred_virtual =df_pred_virtual.dropna()
        df_pred_virtual.to_csv(parent_path / 'results' /  theme_name / 'chainer' / 'virtual-search' /  'virtual.csv')

        png_list = (parent_path / 'results' /  theme_name / 'chainer' / 'virtual-search'  / 'png').glob('*.png')

        #print(len(df_pred_virtual[t_task.get()]))
        for i, png_path in enumerate(png_list):
            if i < len(df_pred_virtual[t_task.get()]):
                #print('i', i )
                #print(str(round(df_pred_virtual[t_task.get()][i],2)))
                img = Image.open(png_path)
                draw = ImageDraw.Draw(img)# im上のImageDrawインスタンスを作る
                draw.text((0,0), t_task.get() + ' : ' + str(round(df_pred_virtual[t_task.get()][i],2)),  (0,0,0))
                img.save(png_path)

        save_json(os.path.join(args.out, 'eval_result.json'), eval_result)
    main()


def make_virtual_lib():
    theme_name = t_theme_name.get() + '-' + method_name
    df_brics = pd.read_csv(t_csv_filepath.get())
    df_brics['mols'] = df_brics[t_smiles.get()].map(apply_molfromsmiles)
    #print(df_brics)

    df_brics = df_brics.dropna()
    #print(df_brics)

    allfrags = set()
    #Applying the for-loop to pandas df is not good.
    for mol in df_brics['mols']:
        frag=BRICS.BRICSDecompose(mol)
        allfrags.update(frag)



    print('the number of allfrags', len(allfrags))

    allcomponents = [apply_molfromsmiles(f) for f in allfrags]
    Nonecomponents = [f for f in allcomponents if f==None or f==""]
    print('len(Nonecomponents)', len(Nonecomponents))
    allcomponents = [f for f in allcomponents if f != ""]
    allcomponents = [f for f in allcomponents if f != None]

    #pprint.pprint(allcomponents)

    for f in allfrags:
        #print('f: ', f)
        #print('Mol: ',Chem.MolFromSmiles(f))
        #print(' ')
        pass

    #print(allcomponents)
    builder = BRICS.BRICSBuild(allcomponents)

    print(builder)

    virtual_mols =[]

    successful_cnt = 0
    error_cnt = 0

    for i in range(virtual_libraly_num):
        try:
            m=next(builder)
            m.UpdatePropertyCache(strict=True)
            virtual_mols.append(m)
            successful_cnt+=1

        except StopIteration:
            #print(i, '- stopiteration of next(builder)')
            error_cnt +=1
            pass
        except :
            print('error')
            error_cnt +=1
            pass

    print('The number of error : ', error_cnt)
    print('The ratio of error : ', error_cnt / virtual_libraly_num)

    for i, mol in enumerate(virtual_mols):
        Draw.MolToFile(mol, str(parent_path / 'results' /  theme_name / 'chainer' / 'virtual-search'  / 'png' / ('tmp-' + str(i) + '.png')))


    virtual_list = []
    for i, mol in enumerate(virtual_mols):
        virtual_list.append([i, Chem.MolToSmiles(mol), 0])

    #print(virtual_list)
    df_virtual = pd.DataFrame(virtual_list,
                              columns=[t_id.get(), t_smiles.get(), t_task.get()])

    #print(df_virtual)
    df_virtual.to_csv(parent_path / 'results' /  theme_name / 'chainer' / 'virtual-search'  / 'virtual.csv')



def fit_by_deepchem():
    graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()

    loader = dc.data.data_loader.CSVLoader( tasks = [t_task.get()], smiles_field = t_smiles.get(), id_field = t_id.get(), featurizer = graph_featurizer )
    dataset = loader.featurize(t_csv_filepath.get())

    splitter = dc.splits.splitters.RandomSplitter()
    trainset, testset = splitter.train_test_split( dataset )

    hp = dc.molnet.preset_hyper_parameters
    param = hp.hps[ 'graphconvreg' ]
    print( param )

    batch_size = 48

    from deepchem.models.tensorgraph.models.graph_models import GraphConvModel
    model = GraphConvModel(n_tasks=1, batch_size=64, uncertainty=False, mode='regression')

    model = dc.models.GraphConvTensorGraph(
        1,
        batch_size=batch_size,
        learning_rate=1e-3,
        use_queue=False,
        mode = 'regression',
        model_dir= t_theme_name.get())

    np.random.seed(1)
    random.seed(1)

    model.fit(dataset, nb_epoch=max(1, int(t_epochs.get())))
    #model.fit(trainset, nb_epoch=max(1, int(t_epochs.get())))

    metric = dc.metrics.Metric(dc.metrics.r2_score)

    print('epoch: ', t_epochs.get() )
    print("Evaluating model")
    train_score = model.evaluate(trainset, [metric])
    test_score  = model.evaluate(testset, [metric])

    model.save()

    pred_train = model.predict(trainset)
    pred_test  = model.predict(testset)

    y_train = np.array(trainset.y, dtype = np.float32)
    y_test = np.array(testset.y, dtype = np.float32)


    plt.figure()

    plt.figure(figsize=(5,5))

    plt.scatter(y_train, pred_train, label = 'Train', c = 'blue')
    plt.title('Graph Convolution')
    plt.xlabel('Measured value')
    plt.ylabel('Predicted value')
    plt.scatter(y_test, pred_test, c = 'lightgreen', label = 'Test', alpha = 0.8)
    plt.legend(loc = 4)
    #plt.show()
    plt.savefig('score-tmp.png')

    img = Image.open('score-tmp.png')

    img_resize = img.resize((400, 400), Image.LANCZOS)
    img_resize.save('score-tmp.png')


    global image_score
    image_score_open = Image.open('score-tmp.png')
    image_score = ImageTk.PhotoImage(image_score_open, master=frame1)

    canvas.create_image(200,200, image=image_score)

    #Calculate R2 score
    print("Train score")
    print(train_score)
    t_train_r2.set(train_score)

    print("Test scores")
    print(test_score)
    t_test_r2.set(test_score)

    #Calculate RMSE
    train_rmse = 1
    test_rmse  = 1
    '''
    print("Train RMSE")
    print(train_rmse)
    t_train_rmse.set(train_rmse)

    print("Test RMSE")
    print(test_rmse)
    t_test_rmse.set(test_rmse)
    '''

    df_save = pd.DataFrame(
    {'pred_train':pred_train,
    'meas_train':y_train
    })

    df_save.to_csv('pred_and_meas_train.csv')

    print('finish!')



def fit_by_test():
    print('fit_by test')
    return

def fit_by_mordred():

    from keras.models import Sequential
    from keras.layers import Input, Activation, Conv2D, BatchNormalization, Flatten, Dense
    from keras.optimizers import SGD
    import tensorflow as tf

    theme_name = t_theme_name.get() + '-' + method_name
    df_mordred = pd.read_csv(t_csv_filepath.get())

    df_mordred['mols'] = df_mordred['smiles'].map(apply_molfromsmiles)
    df_mordred = df_mordred.dropna()
    print(df_mordred)

    calc =Calculator(descriptors, ignore_3D = True)


    X = calc.pandas(df_mordred['mols']).astype('float').dropna(axis = 1)

    X = np.array(X, dtype = np.float32)


    #各記述子について平均0, 分散1に変換
    st = StandardScaler()
    X= st.fit_transform(X)


    #後で再利用するためにファイルに保存
    #np.save(parent_path_str + os.sep + 'models' + os.sep + theme_name + os.sep + 'mordred_model' + os.sep + 'X_2d.npy', X)
    np.save(os.path.join(parent_path, 'models', theme_name, 'mordred_model', 'X_2d.npy'), X)

    y = df_mordred[t_task.get()]

    #Numpy形式の配列に変換
    y = np.array(Y, dtype = np.float32)


    #後で再利用するためにファイルに保存
    #np.save(parent_path_str + os.sep + 'models' + os.sep + theme_name + os.sep + 'mordred_model' + os.sep + 'Y_2d.npy', Y)
    np.save(os.path.join(parent_path, 'models', theme_name, 'mordred_model', 'Y_2d.npy'), Y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y, test_size=0.25, random_state=42)

    model = Sequential()
    #入力層．Denseは全結合層の意味．次の層に渡される次元は50．入力データの次元（input_dim）は1114．
    model.add(Dense(units = 2400, input_dim = X.shape[1]))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    #出力層．次元1，つまり一つの値を出力する．
    model.add(Dense(units = 1))
    model.summary()

    model.compile(loss = 'mean_squared_error',
    optimizer = 'Adam',
    metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs = max(5, int(t_epochs.get())), batch_size = 32,
    validation_data = (X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose = 0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    y_pred = model.predict(X_test)
    rms = (np.mean((y_test - y_pred) ** 2)) ** 0.5
    #s = np.std(y_test - y_pred)
    print("Neural Network RMS", rms)

    #model.save(parent_path_str + os.sep + 'models' + os.sep + theme_name + os.sep + 'mordred_model' + os.sep  + 'model.h5')
    model.save(os.path.join(parent_path, 'models', theme_name, 'mordred_model', 'model.h5'))

    pred_train = model.predict(X_train, batch_size = 32)
    pred_test  = model.predict(X_test, batch_size = 32)


    print('pred_train')
    print(pred_train)
    print('y_train')
    print(y_train)

    df_save = pd.DataFrame(
        {'pred_train':pred_train,
        'meas_train':y_train
        })


    df_save.to_csv('pred_and_meas_train.csv')

    plt.figure(figsize=(5,5))
    plt.scatter(y_train, pred_train, label = 'Train', c = 'blue')
    plt.title('Mordred predict')
    plt.xlabel('Measured value')
    plt.ylabel('Predicted value')
    plt.scatter(y_test, pred_test, c = 'lightgreen', label = 'Test', alpha = 0.8)
    plt.legend(loc = 4)
    #plt.show()

    #plt.savefig(parent_path_str + os.sep + 'models' + os.sep + theme_name + os.sep + 'mordred_model' + os.sep  + 'score-tmp.png')
    plt.savefig(parent_path / 'results' / theme_name / 'moredred' / 'scatter.png')

    global image_score
    image_score_open = Image.open(os.path.join(parent_path / 'results' / theme_name / 'moredred' / 'scatter.png'))
    image_score = ImageTk.PhotoImage(image_score_open, master=frame1)

    canvas.create_image(200,200, image=image_score)




def fit_data():
    theme_name = t_theme_name.get() + '-' + method_name
    chk_mkdir(theme_name)

    train_by_mordred = Booleanvar_mordred.get()
    train_by_deepchem = Booleanvar_deepchem.get()
    train_by_chainer_chemistry = Booleanvar_chainer_chemistry.get()

    if os.name == 'nt' and train_by_deepchem == True:
        print(' - 起動環境はWindowsです - ')
        print(' - deepchemは利用できません - ')
        train_by_deepchem = False

        if train_by_mordred == False:
            train_by_mordred = True

    if train_by_mordred == True:
        print('fit by mordred')
        fit_by_mordred()
    elif train_by_deepchem == True:
        print('fit by deepchem')
        fit_by_deepchem()
    elif train_by_chainer_chemistry == True:
        print('fit by chainer')
        fit_by_chainer_chemistry()
        make_virtual_lib()
        predict_by_chainer_chemistry()
        print('finish ' , t_theme_name.get())

        return



# tkinter
root = tkinter.Tk()
root.title('test-horie')

#font and style
font1 = font.Font(family='游ゴシック', size=10, weight='bold')
root.option_add("*Font", font1)
root.option_add("*Button.font", font1)

style1  = ttk.Style()
style1.configure('my.TButton', font = ('游ゴシック',10)  )


#frame1
frame1  = tkinter.ttk.Frame(root)

#label and entry
label_csv  = tkinter.ttk.Label(frame1, text = 'csv file path:')
t_csv_filepath = tkinter.StringVar()
t_csv_filename = tkinter.StringVar()

entry_csv  = ttk.Entry(frame1, textvariable=t_csv_filename, width = 60)

button_getcsv = ttk.Button(frame1, text='CSVデータの選択', command = get_csv, style = 'my.TButton')

label_task = tkinter.ttk.Label(frame1, text = '特性値の列名:')
label_smiles = tkinter.ttk.Label(frame1, text = 'smilesの列名:')
label_id = tkinter.ttk.Label(frame1, text = 'IDの列名:')
label_themename = tkinter.ttk.Label(frame1, text = '保存フォルダ名:')

t_task = tkinter.StringVar()
t_smiles = tkinter.StringVar()
t_id = tkinter.StringVar()
t_theme_name = tkinter.StringVar()

entry_task = ttk.Entry(frame1, textvariable=t_task, width = 60)
entry_smiles = ttk.Entry(frame1, textvariable=t_smiles, width = 60)
entry_id = ttk.Entry(frame1, textvariable=t_id, width = 60)
entry_themename = ttk.Entry(frame1, textvariable=t_theme_name, width = 60)

t_epochs = tkinter.StringVar()
t_epochs.set(50)
label_epochs = tkinter.ttk.Label(frame1, text = '学習回数:')
entry_epochs = ttk.Entry(frame1, textvariable=t_epochs, width = 60)

Booleanvar_mordred = tkinter.BooleanVar()
Booleanvar_deepchem = tkinter.BooleanVar()
Booleanvar_chainer_chemistry = tkinter.BooleanVar()
Booleanvar_transfer_learning = tkinter.BooleanVar()

Booleanvar_mordred.set(False)
Booleanvar_deepchem.set(False)
Booleanvar_chainer_chemistry.set(True)
Booleanvar_transfer_learning.set(True)

#Checkbutton_mordred = tkinter.Checkbutton(frame1, text = 'mordred', variable = Booleanvar_mordred)
#Checkbutton_deepchem = tkinter.Checkbutton(frame1, text = 'deepchem', variable = Booleanvar_deepchem)
Checkbutton_chainer_chemistry = tkinter.Checkbutton(frame1, text = 'chainer-chemistry', variable = Booleanvar_chainer_chemistry)
Checkbutton_transfer_learning = tkinter.Checkbutton(frame1, text = '転移学習', variable = Booleanvar_transfer_learning)


button_fit = ttk.Button(frame1, text = '訓練開始', command = fit_data, style = 'my.TButton')

label_train_r2 = tkinter.ttk.Label(frame1, text = '訓練用データのR2 score:')
t_train_r2 = tkinter.StringVar()
entry_train_r2 = ttk.Entry(frame1, textvariable=t_train_r2, width = 60)

label_test_r2 = tkinter.ttk.Label(frame1, text = 'テスト用データのR2 score:')
t_test_r2 = tkinter.StringVar()
entry_test_r2 = ttk.Entry(frame1, textvariable=t_test_r2, width = 60)

'''
label_train_rmse = tkinter.ttk.Label(frame1, text = '訓練用データのRMSE score:')
t_train_rmse = tkinter.StringVar()
entry_train_rmse = ttk.Entry(frame1, textvariable=t_train_rmse, width = 60)

label_test_rmse = tkinter.ttk.Label(frame1, text = 'テスト用データのRMSE score:')
t_test_rmse = tkinter.StringVar()
entry_test_rmse = ttk.Entry(frame1, textvariable=t_test_rmse, width = 60)
'''

frame1.grid(row=0,column=0,sticky=(N,E,S,W))
label_csv.grid(row=1,column=1,sticky=E)
entry_csv.grid(row=1,column=2,sticky=W)
button_getcsv.grid(row=2,column=2,sticky=W)

label_task.grid(row=4,column=1,sticky=E)
label_smiles.grid(row=5,column=1,sticky=E)
label_id.grid(row=6,column=1,sticky=E)
label_themename.grid(row=7,column=1,sticky=E)
label_epochs.grid(row=8,column=1,sticky=E)


entry_task.grid(row=4,column=2,sticky=W)
entry_smiles.grid(row=5,column=2,sticky=W)
entry_id.grid(row=6,column=2,sticky=W)
entry_themename.grid(row=7,column=2,sticky=W)
entry_epochs.grid(row=8,column=2,sticky=W)

#Checkbutton_mordred.grid(           row=9, column=2,sticky=W)
#Checkbutton_deepchem.grid(          row=9,column=3,sticky=W)
Checkbutton_chainer_chemistry.grid( row=9,column=2,sticky=W)
Checkbutton_transfer_learning.grid( row=9,column=3,sticky=W)

button_fit.grid(row=10,column=1,sticky=W)


label_train_r2.grid(row=12,column=1,sticky=E)
entry_train_r2.grid(row=12,column=2,sticky=W)

label_test_r2.grid(row=13,column=1,sticky=E)
entry_test_r2.grid(row=13,column=2,sticky=W)

#label_train_rmse.grid(row=13,column=1,sticky=E)
#entry_train_rmse.grid(row=13,column=2,sticky=W)

#label_test_rmse.grid(row=14,column=1,sticky=E)
#entry_test_rmse.grid(row=14,column=2,sticky=W)

frame2 = tkinter.Toplevel()
frame2.title('graph')
frame2.geometry('800x800')
frame2.grid()
photo_size = 400


canvas = tkinter.Canvas(frame2, width = 400, height = 400)
canvas.grid(row=1, column = 1)

global image_score
image_score_open = Image.open(os.path.join('logo', 'sample1.png'))
image_score = ImageTk.PhotoImage(image_score_open, master=frame1)
canvas.create_image(int(photo_size/2), int(photo_size/2), image=image_score)

for child in frame1.winfo_children():
    child.grid_configure(padx=5, pady=5)

for child in frame2.winfo_children():
    child.grid_configure(padx=5, pady=5)

root.mainloop()
