
from __future__ import print_function
#© Copyright 2019 Yuki Horie

print('importing libraries...')
import os
import sys

from csv import reader
import random
from pathlib import Path
from argparse import ArgumentParser
import pprint
import glob

import tkinter
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import ttk
from tkinter import N, E, S, W
from tkinter import font

import pdb

import numpy as np
import numpy # Necessary
import pandas as pd

import cloudpickle

#scikit-learn
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import sascorer # need fpscores.pkl.gz, sascorer.py

import matplotlib.pyplot as plt
from PIL import ImageTk, Image, ImageDraw, ImageFont

# chemical
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem import BRICS
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import rdMolDraw2D

import pubchempy as pcp

import sascorer
#from mordred import descriptors, Calculator

print('importing chainer - wait a minute')
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
from chainer_chemistry.models import ggnn  # NOQA
from chainer_chemistry.models import gin  # NOQA
from chainer_chemistry.models import gwm  # NOQA
from chainer_chemistry.models import mlp  # NOQA
from chainer_chemistry.models import mpnn  # NOQA
from chainer_chemistry.models import nfp  # NOQA
from chainer_chemistry.models import prediction  # NOQA
from chainer_chemistry.models import relgat  # NOQA
from chainer_chemistry.models import relgcn  # NOQA
from chainer_chemistry.models import rsgcn  # NOQA
from chainer_chemistry.models import schnet  # NOQA
from chainer_chemistry.models import weavenet  # NOQA

from chainer_chemistry.models.ggnn import GGNN  # NOQA
from chainer_chemistry.models.ggnn import SparseGGNN  # NOQA
from chainer_chemistry.models.gin import GIN  # NOQA
from chainer_chemistry.models.mlp import MLP  # NOQA
from chainer_chemistry.models.mpnn import MPNN  # NOQA
from chainer_chemistry.models.nfp import NFP  # NOQA
from chainer_chemistry.models.relgat import RelGAT  # NOQA
from chainer_chemistry.models.relgcn import RelGCN  # NOQA
from chainer_chemistry.models.rsgcn import RSGCN  # NOQA
from chainer_chemistry.models.schnet import SchNet  # NOQA
from chainer_chemistry.models.weavenet import WeaveNet  # NOQA
from chainer_chemistry.models.gnn_film import GNNFiLM  # NOQA

from chainer_chemistry.models.gwm.gwm_net import GGNN_GWM  # NOQA
from chainer_chemistry.models.gwm.gwm_net import GIN_GWM  # NOQA
from chainer_chemistry.models.gwm.gwm_net import NFP_GWM  # NOQA
from chainer_chemistry.models.gwm.gwm_net import RSGCN_GWM  # NOQA

from chainer_chemistry.models.prediction import set_up_predictor
from chainer_chemistry.training.extensions.auto_print_report import AutoPrintReport
from chainer.training.extensions import Evaluator

from chainer_chemistry.links.scaler.standard_scaler import StandardScaler  # NOQA
from chainer_chemistry.models.prediction import GraphConvPredictor  # NOQA
from chainer_chemistry.utils import run_train
from chainer_chemistry.utils import save_json

print('finish importing the libraries')

################################################################################
# refer https://horomary.hatenablog.com/entry/2018/10/21/122025
# refer https://www.ag.kagawa-u.ac.jp/charlesy/2017/07/27/deepchem%E3%81%AB%E3%82%88%E3%82%8B%E6%BA%B6%E8%A7%A3%E5%BA%A6%E4%BA%88%E6%B8%AC-graph-convolution-%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF/
# refer https://future-chem.com/rdkit-intro/
################################################################################

# default setting of chainer chemistry
training_method = ['nfp', 'weavenet', 'rsgcn', 'mpnn']
training_method = ['ggnn']
training_method = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn','relgat']
#You can choose from ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn','relgat', 'mpnn']
# Low  complexity model : nfp, weavenet, mpnn
# High complexity model : schnet, relgat,
# Complexity indication
method_complexity = {'nfp':0.3, 'ggnn':0.4, 'schnet':11, 'weavenet':0.1,
                     'rsgcn':0.1, 'relgcn':2,'relgat':1.5}

complexity_degree = {'high':30, 'middle':10, 'low':1}


default_epochs=30
virtual_libraly_num = 200
default_transfer_source = 'Lipophilicity'

# paths
current_path = Path.cwd()
program_path = Path(__file__).parent   # x.pyのあるディレクトリ 相対パス
parent_path = program_path / '../'     # ディレクトリ移動
parent_path_str = str(parent_path.resolve()) #絶対パスに変換してから、文字列に変換　（C\desktop\sssss\aaaaa\） という文字列になる

data_path           = parent_path / 'data'
data_processed_path = data_path / 'processed'
models_path = parent_path / 'models'

def chk_mkdir(theme_name, method_name, high_low):

    paths =[parent_path / 'results' / theme_name / method_name / high_low / 'predict',
            parent_path / 'results' / theme_name / method_name / high_low / 'search',
            parent_path / 'results' / theme_name / method_name / high_low / 'brics_virtual'  / 'molecular-structure',
            parent_path / 'models'  / theme_name / method_name / high_low
            ]

    for path_name in paths:
        os.makedirs(path_name, exist_ok=True)
    return


def select_folder():
    current_dir = os.getcwd()
    model_path = askdirectory(initialdir = models_path,
                              title = 'choose the source of model for transfer learning')

    print('before', model_path)
    model_path = Path(model_path)
    print('after', model_path)

    t_model_name.set(str(Path(model_path).name))
    t_model_path.set(model_path)


def get_csv():
    current_dir = os.getcwd()

    csv_file_path = askopenfilename(initialdir = data_processed_path,
                                    title = 'choose the csv',
                                    filetypes = [('csv file', '*.csv')])


    t_csv_filename.set(str(Path(csv_file_path).name))
    t_csv_filepath.set(csv_file_path)

    t_theme_name.set(Path(csv_file_path).parent.name)

    with open(t_csv_filepath.get()) as f:
        reader_ = reader(f)
        l = [row for row in reader_]

        t_id.set(l[0][0])       #１列目の列名を取得：IDや名前
        t_task.set(l[0][1])     #２列目の列名を取得：目的関数（例：水の溶解度）
        t_smiles.set(l[0][2])   #３列目の列名を取得：SMILES表記

def apply_molfromsmiles(smiles_name):
    try:
        mols = Chem.MolFromSmiles(smiles_name)

    except:
        #SMILESからMOLに変換できなかった場合、""を返す
        mols = ""
        print(smiles_name)
        print('Error : ', smiles_name, ' couldn\'t be converted to MOL, skipped')

    return mols

def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn', 'relgat', 'mpnn', 'gnnfilm']
    scale_list = ['standardize', 'none']

    # Set up the argument parser.
    parser = ArgumentParser(description='Regression on own dataset')

    parser.add_argument('--datafile', '-d', type=str,
                        default='dataset_train.csv',
                        help='csv file containing the dataset')

    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='nfp')

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

    parser.add_argument('--epoch', '-e', type=int, default=default_epochs,
                        help='number of epochs')

    parser.add_argument('--unit-num', '-u', type=int, default=16,
                        help='number of units in one layer of the model')

    parser.add_argument('--seed', '-s', type=int, default=777,
                        help='random seed value')

    parser.add_argument('--train-data-ratio', '-r', type=float, default=0.7,
                        help='ratio of training data w.r.t the dataset')

    parser.add_argument('--protocol', type=int, default=2,
                        help='pickle protocol version')

    parser.add_argument('--model-foldername', type=str, default='chainer',
                        help='saved model foldername')

    parser.add_argument('--model-filename', type=str, default='regressor.pkl',
                        help='saved model filename')

    parser.add_argument('--source-transferlearning', type=str, default=parent_path / 'models' / default_transfer_source ,
                        help='source model of transfer learning')

    return parser.parse_args()

def rmse(x0, x1):
    return functions.sqrt(functions.mean_squared_error(x0, x1))


def save_scatter(Y_train, pred_train, Y_test, pred_test, title, save_path):
    plt_sct = plt.figure(figsize=(5,5))

    plt.xlabel('Measured value')
    plt.ylabel('Predicted value')
    plt.title(title)
    plt.scatter(Y_train, pred_train, label = 'Train', c = 'blue')
    plt.scatter(Y_test, pred_test, c = 'lightgreen', label = 'Test', alpha = 0.8)
    plt.legend(loc = 4)
    plt.savefig(save_path)
    #plt.close(plt_sct)



def fit_by_chainer_chemistry(theme_name, method_name, high_low):

    def main():
        # Parse the arguments.
        args = parse_arguments()

        args.model_folder_name = os.path.join(theme_name , 'chainer')


        args.epoch = int(epoch_high_low * 60 / method_complexity[method_name])
        args.epoch = max(args.epoch, 5)

        #args.epoch = int(float(t_epochs.get()))
        args.out = parent_path / 'models' / theme_name / method_name
        args.method = method_name

        if t_model_path != "":
            args.source_transferlearning = Path(t_model_path.get())

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

        print((args.source_transferlearning / method_name / high_low /'regressor.pickle' ))
        print((args.source_transferlearning / method_name / high_low /'regressor.pickle' ).exists())

        # Set up the predictor.

        if  Booleanvar_transfer_learning.get() == True  \
              and (args.source_transferlearning / method_name / high_low /'regressor.pickle').exists() == True:

            # refer https://github.com/pfnet-research/chainer-chemistry/issues/407
            with open(args.source_transferlearning / method_name / high_low /'regressor.pickle',  'rb') as f:
                regressor = cloudpickle.loads(f.read())
                pre_predictor = regressor.predictor
                predictor = GraphConvPredictor(pre_predictor.graph_conv, MLP(out_dim=1, hidden_dim=16))

        else :
            predictor = set_up_predictor(
            args.method, args.unit_num,
            args.conv_layers, class_num, label_scaler=scaler)


        # Set up the regressor.
        device = chainer.get_device(args.device)
        metrics_fun = {'mae': functions.mean_absolute_error, 'rmse': rmse}

        regressor = Regressor(predictor, lossfun=functions.mean_squared_error,
                              metrics_fun=metrics_fun, device=device)

        print('Training... : ' , method_name)
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


        with open(parent_path / 'models' / theme_name / method_name / high_low / ('regressor.pickle'),  'wb') as f:
            cloudpickle.dump(regressor, f)

        #with open(parent_path / 'models' / theme_name / method_name / high_low /('predictor.pickle'),  'wb') as f:
        #    cloudpickle.dump(predictor, f)


        print('Evaluating... : ' , method_name)
        test_iterator = SerialIterator(testset, 16, repeat=False, shuffle=False)
        eval_result = Evaluator(test_iterator, regressor, converter=concat_mols,
                                device=device)()
        print('Evaluation result: : ' , method_name)
        print(eval_result)

        @chainer.dataset.converter()
        def extract_inputs(batch, device=None):
            return concat_mols(batch, device=device)[:-1]


        pred_train = regressor.predict(trainset, converter=extract_inputs)
        pred_train = [i[0] for i in pred_train]
        pred_test = regressor.predict(testset, converter=extract_inputs)
        pred_test = [i[0] for i in pred_test]

        y_train = [i[2][0] for i in trainset]
        y_test = [i[2][0] for i in testset]
        title = args.label
        save_path = parent_path / 'results' / theme_name / method_name / high_low /'scatter.png'
        save_scatter(y_train, pred_train, y_test, pred_test, title ,save_path)

        global image_score
        image_score_open = Image.open(parent_path / 'results' / theme_name / method_name / high_low /'scatter.png')
        image_score = ImageTk.PhotoImage(image_score_open, master=frame1)

        canvas.create_image(200,200, image=image_score)


        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from sklearn.metrics import r2_score

        train_mse = mean_squared_error(y_train, pred_train)
        test_mse  = mean_squared_error(y_test, pred_test)

        train_rmse = np.sqrt(train_mse)
        test_rmse  = np.sqrt(test_mse)

        train_mae = mean_absolute_error(y_train, pred_train)
        test_mae  = mean_absolute_error(y_test, pred_test)

        train_r2score = r2_score(y_train, pred_train)
        test_r2score  = r2_score(y_test, pred_test)


        print('train_mse : ', train_mse)
        print('test_mse : ', test_mse)
        print('train_rmse : ', train_rmse)
        print('test_rmse : ', test_rmse)
        print('train_mae : ', train_mae)
        print('test_mae : ', train_mae)
        print('train_r2score : ', train_r2score)
        print('test_r2score : ', test_r2score)



    main()

def predict_by_chainer_chemistry(method_name, csv_path, data_name):
    def main():
        # Parse the arguments.
        args = parse_arguments()
        theme_name = t_theme_name.get()

        args.model_folder_name = os.path.join(theme_name , 'chainer')
        #args.epoch = int(float(t_epochs.get()))
        args.out = parent_path / 'models' / theme_name / method_name
        args.method = method_name


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

        #args.datafile=parent_path / 'results' /  theme_name / method_name / high_low /'brics_virtual'  / 'virtual.csv'
        args.datafile=csv_path
        dataset = parser.parse(args.datafile)['dataset']

        @chainer.dataset.converter()
        def extract_inputs(batch, device=None):
            return concat_mols(batch, device=device)[:-1]

        print('Predicting the virtual library')
        # Set up the regressor.
        device = chainer.get_device(args.device)
        model_path = os.path.join(args.out, args.model_foldername, args.model_filename)

        with open(parent_path / 'models' / theme_name / method_name / high_low /('regressor.pickle'), 'rb') as f:
            regressor = cloudpickle.loads(f.read())

        # Perform the prediction.
        print('Evaluating...')
        converter = converter_method_dict[args.method]
        data_iterator = SerialIterator(dataset, 16, repeat=False, shuffle=False)
        eval_result = Evaluator(data_iterator, regressor, converter=converter,
                                device=device)()
        print('Evaluation result: ', eval_result)

        predict_ = regressor.predict(dataset, converter=extract_inputs)
        predict_ = [i[0] for i in predict_]
        df_data = pd.read_csv(csv_path)

        df_predict = df_data
        df_predict[t_task.get()] = predict_
        df_predict =df_predict.dropna()

        PandasTools.AddMoleculeColumnToFrame(frame=df_predict, smilesCol=t_smiles.get())
        df_predict['sascore'] = df_predict.ROMol.map(sascorer.calculateScore)

        df_predict.to_csv(csv_path)

        png_generator = (parent_path / 'results' /  theme_name / method_name / high_low /data_name / 'molecular-structure').glob('*.png')
        #png_generator.sort()

        for i, png_path in enumerate(png_generator):
            #print((png_path.name)[4:10])
            i = int((png_path.name)[4:10])
            if i < len(df_predict[t_task.get()]):
                img = Image.open(png_path)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype('arial.ttf', 26)
                draw.text((0,0), t_task.get() + ' : ' + str(round(df_predict[t_task.get()][i],2)),  (0,0,0), font=font)
                draw.text((0,30),  'sascore : ' + str(round(df_predict['sascore'][i],2)),  (0,0,0), font=font)

                img.save(png_path)

        save_json(os.path.join(args.out,'eval_result.json'), eval_result)
    main()

def make_virtual_lib(method_name):
    theme_name = t_theme_name.get()
    df_brics = pd.read_csv(t_csv_filepath.get())
    df_brics['mols'] = df_brics[t_smiles.get()].map(apply_molfromsmiles)
    df_brics = df_brics.dropna()

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

    for f in allfrags:
        #print('f: ', f)
        #print('Mol: ',Chem.MolFromSmiles(f))
        pass

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
    print('The total number : ', virtual_libraly_num)
    print('The number of error : ', error_cnt)
    print('The ratio of error : ', error_cnt / virtual_libraly_num)

    for i, mol in enumerate(virtual_mols):
        Draw.MolToFile(mol, str(parent_path / 'results' /  theme_name / method_name / high_low /'brics_virtual'  / 'molecular-structure' / ('tmp-' + str(i).zfill(6) + '.png')))


    virtual_list = []
    for i, mol in enumerate(virtual_mols):
        virtual_list.append([i, Chem.MolToSmiles(mol), 0])

    #print(virtual_list)
    df_virtual = pd.DataFrame(virtual_list,
                              columns=[t_id.get(), t_smiles.get(), t_task.get()])

    #print(df_virtual)
    csv_path =parent_path / 'results' /  theme_name / method_name / high_low /'brics_virtual'  / 'virtual.csv'
    df_virtual.to_csv(csv_path)
    return csv_path


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

    model.fit(dataset, nb_epoch=max(1, int(100)))

    metric = dc.metrics.Metric(dc.metrics.r2_score)

    print('epoch: ', 100 )
    print("Evaluating model")
    train_score = model.evaluate(trainset, [metric])
    test_score  = model.evaluate(testset, [metric])

    model.save()

    pred_train = model.predict(trainset)
    pred_test  = model.predict(testset)

    y_train = np.array(trainset.y, dtype = np.float32)
    y_test = np.array(testset.y, dtype = np.float32)

    plt.figure(figsize=(5,5))

    plt.scatter(y_train, pred_train, label = 'Train', c = 'blue')
    plt.title('Graph Convolution')
    plt.xlabel('Measured value')
    plt.ylabel('Predicted value')
    plt.scatter(y_test, pred_test, c = 'lightgreen', label = 'Test', alpha = 0.8)
    plt.legend(loc = 4)
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

    print('All Finished!')


def fit_by_mordred():

    from keras.models import Sequential
    from keras.layers import Input, Activation, Conv2D, BatchNormalization, Flatten, Dense
    from keras.optimizers import SGD
    import tensorflow as tf

    theme_name = t_theme_name.get()
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
    history = model.fit(X_train, y_train, epochs = max(5, int(100)), batch_size = 32,
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
    plt.savefig(parent_path / 'results' / theme_name / method_name / high_low /'moredred' / 'scatter.png')

    global image_score
    image_score_open = Image.open(os.path.join(parent_path / 'results' / theme_name / method_name / high_low /'moredred' / 'scatter.png'))
    image_score = ImageTk.PhotoImage(image_score_open, master=frame1)
    canvas.create_image(200,200, image=image_score)


def training_and_searching():
    for method_name in training_method:
        train_by_chainer_chemistry = Booleanvar_chainer_chemistry.get()
        print('fit by chainer')

        high_low = str(var_epoch.get())
        theme_name = t_theme_name.get()

        chk_mkdir(theme_name,  method_name, high_low)

        fit_by_chainer_chemistry(theme_name, method_name, high_low)

        print('start making virtual library')
        #virtual_csv_path = make_virtual_lib(method_name)

        print('start predict the virtual library')
        #predict_by_chainer_chemistry(method_name, virtual_csv_path, 'brics_virtual')

        print('finish ' , t_theme_name.get())
    return


# tkinter
root = tkinter.Tk()
root.title('Chemical Data Science')

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

entry_csv  = ttk.Entry(frame1, textvariable=t_csv_filename, width = 15)

button_getcsv = ttk.Button(frame1, text='CSVデータの選択', command = get_csv, style = 'my.TButton')

label_task = tkinter.ttk.Label(frame1, text = '特性値の列名:')
label_smiles = tkinter.ttk.Label(frame1, text = 'smilesの列名:')
label_id = tkinter.ttk.Label(frame1, text = 'IDの列名:')
label_themename = tkinter.ttk.Label(frame1, text = '保存フォルダ名:')

t_task = tkinter.StringVar()
t_smiles = tkinter.StringVar()
t_id = tkinter.StringVar()
t_theme_name = tkinter.StringVar()

entry_task = ttk.Entry(frame1, textvariable=t_task, width = 15)
entry_smiles = ttk.Entry(frame1, textvariable=t_smiles, width = 15)
entry_id = ttk.Entry(frame1, textvariable=t_id, width = 15)
entry_themename = ttk.Entry(frame1, textvariable=t_theme_name, width = 15)

t_epochs = tkinter.StringVar()
t_epochs.set(default_epochs)
label_epochs = tkinter.ttk.Label(frame1, text = '学習回数:')
#entry_epochs = ttk.Entry(frame1, textvariable=t_epochs, width = 60)

var_epoch = tkinter.StringVar()
var_epoch.set('low')

Radiobutton_bayesian_max = tkinter.Radiobutton(frame1, value = 'high',   text = 'とことん', variable = var_epoch)
Radiobutton_bayesian_avg = tkinter.Radiobutton(frame1, value = 'middle', text = '普通', variable = var_epoch)
Radiobutton_bayesian_min = tkinter.Radiobutton(frame1, value = 'low',    text = '軽く', variable = var_epoch)


Booleanvar_chainer_chemistry = tkinter.BooleanVar()
Booleanvar_transfer_learning = tkinter.BooleanVar()
Booleanvar_chainer_chemistry.set(True)
Booleanvar_transfer_learning.set(False)

Checkbutton_chainer_chemistry = tkinter.Checkbutton(frame1, text = 'chainer-chemistry', variable = Booleanvar_chainer_chemistry)
Checkbutton_transfer_learning = tkinter.Checkbutton(frame1, text = '転移学習', variable = Booleanvar_transfer_learning)

t_model_name = tkinter.StringVar()
t_model_path = tkinter.StringVar()

t_model_name.set(default_transfer_source)

entry_model_name = ttk.Entry(frame1, textvariable = t_model_name, width =15)
button_change_transfersource = ttk.Button(frame1, text = '転移学習　元モデル変更', command = select_folder, style = 'my.TButton')
button_fit = ttk.Button(frame1, text = '訓練開始', command = training_and_searching, style = 'my.TButton')

label_train_r2 = tkinter.ttk.Label(frame1, text = '訓練用データのR2 score:')
t_train_r2 = tkinter.StringVar()
entry_train_r2 = ttk.Entry(frame1, textvariable=t_train_r2, width = 15)

label_test_r2 = tkinter.ttk.Label(frame1, text = 'テスト用データのR2 score:')
t_test_r2 = tkinter.StringVar()
entry_test_r2 = ttk.Entry(frame1, textvariable=t_test_r2, width = 15)

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
#entry_epochs.grid(row=8,column=2,sticky=W)

Radiobutton_bayesian_max.grid(row=8,column=2,sticky=W)
Radiobutton_bayesian_avg.grid(row=8,column=3,sticky=W)
Radiobutton_bayesian_min.grid(row=8,column=4,sticky=W)


#Checkbutton_mordred.grid(           row=9, column=2,sticky=W)
#Checkbutton_deepchem.grid(          row=9,column=3,sticky=W)
Checkbutton_chainer_chemistry.grid( row=9,column=2,sticky=W)
Checkbutton_transfer_learning.grid( row=9,column=3,sticky=W)

button_change_transfersource.grid(row=10,column=1,sticky=W)
entry_model_name.grid(row=10, column=2, sticky=W)

button_fit.grid(row=11,column=1,sticky=W)

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
frame2.geometry('100x100')
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
