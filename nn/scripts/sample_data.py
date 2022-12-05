# config : utf-8

from __future__ import print_function, absolute_import, division, unicode_literals
import sys, os
import torch
from . import binarize_libsvm_data 
from . import data_loader
if sys.version_info[0] == 2:
    import cPickle
else:
    import _pickle as cPickle

def save( train_path, train_file, test_path=None, test_file=None ):
    if not os.path.exists( './data' ):
        os.mkdir( './data' )
    test_f = True if test_path is not None and test_file is not None else False
    bz2_f = True if os.path.splitext(train_path)[1] == '.bz2' else False

    
    if not os.path.exists( train_path ):
        os.system( 'wget %s' % train_path )
    if test_f:
        if not os.path.exists( test_path ):
            os.system( 'wget %s' % test_path )
        
    X_, Y_, xdic, ydic = binarize_libsvm_data.preprocess( train_path, bz2_f=bz2_f )
    if test_f:
        Xt_, Yt_, xdic, ydic = binarize_libsvm_data.preprocess( test_path, xdic, ydic, bz2_f=bz2_f )
    else:
        Xt_, Yt_ = None, None

    X, Y = binarize_libsvm_data.binarize( X_, Y_, xdic, ydic )
    fout = open( train_file, 'wb' )
    cPickle.dump( [X,Y], fout )
    fout.close()

    if test_f:
        Xt, Yt = binarize_libsvm_data.binarize( Xt_, Yt_, xdic, ydic )
        fout = open( test_file, 'wb' )
        cPickle.dump( [Xt,Yt], fout )
        fout.close()

def save_from_url( train_url, train_file, test_url=None, test_file=None ):
    if not os.path.exists( './data' ):
        os.mkdir( './data' )
    test_f = True if test_url is not None and test_file is not None else False
    bz2_f = True if os.path.splitext(train_url)[1] == '.bz2' else False

    train_raw = os.path.basename(train_url)
    if not os.path.exists( train_raw ):
        os.system( 'wget %s' % train_url )
    if test_f:
        test_raw = os.path.basename(test_url)
        if not os.path.exists( test_raw ):
            os.system( 'wget %s' % test_url )
        
    X_, Y_, xdic, ydic = binarize_libsvm_data.preprocess( train_raw, bz2_f=bz2_f )
    if test_f:
        Xt_, Yt_, xdic, ydic = binarize_libsvm_data.preprocess( test_raw, xdic, ydic, bz2_f=bz2_f )
    else:
        Xt_, Yt_ = None, None

    X, Y = binarize_libsvm_data.binarize( X_, Y_, xdic, ydic )
    fout = open( train_file, 'wb' )
    cPickle.dump( [X,Y], fout )
    fout.close()

    if test_f:
        Xt, Yt = binarize_libsvm_data.binarize( Xt_, Yt_, xdic, ydic )
        fout = open( test_file, 'wb' )
        cPickle.dump( [Xt,Yt], fout )
        fout.close()

n_batch = 128

def get_data(X, Y, Xt, Yt):
    train_dataset = torch.utils.data.TensorDataset(X, Y)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = n_batch,
        shuffle = True)

    test_dataset = torch.utils.data.TensorDataset(Xt, Yt)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = n_batch,
        shuffle = True)

    return train_dataset, train_dataloader, test_dataset, test_dataloader

normalize   = False
bias        = False

def get_ijcnn1():
    train_file = './data/ijcnn1.data'
    test_file = './data/ijcnn1.t.data'
    split       = False
    standardize = True
    scale       = False

    if not os.path.exists( train_file ) or not os.path.exists( test_file ):
        train_path = './data_raw/ijcnn1.bz2'
        test_path = './data_raw/ijcnn1.t.bz2' 
        save( train_path, train_file, test_path, test_file )
        
    X, Y, Xt, Yt = data_loader.load( train_file, test_file, split, 
                                     standardize, scale, normalize, bias )

    train_dataset, train_dataloader, test_dataset, test_dataloader = get_data(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xt), torch.from_numpy(Yt))
    return X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader
        
def get_mnist():
    train_file = './data/mnist.data'
    test_file = './data/mnist.t.data'
    split       = False
    standardize = False
    scale       = True

    if not os.path.exists( train_file ) or not os.path.exists( test_file ):
        train_path = './data_raw/mnist.scale.bz2'
        test_path = './data_raw/mnist.scale.t.bz2' 
        save( train_path, train_file, test_path, test_file )
        
    X, Y, Xt, Yt = data_loader.load( train_file, test_file, split, 
                                     standardize, scale, normalize, bias )

    train_dataset, train_dataloader, test_dataset, test_dataloader = get_data(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xt), torch.from_numpy(Yt))
    return X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader

def get_usps():
    train_file = './data/usps.data'
    test_file = './data/usps.t.data'
    split       = False
    standardize = False
    scale       = True

    if not os.path.exists( train_file ) or not os.path.exists( test_file ):
        train_path = './data_raw/usps.bz2'
        test_path = './data_raw/usps.t.bz2' 
        save( train_path, train_file, test_path, test_file )
        
    X, Y, Xt, Yt = data_loader.load( train_file, test_file, split, 
                                     standardize, scale, normalize, bias )

    train_dataset, train_dataloader, test_dataset, test_dataloader = get_data(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xt), torch.from_numpy(Yt))
    return X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader


def get_covtype():
    train_file = './data/covtype.data'
    test_file = None
    split       = True
    standardize = True
    scale       = False

    if not os.path.exists( train_file ):
        train_path = './data_raw/covtype.scale01.bz2'
        save( train_path, train_file, None, None )
        
    X, Y, Xt, Yt = data_loader.load( train_file, test_file, split, 
                                     standardize, scale, normalize, bias )

    train_dataset, train_dataloader, test_dataset, test_dataloader = get_data(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xt), torch.from_numpy(Yt))
    return X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader

def get_letter():
    train_file = './data/letter.data'
    test_file = './data/letter.t.data'
    split       = False
    standardize = False
    scale       = True

    if not os.path.exists( train_file ) or not os.path.exists( test_file ):
        train_path = './data_raw/letter.scale'
        test_path = './data_raw/letter.scale.t' 
        save( train_path, train_file, test_path, test_file )
        
    X, Y, Xt, Yt = data_loader.load( train_file, test_file, split, 
                                     standardize, scale, normalize, bias )

    train_dataset, train_dataloader, test_dataset, test_dataloader = get_data(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xt), torch.from_numpy(Yt))
    return X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader

def get_cifar10():
    train_file = './data/cifar10.data'
    test_file = './data/cifar10.t.data'
    split       = False
    standardize = False
    scale       = True

    if not os.path.exists( train_file ) or not os.path.exists( test_file ):
        train_path = './data_raw/cifar10.bz2'
        test_path = './data_raw/cifar10.t.bz2' 
        save( train_path, train_file, test_path, test_file )
        
    X, Y, Xt, Yt = data_loader.load( train_file, test_file, split, 
                                     standardize, scale, normalize, bias )

    train_dataset, train_dataloader, test_dataset, test_dataloader = get_data(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xt), torch.from_numpy(Yt))
    return X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader

def get_dna():
    train_file = './data/dna.data'
    test_file = './data/dna.t.data'
    split       = False
    standardize = False
    scale       = True

    if not os.path.exists( train_file ) or not os.path.exists( test_file ):
        train_path = './data_raw/dna.scale'
        test_path = './data_raw/dna.scale.t' 
        save( train_path, train_file, test_path, test_file )
        
    X, Y, Xt, Yt = data_loader.load( train_file, test_file, split, 
                                     standardize, scale, normalize, bias )

    train_dataset, train_dataloader, test_dataset, test_dataloader = get_data(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xt), torch.from_numpy(Yt))
    return X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader

def get_aloi():
    train_file = './data/aloi.data'
    test_file = None
    split       = True
    standardize = True
    scale       = False

    if not os.path.exists( train_file ):
        train_path = './data_raw/aloi.scale.bz2'
        save( train_path, train_file, None, None )
        
    X, Y, Xt, Yt = data_loader.load( train_file, test_file, split, 
                                     standardize, scale, normalize, bias )

    train_dataset, train_dataloader, test_dataset, test_dataloader = get_data(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xt), torch.from_numpy(Yt))
    return X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader

def get_sector():
    train_file = './data/sector.data'
    test_file = './data/sector.t.data'
    split       = False
    standardize = False
    scale       = True

    if not os.path.exists( train_file ) or not os.path.exists( test_file ):
        train_path = './data_raw/sector.scale.bz2'
        test_path = './data_raw/sector.t.scale.bz2' 
        save( train_path, train_file, test_path, test_file )
        
    X, Y, Xt, Yt = data_loader.load( train_file, test_file, split, 
                                     standardize, scale, normalize, bias )

    train_dataset, train_dataloader, test_dataset, test_dataloader = get_data(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xt), torch.from_numpy(Yt))
    return X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader

def get_shuttle():
    train_file = './data/shuttle.data'
    test_file = './data/shuttle.t.data'
    split       = False
    standardize = False
    scale       = True

    if not os.path.exists( train_file ) or not os.path.exists( test_file ):
        train_path = './data_raw/shuttle.scale'
        test_path = './data_raw/shuttle.scale.t' 
        save( train_path, train_file, test_path, test_file )
        
    X, Y, Xt, Yt = data_loader.load( train_file, test_file, split, 
                                     standardize, scale, normalize, bias )

    train_dataset, train_dataloader, test_dataset, test_dataloader = get_data(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xt), torch.from_numpy(Yt))
    return X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader


def get_susy():
    train_file = './data/susy.data'
    test_file = None
    split       = True
    standardize = True
    scale       = False

    if not os.path.exists( train_file ):
        train_path = './data_raw/SUSY.xz'
        save( train_path, train_file, None, None )
        
    X, Y, Xt, Yt = data_loader.load( train_file, test_file, split, 
                                     standardize, scale, normalize, bias )

    train_dataset, train_dataloader, test_dataset, test_dataloader = get_data(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xt), torch.from_numpy(Yt))
    return X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader


