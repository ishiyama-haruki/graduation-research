# coding : utf-8

from __future__ import print_function, absolute_import, division, unicode_literals
import sys, bz2, lzma
import numpy as np
# import theano

def preprocess( datafile_name, xdic=None, ydic=None, bz2_f=True ):
    X_org = []
    Y_org = []
    
    if xdic is None:
        xdic = {}
    if ydic is None:
        ydic = {}

    if bz2_f:
        print('load bz2data')
        sys.stdout.flush()
        fin = bz2.BZ2File(datafile_name)
        lines = [ s.split() for s in fin.readlines() ]
        fin.close()
    else:
        if 'SUSY' in datafile_name:
            with lzma.LZMAFile(datafile_name, 'rb') as fin:
                lines = [ s.split() for s in fin.readlines() ]
                fin.close()
        else:
            fin = open(datafile_name,u'r')
            lines = [ s.split() for s in fin.readlines() ]
            fin.close()


    print('preprocess')
    sys.stdout.flush()

    for i,l in enumerate(lines):
        if int( i % 0.1*len(lines) ) == 0:
            print ('.', end='' )

        y_ = int(l[0])
        # エラーが出るので対処
        if ('letter' in datafile_name or 'dna' in datafile_name or 'shuttle' in datafile_name):
            kvs = [ (int(s.split(u':')[0]), float(s.split(u':')[1]) ) for s in l[1:] ]
        else:
            if 'SUSY' in datafile_name:
                # susyでのエラー対処
                for s in l[1:]: 
                    if  s.split(b':')[1] == b'-9.537825584411621094-01':
                        kvs = [ (int(s.split(b':')[0]), float(b'-9.537825584411621094e-01') ) ]
                    else:
                        kvs = [ (int(s.split(b':')[0]), float(s.split(b':')[1]) ) ]
            else:
                kvs = [ (int(s.split(b':')[0]), float(s.split(b':')[1]) ) for s in l[1:] ]

    

        if y_ not in ydic:
            label = len(ydic)
            ydic[y_] = label

        for k,v in kvs:
            if k not in xdic:
                key = len(xdic)
                xdic[k] = key
            
        Y_org.append(y_)
        X_org.append(kvs)
    print('')
    return X_org, Y_org, xdic, ydic        

def binarize( X_org, Y_org, xdic, ydic ):
    X = []    
    Y = []
    dim = len(xdic)

    for y_,x_ in zip(Y_org,X_org):
        y = ydic[y_]
        x = [0] * dim
        for k,v in x_:
            x[ xdic[k] ] = v

        Y.append(y)
        X.append(x)

    Y = np.asarray( Y, dtype=u'int32' )
    X = np.asarray( X, dtype=u'float32' )

    return X, Y
