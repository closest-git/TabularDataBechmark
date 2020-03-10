import os
import bz2
import numpy as np
import pandas as pd
import gzip
import shutil
import torch
import random
import warnings
import gc
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import QuantileTransformer
from category_encoders import LeaveOneOutEncoder
import matplotlib.pyplot as plt

def download(url, filename, delete_if_interrupted=True, chunk_size=4096):
    """ saves file from url to filename with a fancy progressbar """
    try:
        with open(filename, "wb") as f:
            print("Downloading {} > {}".format(url, filename))
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                total_length = int(total_length)
                with tqdm(total=total_length) as progressbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        if data:  # filter-out keep-alive chunks
                            f.write(data)
                            progressbar.update(len(data))
    except Exception as e:
        if delete_if_interrupted:
            print("Removing incomplete download {}.".format(filename))
            os.remove(filename)
        raise e
    return filename

'''
    partition each dataset into five parts[S1, S2, S3, S4,S5] for five-fold cross validation.
    In each fold, three parts for training, one part for validation, and the remaining part for test
'''
class TabularDataset:
    def RemoveConstant(self,X_samp, listX):    
        self.nFeat = X_samp.shape[1]    
        stds = np.std(X_samp, axis=0)
        keep = np.where(stds != 0)[0]
        keep = list(keep)
        self.zero_feats=[]
        if len(keep)<self.nFeat:            
            self.zero_feats = list(np.where(stds == 0)[0])
            print(f"====== TabularDataset::RemoveConstant zeros={len(self.zero_feats)} {self.zero_feats}")
            for i,X_ in enumerate(listX):
                if X_ is None:
                    continue
                listX[i] = X_[:,keep]
        return listX

    #确实有用，需要进一步分析
    def quantile_trans_(self, random_state, X_samp, listX, distri='normal', noise=0):
        quantile_train = np.copy(X_samp)
        if noise:
            stds = np.std(quantile_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)
            #assert np.max(noise_std)<0.01
            quantile_train += noise_std * np.random.randn(*quantile_train.shape)

        qt = QuantileTransformer(random_state=random_state, output_distribution=distri).fit(quantile_train)
        for i,X_ in enumerate(listX):
            if X_ is None:
                continue
            listX[i] = qt.transform(X_)
        return listX,qt

    def OnFeatInfo(self,feat_info,weight_1):
        assert weight_1 is not None
        nFeat = weight_1.shape[0]        
        fmax, fmin = torch.max(weight_1), torch.min(weight_1)
        picks = [i for i in range(nFeat) if weight_1[i] <0.9]     #非常奇怪，难以理解
        picks = list(range(nFeat))      #先用全部吧
        nPick = len(picks)
        if nPick<nFeat:
            feat_info = feat_info.iloc[picks, :]
            if hasattr(self,'X_train'):
                assert nFeat == self.X_train.shape[1]
                self.X_train = self.X_train[:, picks]
                self.X_valid = self.X_valid[:, picks]
                if self.X_test is not None:
                    self.X_test = self.X_test[:, picks]
            else:
                self.X = self.X[:, picks]
        return feat_info

    def Y_trans(self,Y,isPredict=True):
        method = self.Y_trans_method
        if isPredict:
            if method=="log":   #确实不行啊
                Y = np.exp(Y)-1
            elif method=="normal":
                Y = Y*self.accu_scale+self.Y_mu_0
            else:
                pass
        else:
            mu, std = self.y_train.mean(), self.y_train.std()
            print("====== Y_trans_\tmean = %.5f, std = %.5f" % (mu, std))
            self.Y_mu_0=mu;                 self.Y_std_0=std            
            self.accu_scale = 1

            if method=="log":
                Y = np.log(Y+1)
            elif method=="normal":   #
                Y = ((Y - self.Y_mu_0) / self.Y_std_0).astype(np.float32)
                #self.y_valid = ((self.y_valid - mu) / std).astype(np.float32)
                #if self.y_test is not None:     self.y_test = ((self.y_test - mu) / std).astype(np.float32)
                self.accu_scale = self.Y_std_0
            else:
                Y = Y.astype(np.float32)
                self.accu_scale=1
        return Y.astype(np.float32)

    def problem(self):
        if hasattr(self,"nClasses"):
            assert self.nClasses>0
            return "classification"
        else:
            return "regression"

    def onTrans(self,trans,config,pkl_path=None, train_index=None, valid_index=None, test_index=None):
        if pkl_path is not None:
            print("====== onTrans pkl_path={} ......".format(pkl_path))
        if pkl_path is not None and os.path.isfile(pkl_path):
            with open(pkl_path, "rb") as fp:
                [self.X_train,self.y_train,self.X_valid, self.y_valid,self.X_test,self.y_test,\
            self.quantile_noise,self.Y_trans_method,self.accu_scale,self.Y_mu_0, self.Y_std_0,self.zero_feats] = pickle.load(fp)
            if self.problem()=="classification":
                print(f"\tnClasses={self.nClasses}" )
            else:
                print("\tmean = %.5f, std = %.5f accu_scale =  %.5f" % (self.Y_mu_0, self.Y_std_0,self.accu_scale))
            gc.collect()
        else:
            if train_index is not None:
                print(f"====== TabularDataset_{trans}\tvalid_index={valid_index},train_index={train_index}......")
                self.X_train, self.y_train = self.X[train_index],self.Y[train_index]
                self.X_valid, self.y_valid = self.X[valid_index], self.Y[valid_index]
                if test_index is not None:
                    self.X_test, self.y_test = self.X[test_index], self.Y[test_index]
            else:
                print(f"====== TabularDataset_{trans}......")
            if hasattr(self,"nClasses"):    #classification
                #self.y_train = self.y_train.astype(np.float32)
                self.Y_trans_method,self.accu_scale,self.Y_mu_0, self.Y_std_0=None,None,None,None
            else:   #regression
                self.Y_trans_method = "normal"
                self.y_train = self.Y_trans(self.y_train,isPredict=False)
                self.y_valid = self.y_valid.astype(np.float32)
                if self.y_test is not None:     self.y_test = self.y_test.astype(np.float32)            
            
            t0=time.time()
            self.quantile_noise = 1.0e-3
            if trans=="Normal":
                mean = np.mean(self.X_train, axis=0)
                std = np.std(self.X_train, axis=0)
                self.X_train = (self.X_train - mean) / std
                self.X_valid = (self.X_valid - mean) / std
                self.X_test = (self.X_test - mean) / std
            elif trans=="Quantile":
                listX, _ = self.quantile_trans_(self.random_state, self.X_train,
                    [self.X_train, self.X_valid, self.X_test],distri='normal', noise=self.quantile_noise)
                self.X_train, self.X_valid, self.X_test = listX[0], listX[1], listX[2]            
                print(f"====== TabularDataset::quantile_transform X_train={self.X_train.shape} X_valid={self.X_valid.shape} noise={self.quantile_noise} time={time.time()-t0:.5f}")
                gc.collect()

            if pkl_path is not None:
                with open(pkl_path, "wb") as fp:
                    pickle.dump([self.X_train,self.y_train,self.X_valid, self.y_valid,self.X_test,self.y_test,\
                    self.quantile_noise,self.Y_trans_method,self.accu_scale,self.Y_mu_0, self.Y_std_0,self.zero_feats], fp)
            gc.collect()
        std = np.std(self.X_train, axis=0)
        if False:
            plt.hist(self.y_train);         plt.show()
            plt.hist(self.y_valid);         plt.show()
            plt.hist(self.y_test);          plt.show()
        return

    def __init__(self, dataset, random_state, data_path='./data', output_distribution='normal', **kwargs):        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)
        self.random_state = random_state

        if dataset in DATASETS:
            data_dict = DATASETS[dataset](os.path.join(data_path, dataset), **kwargs)
        else:
            assert all(key in kwargs for key in ('X_train', 'y_train', 'X_valid', 'y_valid', 'X_test', 'y_test')), \
                "Unknown dataset. Provide X_train, y_train, X_valid, y_valid, X_test and y_test params"
            data_dict = kwargs

        self.data_path = data_path
        self.name = dataset
        if 'X_train' in data_dict:
            self.X_train = data_dict['X_train']
            self.y_train = data_dict['y_train']
            self.X_valid = data_dict['X_valid']
            self.y_valid = data_dict['y_valid']
            self.X_test = data_dict['X_test']
            self.y_test = data_dict['y_test']
            #数据源就被剖分。X_train中的常量特征还是没用啊
            listX = self.RemoveConstant(self.X_train,[self.X_train, self.X_valid, self.X_test])
            self.X_train, self.X_valid, self.X_test = listX[0], listX[1], listX[2]    

            self.nFeature = self.X_train.shape[1]
            
        else:
            self.X, self.Y = data_dict['X'],data_dict['Y']
            listX = self.RemoveConstant(self.X,[self.X])
            self.X = listX[0]
            self.nFeature = self.X.shape[1]
        if "num_classes" in data_dict:
            self.nClasses = data_dict["num_classes"]            

        if False:
            if all(query in data_dict.keys() for query in ('query_train', 'query_valid', 'query_test')):
                self.query_train = data_dict['query_train']
                self.query_valid = data_dict['query_valid']
                self.query_test = data_dict['query_test']
            
    def to_csv(self, path=None):
        if path == None:
            path = os.path.join(self.data_path, self.dataset)

        np.savetxt(os.path.join(path, 'X_train.csv'), self.X_train, delimiter=',')
        np.savetxt(os.path.join(path, 'X_valid.csv'), self.X_valid, delimiter=',')
        np.savetxt(os.path.join(path, 'X_test.csv'), self.X_test, delimiter=',')
        np.savetxt(os.path.join(path, 'y_train.csv'), self.y_train, delimiter=',')
        np.savetxt(os.path.join(path, 'y_valid.csv'), self.y_valid, delimiter=',')
        np.savetxt(os.path.join(path, 'y_test.csv'), self.y_test, delimiter=',')


def fetch_A9A(path, train_size=None, valid_size=None, test_size=None):
    train_path = os.path.join(path, 'a9a')
    test_path = os.path.join(path, 'a9a.t')
    if not all(os.path.exists(fname) for fname in (train_path, test_path)):
        os.makedirs(path, exist_ok=True)
        download("https://www.dropbox.com/s/9cqdx166iwonrj9/a9a?dl=1", train_path)
        download("https://www.dropbox.com/s/sa0ds895c0v4xc6/a9a.t?dl=1", test_path)

    X_train, y_train = load_svmlight_file(train_path, dtype=np.float32, n_features=123)
    X_test, y_test = load_svmlight_file(test_path, dtype=np.float32, n_features=123)
    X_train, X_test = X_train.toarray(), X_test.toarray()
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)

    if all(sizes is None for sizes in (train_size, valid_size, test_size)):
        train_idx_path = os.path.join(path, 'stratified_train_idx.txt')
        valid_idx_path = os.path.join(path, 'stratified_valid_idx.txt')
        if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/xy4wwvutwikmtha/stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/nthpxofymrais5s/stratified_test_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)
        if test_size is not None:
            warnings.warn('Test set is fixed for this dataset.', Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]    

    return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test
    )


def fetch_EPSILON(path, train_size=None, valid_size=None, test_size=None):
    train_path = os.path.join(path, 'epsilon_normalized')
    test_path = os.path.join(path, 'epsilon_normalized.t')
    if not all(os.path.exists(fname) for fname in (train_path, test_path)):
        os.makedirs(path, exist_ok=True)
        train_archive_path = os.path.join(path, 'epsilon_normalized.bz2')
        test_archive_path = os.path.join(path, 'epsilon_normalized.t.bz2')
        if not all(os.path.exists(fname) for fname in (train_archive_path, test_archive_path)):
            download("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2", train_archive_path)
            download("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2", test_archive_path)
        print("unpacking dataset")
        for file_name, archive_name in zip((train_path, test_path), (train_archive_path, test_archive_path)):
            zipfile = bz2.BZ2File(archive_name)
            with open(file_name, 'wb') as f:
                f.write(zipfile.read())

    print("reading dataset (it may take a long time)")
    X_train, y_train = load_svmlight_file(train_path, dtype=np.float32, n_features=2000)
    X_test, y_test = load_svmlight_file(test_path, dtype=np.float32, n_features=2000)
    X_train, X_test = X_train.toarray(), X_test.toarray()
    y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    if all(sizes is None for sizes in (train_size, valid_size, test_size)):
        train_idx_path = os.path.join(path, 'stratified_train_idx.txt')
        valid_idx_path = os.path.join(path, 'stratified_valid_idx.txt')
        if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/wxgm94gvm6d3xn5/stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/fm4llo5uucdglti/stratified_valid_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)
        if test_size is not None:
            warnings.warn('Test set is fixed for this dataset.', Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]

    return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test
    )


def fetch_PROTEIN(path, train_size=None, valid_size=None, test_size=None):
    """
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#protein
    """
    train_path = os.path.join(path, 'protein')
    test_path = os.path.join(path, 'protein.t')
    if not all(os.path.exists(fname) for fname in (train_path, test_path)):
        os.makedirs(path, exist_ok=True)
        download("https://www.dropbox.com/s/pflp4vftdj3qzbj/protein.tr?dl=1", train_path)
        download("https://www.dropbox.com/s/z7i5n0xdcw57weh/protein.t?dl=1", test_path)
    for fname in (train_path, test_path):
        raw = open(fname).read().replace(' .', '0.')
        with open(fname, 'w') as f:
            f.write(raw)

    X_train, y_train = load_svmlight_file(train_path, dtype=np.float32, n_features=357)
    X_test, y_test = load_svmlight_file(test_path, dtype=np.float32, n_features=357)
    X_train, X_test = X_train.toarray(), X_test.toarray()
    y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)

    if all(sizes is None for sizes in (train_size, valid_size, test_size)):
        train_idx_path = os.path.join(path, 'stratified_train_idx.txt')
        valid_idx_path = os.path.join(path, 'stratified_valid_idx.txt')
        if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/wq2v9hl1wxfufs3/small_stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/7o9el8pp1bvyy22/small_stratified_valid_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)
        if test_size is not None:
            warnings.warn('Test set is fixed for this dataset.', Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]

    return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test
    )


def fetch_YEAR(path, train_size=None, valid_size=None, test_size=51630):
    data_path = os.path.join(path, 'data.csv')
    if not os.path.exists(data_path):
        os.makedirs(path, exist_ok=True)
        download('https://www.dropbox.com/s/l09pug0ywaqsy0e/YearPredictionMSD.txt?dl=1', data_path)
    n_features = 91
    types = {i: (np.float32 if i != 0 else np.int) for i in range(n_features)}
    data = pd.read_csv(data_path, header=None, dtype=types)
    if True:
        data_dict={'X':data.iloc[:, 1:].values, 'Y':data.iloc[:, 0].values}
        return data_dict
    else:
        data_train, data_test = data.iloc[:-test_size], data.iloc[-test_size:]

        X_train, y_train = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values
        X_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values

        if all(sizes is None for sizes in (train_size, valid_size)):
            train_idx_path = os.path.join(path, 'stratified_train_idx.txt')
            valid_idx_path = os.path.join(path, 'stratified_valid_idx.txt')
            if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
                download("https://www.dropbox.com/s/00u6cnj9mthvzj1/stratified_train_idx.txt?dl=1", train_idx_path)
                download("https://www.dropbox.com/s/420uhjvjab1bt7k/stratified_valid_idx.txt?dl=1", valid_idx_path)
            train_idx = pd.read_csv(train_idx_path, header=None)[0].values
            valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
        else:
            assert train_size, "please provide either train_size or none of sizes"
            if valid_size is None:
                valid_size = len(X_train) - train_size
                assert valid_size > 0
            if train_size + valid_size > len(X_train):
                warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                    train_size + valid_size, len(X_train)), Warning)

            shuffled_indices = np.random.permutation(np.arange(len(X_train)))
            train_idx = shuffled_indices[:train_size]
            valid_idx = shuffled_indices[train_size: train_size + valid_size]
        print(f"fetch_YEAR\ttrain={X_train[train_idx].shape} valid={X_train[valid_idx].shape} test={X_test.shape}")
        return dict(
            X_train=X_train[train_idx], y_train=y_train[train_idx],
            X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
            X_test=X_test, y_test=y_test,
        )


def fetch_HIGGS(path, train_size=None, valid_size=None, test_size=5 * 10 ** 5):
    pkl_path = f'{path}/HIGGS__set_1_.pickle'
    if os.path.isfile(pkl_path):
        print("====== fetch_HIGGS@{} ......".format(pkl_path))
        with open(pkl_path, "rb") as fp:
            data_dict = pickle.load(fp)
        #print(f"====== fetch_HIGGS:\tX_={data_dict['X'].shape}\tY={data_dict['Y'].shape}")
    else:
        data_path = os.path.join(path, 'higgs.csv')

        if not os.path.exists(data_path):
            os.makedirs(path, exist_ok=True)
            archive_path = os.path.join(path, 'HIGGS.csv.gz')
            download('https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz', archive_path)
            with gzip.open(archive_path, 'rb') as f_in:
                with open(data_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        n_features = 29
        types = {i: (np.float32 if i != 0 else np.int) for i in range(n_features)}
        data = pd.read_csv(data_path, header=None, dtype=types)
        num_features = data.shape[1]-1
        Y=data.iloc[:, 0].values
        num_classes = len(set(Y))
        if False:            
            data_dict={'X':data.iloc[:, 1:].values.astype(np.float32), 'Y':Y,
                'num_features':num_features,                'num_classes':num_classes
                }             
        else:       #为了和NODE对比
            data_train, data_test = data.iloc[:-test_size], data.iloc[-test_size:]

            X_train, y_train = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values
            X_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values

            if all(sizes is None for sizes in (train_size, valid_size)):
                train_idx_path = os.path.join(path, 'stratified_train_idx.txt')
                valid_idx_path = os.path.join(path, 'stratified_valid_idx.txt')
                if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
                    download("https://www.dropbox.com/s/i2uekmwqnp9r4ix/stratified_train_idx.txt?dl=1", train_idx_path)
                    download("https://www.dropbox.com/s/wkbk74orytmb2su/stratified_valid_idx.txt?dl=1", valid_idx_path)
                train_idx = pd.read_csv(train_idx_path, header=None)[0].values
                valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
            else:
                assert train_size, "please provide either train_size or none of sizes"
                if valid_size is None:
                    valid_size = len(X_train) - train_size
                    assert valid_size > 0
                if train_size + valid_size > len(X_train):
                    warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                        train_size + valid_size, len(X_train)), Warning)

                shuffled_indices = np.random.permutation(np.arange(len(X_train)))
                train_idx = shuffled_indices[:train_size]
                valid_idx = shuffled_indices[train_size: train_size + valid_size]

            data_dict = dict(
                X_train=X_train[train_idx], y_train=y_train[train_idx],
                X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
                X_test=X_test, y_test=y_test,
                num_features=num_features,                num_classes=num_classes
            )
        with open(pkl_path, "wb") as fp:
            pickle.dump(data_dict, fp)
    #print(f"====== fetch_HIGGS:\tX={data_dict['X'].shape}")
    print(f"====== fetch_HIGGS:\tX_train={data_dict['X_train'].shape}\tX_valid={data_dict['X_valid'].shape}"\
        f"\tX_test={data_dict['X_test'].shape}")
    return data_dict      


def fetch_MICROSOFT(path):
    pkl_path = f'{path}/MICROSOFT_set_1_.pickle'
    if os.path.isfile(pkl_path):
        print("====== fetch_MICROSOFT@{} ......".format(pkl_path))
        with open(pkl_path, "rb") as fp:
            data_dict = pickle.load(fp)
        #X_train=(580539, 0)	X_valid=(142873, 0)	X_test=(241521, 0)
        print(f"====== fetch_MICROSOFT:\tX_train={data_dict['X_train'].shape}\tX_valid={data_dict['X_valid'].shape}\tX_test={data_dict['X_test'].shape}")
    else:
        train_path = os.path.join(path, 'msrank_train.tsv')
        test_path = os.path.join(path, 'msrank_test.tsv')
        if not all(os.path.exists(fname) for fname in (train_path, test_path)):
            os.makedirs(path, exist_ok=True)
            download("https://www.dropbox.com/s/izpty5feug57kqn/msrank_train.tsv?dl=1", train_path)
            download("https://www.dropbox.com/s/tlsmm9a6krv0215/msrank_test.tsv?dl=1", test_path)

        for fname in (train_path, test_path):
            raw = open(fname).read().replace('\\t', '\t')
            with open(fname, 'w') as f:
                f.write(raw)

        data_train = pd.read_csv(train_path, header=None, skiprows=1, sep='\t')
        data_test = pd.read_csv(test_path, header=None, skiprows=1, sep='\t')

        train_idx_path = os.path.join(path, 'train_idx.txt')
        valid_idx_path = os.path.join(path, 'valid_idx.txt')
        if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/pba6dyibyogep46/train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/yednqu9edgdd2l1/valid_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values

        X_train, y_train, query_train = data_train.iloc[train_idx, 2:].values, data_train.iloc[train_idx, 0].values, data_train.iloc[train_idx, 1].values
        X_valid, y_valid, query_valid = data_train.iloc[valid_idx, 2:].values, data_train.iloc[valid_idx, 0].values, data_train.iloc[valid_idx, 1].values
        X_test, y_test, query_test = data_test.iloc[:, 2:].values, data_test.iloc[:, 0].values, data_test.iloc[:, 1].values

        data_dict = dict(
            X_train=X_train.astype(np.float32), y_train=y_train.astype(np.int64), query_train=query_train,
            X_valid=X_valid.astype(np.float32), y_valid=y_valid.astype(np.int64), query_valid=query_valid,
            X_test=X_test.astype(np.float32), y_test=y_test.astype(np.int64), query_test=query_test,
        )
        print(f"====== fetch_MICROSOFT:\tX_train={X_train.shape}\tX_valid={X_valid.shape}\tX_test={X_test.shape}")
        with open(pkl_path, "wb") as fp:
            pickle.dump(data_dict, fp)
    return data_dict


def fetch_YAHOO(path):
    pkl_path = f'{path}/yahoo_rank_set_1_.pickle'
    if os.path.isfile(pkl_path):
        print("====== fetch_YAHOO@{} ......".format(pkl_path))
        with open(pkl_path, "rb") as fp:
            data_dict = pickle.load(fp)
    else:
        train_path = os.path.join(path, 'yahoo_train.tsv')
        valid_path = os.path.join(path, 'yahoo_valid.tsv')
        test_path = os.path.join(path, 'yahoo_test.tsv')
        if not all(os.path.exists(fname) for fname in (train_path, valid_path, test_path)):
            os.makedirs(path, exist_ok=True)
            train_archive_path = os.path.join(path, 'yahoo_train.tsv.gz')
            valid_archive_path = os.path.join(path, 'yahoo_valid.tsv.gz')
            test_archive_path = os.path.join(path, 'yahoo_test.tsv.gz')
            if not all(os.path.exists(fname) for fname in (train_archive_path, valid_archive_path, test_archive_path)):
                download("https://www.dropbox.com/s/7rq3ki5vtxm6gzx/yahoo_set_1_train.gz?dl=1", train_archive_path)
                download("https://www.dropbox.com/s/3ai8rxm1v0l5sd1/yahoo_set_1_validation.gz?dl=1", valid_archive_path)
                download("https://www.dropbox.com/s/3d7tdfb1an0b6i4/yahoo_set_1_test.gz?dl=1", test_archive_path)

            for file_name, archive_name in zip((train_path, valid_path, test_path), (train_archive_path, valid_archive_path, test_archive_path)):
                with gzip.open(archive_name, 'rb') as f_in:
                    with open(file_name, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

            for fname in (train_path, valid_path, test_path):
                raw = open(fname).read().replace('\\t', '\t')
                with open(fname, 'w') as f:
                    f.write(raw)

        data_train = pd.read_csv(train_path, header=None, skiprows=1, sep='\t')
        data_valid = pd.read_csv(valid_path, header=None, skiprows=1, sep='\t')
        data_test = pd.read_csv(test_path, header=None, skiprows=1, sep='\t')

        X_train, y_train, query_train = data_train.iloc[:, 2:].values, data_train.iloc[:, 0].values, data_train.iloc[:, 1].values
        X_valid, y_valid, query_valid = data_valid.iloc[:, 2:].values, data_valid.iloc[:, 0].values, data_valid.iloc[:, 1].values
        X_test, y_test, query_test = data_test.iloc[:, 2:].values, data_test.iloc[:, 0].values, data_test.iloc[:, 1].values

        data_dict = dict(
            X_train=X_train.astype(np.float32), y_train=y_train, query_train=query_train,
            X_valid=X_valid.astype(np.float32), y_valid=y_valid, query_valid=query_valid,
            X_test=X_test.astype(np.float32), y_test=y_test, query_test=query_test,
        )
        #====== fetch_YAHOO:	X_train=(473134, 699)	X_valid=(71083, 699)	X_test=(165660, 699)
        print(f"====== fetch_YAHOO:\tX_train={X_train.shape}\tX_valid={X_valid.shape}\tX_test={X_test.shape}")
        with open(pkl_path, "wb") as fp:
            pickle.dump(data_dict,fp)
    return data_dict

def fetch_CLICK(path, valid_size=100_000, validation_seed=None):
    pkl_path = f'{path}/click_set_1_.pickle'
    if os.path.isfile(pkl_path):
        print("====== fetch_CLICK@{} ......".format(pkl_path))
        with open(pkl_path, "rb") as fp:
            data_dict = pickle.load(fp)
        print(f"{data_dict['X_train'][0:5,:]}")
        print(f"{data_dict['X_test'][0:5,:]}")
        print(f"{data_dict['X_valid'][0:5,:]}")
    else:
        # based on: https://www.kaggle.com/slamnz/primer-airlines-delay
        csv_path = os.path.join(path, 'click.csv')
        if not os.path.exists(csv_path):
            os.makedirs(path, exist_ok=True)
            download('https://www.dropbox.com/s/w43ylgrl331svqc/click.csv?dl=1', csv_path)

        data = pd.read_csv(csv_path, index_col=0)
        X, y = data.drop(columns=['target']), data['target']
        X_train, X_test = X[:-100_000].copy(), X[-100_000:].copy()
        y_train, y_test = y[:-100_000].copy(), y[-100_000:].copy()

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')

        cat_features = ['url_hash', 'ad_id', 'advertiser_id', 'query_id',
                        'keyword_id', 'title_id', 'description_id', 'user_id']

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=valid_size, random_state=validation_seed)

        num_features = X_train.shape[1]
        num_classes = len(set(y_train))
        cat_encoder = LeaveOneOutEncoder()
        cat_encoder.fit(X_train[cat_features], y_train)
        X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
        X_val[cat_features] = cat_encoder.transform(X_val[cat_features])
        X_test[cat_features] = cat_encoder.transform(X_test[cat_features])
        data_dict = dict(
            X_train=X_train.values.astype('float32'), y_train=y_train,
            X_valid=X_val.values.astype('float32'), y_valid=y_val,
            X_test=X_test.values.astype('float32'), y_test=y_test,
            num_features = num_features,
            num_classes = num_classes
        )
        #print(f"====== fetch_CLICK:\tX_train={X_train.shape}\tX_valid={X_valid.shape}\tX_test={X_test.shape}")
        with open(pkl_path, "wb") as fp:
            pickle.dump(data_dict,fp)
    print(f"====== fetch_CLICK:\tX_train={data_dict['X_train'].shape}\tX_valid={data_dict['X_valid'].shape}"\
        f"\tX_test={data_dict['X_test'].shape}")
    return data_dict        


DATASETS = {
    'A9A': fetch_A9A,
    'EPSILON': fetch_EPSILON,
    'PROTEIN': fetch_PROTEIN,
    'YEAR': fetch_YEAR,
    'HIGGS': fetch_HIGGS,
    'MICROSOFT': fetch_MICROSOFT,
    'YAHOO': fetch_YAHOO,
    'CLICK': fetch_CLICK,
    #'Allstate': fetch_Allstate,
}


if __name__ == "__main__":
    data = TabularDataset("MICROSOFT", data_path="F:\Datasets", random_state=1337, quantile_transform=True,quantile_noise=1e-3)
    #data = TabularDataset("HIGGS", data_path="F:\Datasets", random_state=1337, quantile_transform=True,quantile_noise=1e-3)