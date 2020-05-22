import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from collections import namedtuple
import catboost as cat
import lightgbm as lgb
import numpy as np
import xgboost as xgb
mort_root = f'../LiteMORT/'
if mort_root is not None:    #liteMORT    
    sys.path.insert(1, f'{mort_root}/python-package/')
    import litemort
    from litemort import *
QF_root = f'../QuantumForest/'
if QF_root is not None:    #liteMORT    
    sys.path.insert(2, f'{QF_root}/python-package/')
    import quantum_forest
    from quantum_forest import *
from sklearn.metrics import mean_squared_error, accuracy_score

RANDOM_SEED = 0


class TimeAnnotatedFile:
    def __init__(self, file_descriptor):
        self.file_descriptor = file_descriptor

    def write(self, message):
        if message == '\n':
            self.file_descriptor.write('\n')
            return

        cur_time = datetime.now()
        #new_message = message
        new_message = "Time: [%d.%06d]\t%s" % (cur_time.second, cur_time.microsecond, message)
        self.file_descriptor.write(new_message)

    def flush(self):
        self.file_descriptor.flush()

    def close(self):
        self.file_descriptor.close()


class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.stdout = sys.stdout

    def __enter__(self):
        self.file = TimeAnnotatedFile(open(self.filename, 'w'))
        sys.stdout = self.file

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is not None:
            print(str(exception_value) + '\n' + str(traceback))

        sys.stdout = self.stdout
        self.file.close()


def eval_metric(data, prediction):
    if data.metric == "RMSE":
        return np.sqrt(mean_squared_error(data.y_test, prediction))
    elif data.metric == "Accuracy":
        if data.task == "binclass":
            prediction = prediction > 0.5
        elif data.task == "multiclass":
            if prediction.ndim > 1:
                prediction = np.argmax(prediction, axis=1)
        return accuracy_score(data.y_test, prediction)
    else:
        raise ValueError("Unknown metric: " + data.metric)


class Learner:
    def __init__(self):
        self.default_params = {}

    def _fit(self, tunable_params):
        params = deepcopy(self.default_params)
        params.update(tunable_params)

        print('Parameters:\n{}' + str(params))
        return params

    def eval(self, data, num_iterations, step=10):
        scores = []

        for n_tree in range(num_iterations, step=step):
            prediction = self.predict(n_tree)
            score = eval_metric(data, prediction)
            scores.append(score)

        return scores

    def predict(self, n_tree):
        raise Exception('Not implemented')

    def set_train_dir(self, params, path):
        pass

    def run(self, params, log_filename):
        log_dir_name = os.path.dirname(log_filename)

        if not os.path.exists(log_dir_name):
            os.makedirs(log_dir_name)

        self.set_train_dir(params, log_filename + 'dir')

        with Logger(log_filename):
            start = time.time()
            self._fit(params)
            elapsed = time.time() - start
            print('Elapsed: ' + str(elapsed))

        return elapsed


class XGBoostLearner(Learner):
    def __init__(self, data, task, metric, use_gpu):
        Learner.__init__(self)
        params = {
            'n_gpus': 1,
            'silent': 0,
            'seed': RANDOM_SEED
        }

        if use_gpu:
            params['tree_method'] = 'gpu_hist'
        else:
            params['tree_method'] = 'hist'

        if task == "regression":
            params["objective"] = "reg:linear"
            if use_gpu:
                params["objective"] = "gpu:" + params["objective"]
        elif task == "multiclass":
            params["objective"] = "multi:softmax"
            params["num_class"] = int(np.max(data.y_test)) + 1
        elif task == "binclass":
            params["objective"] = "binary:logistic"
            if use_gpu:
                params["objective"] = "gpu:" + params["objective"]
        else:
            raise ValueError("Unknown task: " + task)

        if metric == 'Accuracy':
            if task == 'binclass':
                params['eval_metric'] = 'error'
            elif task == 'multiclass':
                params['eval_metric'] = 'merror'

        self.train = xgb.DMatrix(data.X_train, data.y_train)
        self.test = xgb.DMatrix(data.X_test, data.y_test)

        self.default_params = params

    @staticmethod
    def name():
        return 'xgboost'

    def _fit(self, tunable_params):
        params = Learner._fit(self, tunable_params)
        self.learner = xgb.train(params, self.train, tunable_params['iterations'], evals=[(self.test, 'eval')])

    def predict(self, n_tree):
        return self.learner.predict(self.test, ntree_limit=n_tree)


class LightGBMLearner(Learner):
    def __init__(self, data, task, metric, use_gpu):
        Learner.__init__(self)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'verbose': 0,       #实际上无用，只由train-verbose_eval控制
            'random_state': RANDOM_SEED,
            'bagging_freq': 1,
        }

        if use_gpu:
            params["device"] = "gpu"

        if task == "regression":
            params["objective"] = "regression"
        elif task == "multiclass":
            params["objective"] = "multiclass"
            params["num_class"] = int(np.max(data.y_test)) + 1
        elif task == "binclass":
            params["objective"] = "binary"
        else:
            raise ValueError("Unknown task: " + task)

        if metric == 'Accuracy':
            if task == 'binclass':
                params['metric'] = 'binary_error'
            elif task == 'multiclass':
                params['metric'] = 'multi_error'
        elif metric == 'RMSE':
            params['metric'] = 'rmse'

        self.train = lgb.Dataset(data.X_train, data.y_train)
        self.test = lgb.Dataset(data.X_test, data.y_test, reference=self.train)

        self.default_params = params

    @staticmethod
    def name():
        return 'lightgbm'

    def _fit(self, tunable_params):
        params_copy = deepcopy(tunable_params)

        if 'max_depth' in params_copy:
            params_copy['num_leaves'] = 2 ** params_copy['max_depth']
            del params_copy['max_depth']

        num_iterations = params_copy['iterations']
        del params_copy['iterations']

        params = Learner._fit(self, params_copy)
        self.learner = lgb.train(
            params,
            self.train,
            num_boost_round=num_iterations,
            valid_sets=self.test,
            #verbose_eval=False,     #bool or int(default=True)
        )
        print()

    def predict(self, n_tree):
        return self.learner.predict(self.test, num_iteration=n_tree)

class LiteMORTLearner(Learner):
    def __init__(self, data, task, metric, use_gpu):
        Learner.__init__(self)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'verbose': 0,
            'random_state': RANDOM_SEED,
            'bagging_freq': 1
        }

        if use_gpu:
            params["device"] = "gpu"

        if task == "regression":
            params["objective"] = "regression"
        elif task == "multiclass":
            params["objective"] = "multiclass"
            params["num_class"] = int(np.max(data.y_test)) + 1
        elif task == "binclass":
            params["objective"] = "binary"
        else:
            raise ValueError("Unknown task: " + task)

        if metric == 'Accuracy':
            if task == 'binclass':
                params['metric'] = 'binary_error'
            elif task == 'multiclass':
                params['metric'] = 'multi_error'
        elif metric == 'RMSE':
            params['metric'] = 'rmse'
        self.X_train, self.y_train = data.X_train, data.y_train
        self.eval_set = [(data.X_test, data.y_test)]
        self.test = data.X_test

        self.default_params = params

    @staticmethod
    def name():
        return 'litemort'

    def _fit(self, tunable_params):
        params_copy = deepcopy(tunable_params)

        if 'max_depth' in params_copy:
            params_copy['num_leaves'] = 2 ** params_copy['max_depth']
            del params_copy['max_depth']

        num_iterations = params_copy['iterations']
        del params_copy['iterations']

        params = Learner._fit(self, params_copy)
        self.learner = LiteMORT(params).fit(self.X_train, self.y_train, eval_set=self.eval_set)
        
    def predict(self, n_tree):
        return self.learner.predict(self.test, num_iteration=n_tree)

class QuantumForestLearner(Learner):
    def __init__(self, data_tuple, task, metric, use_gpu):
        Learner.__init__(self)
        params = {
            'task': 'train',
            'verbose': 0,
            'random_state': RANDOM_SEED,
        }

        if use_gpu:
            params["device"] = "gpu"

        if task == "regression":
            params["objective"] = "regression"
        elif task == "multiclass":
            params["objective"] = "multiclass"
            params["num_class"] = int(np.max(data.y_test)) + 1
        elif task == "binclass":
            params["objective"] = "binary"
        else:
            raise ValueError("Unknown task: " + task)

        if metric == 'Accuracy':
            if task == 'binclass':
                params['metric'] = 'binary_error'
            elif task == 'multiclass':
                params['metric'] = 'multi_error'
        elif metric == 'RMSE':
            params['metric'] = 'rmse'
        
        data_dict = data_tuple._asdict()
        self.data = quantum_forest.TabularDataset(data_tuple.name,random_state=42,**data_dict)
        self.data.onFold(0,params)
        self.config = QForest_config(self.data,0.002,feat_info="")  
        self.config, self.visual = InitExperiment(self.config, 0)      
        self.config.in_features = self.data.X_train.shape[1]
        self.config.device = quantum_forest.OnInitInstance(seed=42)
        self.config.model = "QForest"
        self.config.tree_module = quantum_forest.DeTree   
        # self.X_train, self.y_train = self.data.X_train, self.data.y_train
        # self.eval_set = [(self.data.X_test, self.data.y_test)]
        # self.test = self.data.X_test

        self.default_params = params

    @staticmethod
    def name():
        return 'QuantumForest'

    def _fit(self, tunable_params):
        params_copy = deepcopy(tunable_params)

        if 'max_depth' in params_copy:
            params_copy['num_leaves'] = 2 ** params_copy['max_depth']
            del params_copy['max_depth']

        num_iterations = params_copy['iterations']
        del params_copy['iterations']
        self.config.no_inner_writer = True
        self.config.nMostEpochs = num_iterations
        self.config.lr_base = params_copy['learning_rate']
        # params = Learner._fit(self, params_copy)
        # params.update({"model":"QForest","response_dim":3,"feat_info":None,"in_features":in_features})        
        # self.config = Dict2Obj(params)
        flags = {"test_once":True,'report_frequency':10}
        self.learner = QuantumForest(self.config,self.data). \
            fit(self.data.X_train, self.data.y_train, [(self.data.X_valid,self.data.y_valid)])  #,flags=flags
        
    def predict(self, n_tree):
        return self.learner.predict(self.test, num_iteration=n_tree)

class CatBoostLearner(Learner):
    def __init__(self, data, task, metric, use_gpu):
        Learner.__init__(self)
        params = {
            'devices': [0],
            'logging_level': 'Info',
            'use_best_model': False,
            'bootstrap_type': 'Bernoulli',
            'random_seed': RANDOM_SEED
        }

        if use_gpu:
            params['task_type'] = 'GPU'

        if task == 'regression':
            params['loss_function'] = 'RMSE'
        elif task == 'binclass':
            params['loss_function'] = 'Logloss'
        elif task == 'multiclass':
            params['loss_function'] = 'MultiClass'

        if metric == 'Accuracy':
            params['custom_metric'] = 'Accuracy'

        self.train = cat.Pool(data.X_train, data.y_train)
        self.test = cat.Pool(data.X_test, data.y_test)

        self.default_params = params

    @staticmethod
    def name():
        return 'catboost'

    def _fit(self, tunable_params):
        params = Learner._fit(self, tunable_params)
        self.model = cat.CatBoost(params)
        self.model.fit(self.train, eval_set=self.test, verbose_eval=True)

    def set_train_dir(self, params, path):
        if not os.path.exists(path):
            os.makedirs(path)
        params["train_dir"] = path

    def predict(self, n_tree):
        if self.default_params['loss_function'] == "MultiClass":
            prediction = self.model.predict_proba(self.test, ntree_end=n_tree)
        else:
            prediction = self.model.predict(self.test, ntree_end=n_tree)

        return prediction

learners = [        
        LightGBMLearner,
        #QuantumForestLearner,
        #XGBoostLearner,
        #LiteMORTLearner,
        
        #CatBoostLearner
    ]
ALGORITHMS = [method + '-' + device_type
              for device_type in ['CPU', 'GPU']
              for method in [x.name() for x in learners]]
print(f"======\tlearners={learners} \n======\tALGORITHMS={ALGORITHMS}\n======")
