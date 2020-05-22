'''
@Author: Yingshi Chen

@Date: 2020-03-10 16:11:28
@
# Description: 
'''
import argparse
from experiments import EXPERIMENTS
from learners import *
from generate_report import get_experiment_stats, print_all_in_one_table
from data_loader import get_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--datasets', default='datasets')
    parser.add_argument('--iterations', default=1000, type=int)
    parser.add_argument('--result', default='result.json')
    parser.add_argument('--table', default='common-table.txt')
    args = parser.parse_args()
    sTime = time.strftime("%m%d_%H_%M_%S", time.localtime())
    #sTime = '{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}'.format(*time.gmtime()[1:5])
    args.result = f'./result/result_{sTime}.json'

    experiments_names = [
        #'abalone',        
        "MICROSOFT",
        "YEAR", 
        "YAHOO",
        #"HIGGS",
        #
        #"CLICK",
        #'EPSILON',
        #'airline',
        #'epsilon',
        #'higgs',
        #'letters',
        #'msrank',
        #'msrank-classification',
        #'synthetic',
        #'synthetic-5k-features'
    ]

    #learners = [#XGBoostLearner,LightGBMLearner,#CatBoostLearner    ]

    iterations = args.iterations
    logs_dir = 'logs'

    params_grid = {
        'iterations': [iterations],
        'max_depth': [6],
        'learning_rate': [0.03, 0.07, 0.15]
    }

    #args.datasets = "L:/Datasets/"
    nEXP= len(experiments_names)
    for i,experiment_name in enumerate(experiments_names):
        print(f"\n********************* {experiment_name} {i+1}/{nEXP} ......",end="")
        data_tuple,desc = get_dataset(experiment_name, args.datasets)
        print(f"\r********************* {experiment_name} {i+1}/{nEXP} *********************\n{desc}")

        
        experiment = EXPERIMENTS[experiment_name]
        #experiment.run(args.use_gpu, learners, params_grid, args.datasets, args.result, logs_dir)
        experiment.run(args.use_gpu, learners, params_grid, data_tuple, args.result, logs_dir)


    stats = get_experiment_stats(args.result, args.use_gpu, niter=iterations)
    print_all_in_one_table(stats, args.use_gpu,learners, params=(6.0, 1.0), output=args.table)


if __name__ == "__main__":
    main()
