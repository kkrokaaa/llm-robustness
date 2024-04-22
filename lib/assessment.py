import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse
import pandas as pd

import lib.perturbations as perturbations
import lib.defenses as defenses
import lib.attacks as attacks
import lib.language_models as language_models
import lib.model_configs as model_configs
import lib.prompt_object_generator as prompt_object_generator                                 #

from datasets import load_dataset, Dataset                              #
from torch.utils.data import DataLoader                                 #

import evaluate 



class Evaluator:
    def __init__(self, metric):
        self.metric = evaluate.load(metric)
    def __call__(self, predictions, references):
        return self.metric.compute(predictions=predictions, references=references)



class MatchEvaluator:
    def __call__(self, predictions, references):
        assert len(predictions) == len(references)
        total = len(predictions)
        count = 0
        for a,b in zip(predictions, references):
            count += (a==b)
        return count, total




def main(args):

    # Create output directories
    if args.save_dir is None:
        args.save_dir = args.results_dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    result = pd.read_pickle(args.results_dir + '/summary.pd')
    
    evaluator = Evaluator(args.metric)
    
    print(evaluator(result['Defense Output'][0], result['Prompts'][0]))
    
    

if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='rouge',
        choices=['rouge', 'bleu']
    )

    args = parser.parse_args()
    main(args)
    
'''
example:
python assessment.py --results_dir=result

python assessment.py --results_dir=result --save_dir=result
'''