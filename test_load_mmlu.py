import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse
import time
import pprint
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import get_original_cwd, instantiate

import lib.perturbations as perturbations
import lib.defenses as defenses
import lib.attacks as attacks
import lib.language_models as language_models
import lib.model_configs as model_configs
import lib.prompt_object_generator as prompt_object_generator   
import lib.data_loader as data_loader
import lib.assessment as assessment

from datasets import load_dataset, Dataset                              #
from torch.utils.data import DataLoader                                 #

@hydra.main(config_path='configs', config_name='config', version_base='1.3')
def main(cfg:DictConfig):
    # Instantiate the targeted LLM
    target_model = language_models.BatchLLM(model_path = cfg.model.model_path, 
                                            tokenizer_path = cfg.model.tokenizer_path, 
                                            conv_template_name = cfg.model.conversation_template, 
                                            device='cuda:0')
    
    # Create SmoothLLM instance
    defense = defenses.BatchSmoothLLMHydra(target_model=target_model,
                                      perturbation = instantiate(cfg.perturbation.perturbation),
                                      num_copies = cfg.smoothllm_num_copies)
    print('start loading the data')
    data = data_loader.load_data(tokenizer = cfg.data_format.tokenizer,
                                 data_source = cfg.data_format.data_source,
                                 task_list = cfg.data_format.task_list,
                                 save_dir = cfg.data_format.save_dir,
                                 cache = cfg.data_format.cache,
                                 num_example = cfg.data_format.num_example,
                                 merge_split = cfg.data_format.merge_split,
                                 conv_generation = cfg.data_format.conv_generation,
                                 cache_dir = cfg.data_format.cache_dir)         # DatasetDict
    print('data loaded')
              
    prompt_generator = prompt_object_generator.MMLUBatchPromptGenerator(target_model)
    prompt_dict = prompt_generator.make_batches(data, batch_size = 8)
    evaluator = assessment.Evaluator(cfg.metric.metric_name)
    
    result_dict = {}
    
    for i, (dataset_name, data_dict) in tqdm(enumerate(prompt_dict.items())):   # data_dict : {'prompt_batches' : ..., 'answer_batches' : ...}
        result_dict[dataset_name] = {'output' : [], 'full_prompt' : [], 'answer' : [], 'performance' : []}
        
        for batch_prompt, batch_answer in zip(data_dict['prompt_batches'], data_dict['answer_batches']):
            print('batch loop')
            batch_output = defense(batch_prompt)
            for j, output in enumerate(batch_output):
                result_dict[dataset_name]['output'].append(output)
                result_dict[dataset_name]['full_prompt'].append(batch_prompt[j].full_prompt)
                result_dict[dataset_name]['answer'].append(batch_answer[j])
            break
        result_dict[dataset_name]['performance'].append(evaluator( result_dict[dataset_name]['output'], result_dict[dataset_name]['answer'] ))
        print(f"dataset name : {dataset_name}")
        print(f"full prompt : {[result_dict[dataset_name]['full_prompt']]}")
        print(f"answer : {[result_dict[dataset_name]['answer']]}")
        print(f"output : {[result_dict[dataset_name]['output']]}")
        print(f"performance : {[result_dict[dataset_name]['performance']]}")
        break

    # Save results to a pandas DataFrame
    summary_df = pd.DataFrame.from_dict({
        'Number of smoothing copies':   [cfg.smoothllm_num_copies],
        'Perturbation type':            [cfg.perturbation],
        'Perturbation percentage':      [cfg.smoothllm_perturbation_rate],
        'Output' :                      [result_dict[dataset_name]['output']],
        'Full prompt' :                 [result_dict[dataset_name]['full_prompt']],
        'Answer' :                      [result_dict[dataset_name]['answer']],
        'Performance' :                 [result_dict[dataset_name]['performance']]
    })                                                                          # pd df : rows should be of same length 
    summary_df.to_pickle(os.path.join( hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'summary.pd' ))
    
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
    
    
    
'''
ex)
python test_load_mmlu.py
    result is saved in outputs
python test_hydra.py -m model=llama2,vicuna smoothllm_perturbation_rate=1,5 smoothllm_num_copies=3 data=(data1),(data2)
    results are saved in multirun
    
'''