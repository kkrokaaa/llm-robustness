import os
import sys
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
import yaml

import lib.perturbations as perturbations
import lib.defenses as defenses
import lib.attacks as attacks
import lib.language_models as language_models
import lib.model_configs as model_configs
import lib.prompt_object_generator as prompt_object_generator   
import lib.data_loader as data_loader
import  lib.assessment as assessment

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

@hydra.main(config_path='configs', config_name='config', version_base='1.3')
def main(cfg:DictConfig):
    total_tik=time.time()
    os.makedirs( os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'summary'), exist_ok=True)
    # Instantiate the targeted LLM
    print('load model')
    target_model = language_models.LLMForwardLight(model_path = cfg.model.model_path, 
                                              tokenizer_path = cfg.model.tokenizer_path,
                                              access_token=cfg.access_token.access_token)
    #target_model = language_models.LLMForwardMultiprocessing(model_path = cfg.model.model_path, 
    #                                                         tokenizer_path = cfg.model.tokenizer_path,
    #                                                         access_token=cfg.access_token.access_token)
    print(f'model loaded, size : {sys.getsizeof(target_model)}')
    
    # Create SmoothLLM instance
    defense = defenses.SmoothLLMHydraForwardLight(target_model=target_model,
                                      perturbation = instantiate(cfg.perturbation.perturbation),
                                      num_copies = cfg.smoothllm_num_copies,
                                      noise_level = cfg.noise_level)
    
    prompt_generator = prompt_object_generator.MMLUBatchPromptGeneratorDatasetwise(target_model)
    evaluator = assessment.MatchEvaluator()
    
    completed_tasks = []
    skipped_tasks = []
    print('begin main loop')
    
    #for dataset_name, data_set in tqdm(whole_data.items()):
    for dataset_name, data_set in data_loader.load_data_datasetwise(tokenizer = cfg.data_format.tokenizer,
                                 data_source = cfg.data_format.data_source,
                                 todo_task_list = cfg.todo_task_list.task_list,
                                 task_list = cfg.data_format.task_list,
                                 save_dir = cfg.data_format.save_dir,
                                 cache = cfg.data_format.cache,
                                 num_example = cfg.data_format.num_example,
                                 merge_split = cfg.data_format.merge_split,
                                 conv_generation = cfg.data_format.conv_generation,
                                 cache_dir = cfg.data_format.cache_dir):
        if dataset_name in cfg.finished_task_list.finished:
            print(f'skipping {dataset_name} (already completed)')
            completed_tasks.append(dataset_name)
            continue
        
        tik=time.time()
        prompt_and_answer = prompt_generator.make_batches(data_set, batch_size = 8)
        print(f'\n{dataset_name} prompt generated, prompt size : {sys.getsizeof(prompt_and_answer)}, time taken : {time.time()-tik}')
    
        result_dict = {'full_prompt' : [], 'answer' : [], 'output' : [], 'correct' : None, 'total' : None}
        
        tik=time.time()
        forward_batch_size = 1 if (dataset_name in ['mmlu__high_school_european_history__test', 'mmlu__high_school_world_history__test', 'mmlu__high_school_us_history__validation', 'mmlu__professional_law__test', 'mmlu__high_school_world_history__validation'] or cfg.smoothllm_num_copies==1) else cfg.forward_batch_size
        try:
            for batch_prompt, batch_answer in tqdm(zip(prompt_and_answer['prompt_batches'], prompt_and_answer['answer_batches']), total=len(prompt_and_answer['prompt_batches'])):
                batch_output = defense(batch_prompt, forward_batch_size = forward_batch_size, answer_choice_list=['A', 'B', 'C', 'D'])
                
                for j, output in enumerate(batch_output):
                    #result_dict['full_prompt'].append(batch_prompt[j].full_prompt)
                    result_dict['answer'].append(batch_answer[j])
                    result_dict['output'].append(output)
                    
            correct, total = evaluator(result_dict['output'], result_dict['answer'])
            result_dict['correct'] = correct
            result_dict['total'] = total
            print(f'{dataset_name} done, time taken : {time.time()-tik}')
            
            # Save results to a pandas DataFrame
            summary_df = pd.DataFrame.from_dict({
                'DatasetName' : [dataset_name], 
                'Prompts' : [result_dict['full_prompt']],
                'Answers' : [result_dict['answer']],
                'Outputs' : [result_dict['output']],
                'Correct' : [result_dict['correct']],
                'Total' : [result_dict['total']],
                'TimeTaken' : [time.time()-tik]
            })                                                                          # pd df : rows should be of same length 
            summary_df.to_pickle(os.path.join( hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'summary/'+dataset_name+'.pd' ))
            completed_tasks.append(dataset_name)
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print(f'error raised in {dataset_name}, skipping the dataset')
            print(f'error message :: {e}')
            skipped_tasks.append(dataset_name)
            continue
    
    os.makedirs( os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'todo_task_list'), exist_ok=True)
    with open(os.path.join( hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'todo_task_list/mmlu.yaml'), 'w') as f:
        yaml.dump({'task_list' : skipped_tasks}, f)
    
    os.makedirs( os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'finished_task_list'), exist_ok=True)
    with open(os.path.join( hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'finished_task_list/mmlu.yaml'), 'w') as f:
        yaml.dump({'finished' : completed_tasks}, f)
    
    print(f'total time taken : {time.time()-total_tik}')
    with open(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'time.yaml'), 'w') as f:
        yaml.dump({'total_time_taken' : time.time()-total_tik}, f)
        
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
    
    
    
'''
ex)
utilizing multiple gpus:
    export CUDA_VISIBLE_DEVICES=0,1     or      CUDA_VISIBLE_DEVICES=0,1 ./cuda_executable
python main_combined.py
    result is saved in outputs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main_combined.py
python test_hydra.py -m model=llama2,vicuna smoothllm_perturbation_rate=1,5 smoothllm_num_copies=3 data=(data1),(data2)
    results are saved in multirun
    
'''