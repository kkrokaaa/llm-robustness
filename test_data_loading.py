import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse
import time
import pprint

import lib.perturbations as perturbations
import lib.defenses as defenses
import lib.attacks as attacks
import lib.language_models as language_models
import lib.model_configs as model_configs
import lib.data_loading as data_loading                                 #
from assessment import Evaluator

from datasets import load_dataset, Dataset                              #
from torch.utils.data import DataLoader                                 #

def main(args):

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Instantiate the targeted LLM
    config = model_configs.MODELS[args.target_model]
    '''
    target_model = language_models.LLM(
        model_path=config['model_path'],
        tokenizer_path=config['tokenizer_path'],
        conv_template_name=config['conversation_template'],
        device='cuda:0'
    )
    '''
    target_model = language_models.BatchLLM(
        model_path=config['model_path'],
        tokenizer_path=config['tokenizer_path'],
        conv_template_name=config['conversation_template'],
        device='cuda:0'
    )
    
    # Create SmoothLLM instance
    '''
    defense = defenses.SmoothLLM(
        target_model=target_model,
        pert_type=args.smoothllm_pert_type,
        pert_pct=args.smoothllm_pert_pct,
        num_copies=args.smoothllm_num_copies
    )
    '''
    defense = defenses.BatchSmoothLLM(
        target_model=target_model,
        pert_type=args.smoothllm_pert_type,
        pert_pct=args.smoothllm_pert_pct,
        num_copies=args.smoothllm_num_copies
    )
           
    '''
    # replacing <GCG> when using general dataset;   <GCG> is used for creating <Prompt>
    data1 = load_dataset('truthful_qa', 'generation')
    print('data : ', data1)    
    prompt_generator1 = data_loading.PromptGeneratorForTruthfulQA(target_model = target_model) 

    jailbroken_results = []
    prompt_list = []                                                            #
    output_list = []                                                            #
    for i, data in tqdm(enumerate(data1['validation'])):                        ###
        prompt = prompt_generator1.generate_prompt(data)
        output = defense(prompt)
        jb = defense.is_jailbroken(output)
        jailbroken_results.append(jb)
        prompt_list.append(prompt.full_prompt)                          #
        output_list.append(output)
        if i>10:
            break
    '''
    
    data1 = load_dataset('truthful_qa', 'generation')
    prompt_generator2 = data_loading.BatchPromptGenerator(target_model)
    prompt_generator2.make_batches(data1['validation']['question'], batch_size = 8)
    prompt_list = prompt_generator2.batches                                     # list of lists of <Prompt>s
    
    jailbroken_results = []                                                    
    output_list = []                                                            #
    full_prompt_list = []                                                       #
    for i, batch_prompt in tqdm(enumerate(prompt_list)):                        # batch_prompt : list of <Prompt>s             #
        batch_output = defense(batch_prompt)
        for j, output in enumerate(batch_output):
            jb = defense.is_jailbroken(output)                                  # TODO : different criterion needed
            jailbroken_results.append(jb)
            output_list.append(output)
            full_prompt_list.append(batch_prompt[j].full_prompt)                # TODO : not exactly corresponding prompt if the model raises error (not if set as bfloat16)
        # 230 sec (batch size 8 * smoothllm num copies 10)
        break
        

    
    '''
    #print(f'We made {num_errors} errors')                  #
    print('-----------------------------------')
    print('prompt list : ', full_prompt_list)
    print('-----------------------------------')
    print('jailbroken results : ', jailbroken_results)
    print('-----------------------------------')
    print('output list : ', output_list)
    print('-----------------------------------')
        
        
        
        
    import pprint
    print('output detail')
    for i in range(3):
        print(i)
        pprint.pprint(full_prompt_list[i])
        print(jailbroken_results[i])
        pprint.pprint(output_list[i])
        print('-----------------------------------')
    '''

    # print(f'We made {num_errors} errors')

    # Save results to a pandas DataFrame
    summary_df = pd.DataFrame.from_dict({
        'Number of smoothing copies': [args.smoothllm_num_copies],
        'Perturbation type': [args.smoothllm_pert_type],
        'Perturbation percentage': [args.smoothllm_pert_pct],
        'JB percentage': [np.mean(jailbroken_results) * 100],
        'Trial index': [args.trial],
        'Prompts' : [full_prompt_list],                                         #
        'Jailbroken' : [jailbroken_results],                                    #
        'Defense Output' : [output_list]                                        #
    })                                                                          # pd df : rows should be of same length 
    summary_df.to_pickle(os.path.join(
        args.results_dir, 'summary.pd'
    ))
    #print(summary_df)
    
    evaluator = Evaluator(args.metric)
    print(evaluator(summary_df['Defense Output'][0], summary_df['Prompts'][0]))
    
    

if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results'
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=0
    )

    # Targeted LLM
    parser.add_argument(
        '--target_model',
        type=str,
        default='vicuna',
        choices=['vicuna', 'llama2']
    )

    # Attacking LLM
    parser.add_argument(
        '--attack',
        type=str,
        default='GCG',
        choices=['GCG', 'PAIR']
    )
    parser.add_argument(
        '--attack_logfile',
        type=str,
        default='data/GCG/vicuna_behaviors.json'
    )

    # SmoothLLM
    parser.add_argument(
        '--smoothllm_num_copies',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--smoothllm_pert_pct',
        type=int,
        default=10
    )
    parser.add_argument(
        '--smoothllm_pert_type',
        type=str,
        default='RandomSwapPerturbation',
        choices=[
            'RandomSwapPerturbation',
            'RandomPatchPerturbation',
            'RandomInsertPerturbation'
        ]
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
ex)
python test_data_loading.py \
    --results_dir ./test \
    --target_model llama2 \
    --attack GCG \
    --attack_logfile data/GCG/llama2_behaviors.json \
    --smoothllm_pert_type RandomSwapPerturbation \
    --smoothllm_pert_pct 1 \
    --smoothllm_num_copies 3
    --metric rouge
'''