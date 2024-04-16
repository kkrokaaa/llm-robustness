import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse

import lib.perturbations as perturbations
import lib.defenses as defenses
import lib.attacks as attacks
import lib.language_models as language_models
import lib.model_configs as model_configs

def main(args):

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Instantiate the targeted LLM
    config = model_configs.MODELS[args.target_model]
    target_model = language_models.LLM(
        model_path=config['model_path'],
        tokenizer_path=config['tokenizer_path'],
        conv_template_name=config['conversation_template'],
        device='cuda:0'
    )

    # Create SmoothLLM instance
    defense = defenses.SmoothLLM(
        target_model=target_model,
        pert_type=args.smoothllm_pert_type,
        pert_pct=args.smoothllm_pert_pct,
        num_copies=args.smoothllm_num_copies
    )

    # Create attack instance, used to create prompts
    attack = vars(attacks)[args.attack](
        logfile=args.attack_logfile,
        target_model=target_model
    )

    jailbroken_results = []
    prompt_list = []
    output_list = []
    pbar = tqdm(attack.prompts)
    for i,prompt in enumerate(pbar):    
        output = defense(prompt)
        jb = defense.is_jailbroken(output)
        jailbroken_results.append(jb)
        # show average jb percentage
        pbar.set_postfix({
            'JB percentage': np.mean(jailbroken_results) * 100
        })
        prompt_list.append(prompt.full_prompt)
        output_list.append(output)
        if i>=3:
            break
        # 19 sec (smoothllm num copies 5)

    # print(f'We made {num_errors} errors')

    # Save results to a pandas DataFrame
    summary_df = pd.DataFrame.from_dict({
        'Number of smoothing copies': [args.smoothllm_num_copies],
        'Perturbation type': [args.smoothllm_pert_type],
        'Perturbation percentage': [args.smoothllm_pert_pct],
        'JB percentage': [np.mean(jailbroken_results) * 100],
        'Trial index': [args.trial],
        'Prompts' : [prompt_list],
        'Jailbroken' : [jailbroken_results],
        'Defense Output' : [output_list] 
    })
    summary_df.to_pickle(os.path.join(
        args.results_dir, 'summary.pd'
    ))
    print(summary_df)


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

    args = parser.parse_args()
    main(args)
'''
need to install modified transformers before run:
python -m pip install git+https://github.com/kkrokii/transformers_noisy.git

ex)
python main.py \
    --results_dir ./test \
    --target_model llama2 \
    --attack GCG \
    --attack_logfile data/GCG/llama2_behaviors.json \
    --smoothllm_pert_type RandomSwapPerturbation \
    --smoothllm_pert_pct 1 \
    --smoothllm_num_copies 3
'''