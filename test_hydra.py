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
import lib.data_loading as data_loading                                 #
from assessment import Evaluator

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
    
    
    
    evaluator = Evaluator(cfg.metric.metric_name)
    performance = evaluator(output_list, full_prompt_list)

    # Save results to a pandas DataFrame
    summary_df = pd.DataFrame.from_dict({
        'Number of smoothing copies': [cfg.smoothllm_num_copies],
        'Perturbation type': [cfg.perturbation],
        'Perturbation percentage': [cfg.smoothllm_perturbation_rate],
        'JB percentage': [np.mean(jailbroken_results) * 100],
        'Prompts' : [full_prompt_list],                                         #
        'Jailbroken' : [jailbroken_results],                                    #
        'Defense Output' : [output_list],                                       #
        'Performance' : [performance]                                           #
    })                                                                          # pd df : rows should be of same length 
    summary_df.to_pickle(os.path.join( hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'summary.pd' ))
    
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
    
    
    
'''
ex)
python test_hydra.py
    result is saved in outputs
python test_hydra.py -m model=llama2,vicuna smoothllm_perturbation_rate=1,5 smoothllm_num_copies=3
    results are saved in multirun
    
'''