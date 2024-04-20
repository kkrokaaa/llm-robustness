from __future__ import annotations

from pathlib import Path
from typing import List, Iterator, Optional
from tqdm.auto import tqdm
import re
#from loguru import logger
from pprint import pprint
import pandas as pd

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import get_original_cwd, instantiate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def try_loading_cache(tokenizer: transformers.PreTrainedTokenizer,
                    num_example: int = 5,
                    merge_split: bool = True,
                    conv_generation: bool = False,
                    cache_local: Optional[str] = None):
    if cache_local is None:
        cache_local = "./data_cache"
        print(f'cache directory not given to the argument "cache_local"; automatically set as {cache_local}')
    caching_path = str( Path(cache_local) / f"mmlu_{tokenizer.__class__.__name__}_exmp{num_example}_merge{merge_split}_conv{conv_generation}" )

    if Path(caching_path).exists():
        #logger.info(f"Loading cached dataset from {caching_path}")
        try:
            merged_datasetdict = datasets.load_from_disk(caching_path)
            return merged_datasetdict
        except:
            #logger.warning(f"Failed to load cached dataset from {caching_path}, need regeneration")
            return None
    else:
        #logger.warning(f"Cache directory {cache_local} does not exist, need regeneration")
        return None


def mmlu_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = "./data/mmlu",
    num_example: int = 5,
    cache: bool = True,
    merge_split: bool = False,
    conv_generation: bool = True,
    cache_local: Optional[str] = None
) -> datasets.DatasetDict:
    '''
    dpath :             path where mmlu is saved;       consisted of sub directories        ex) mmlu/astronomy, mmlu/abstract_algebra, ...
    num_example :       number of IC ex
    cache :             if the data was processed in the same way before, load the cache instead of processing the whole dataset all over again
                            note that the processing depends on setting (tokenizer, IC ex, ...)
    merge_split :       True : merge 'train', 'validation', 'test' into a single dataset
    conv_generation :   when constructing a prompt,     True : remove only the 1st IC ex, append the previous query as a new IC ex, and select a new query from the dataset 
                                                        False : discard all of previous IC ex & query, and newly select them from them dataset 
    '''
    step_size = 1 + num_example
    merged_datasets = {}

    if cache:
        if cache_local is None:
            cache_local = "./data_cache"
        caching_path = str( Path(cache_local) / f"mmlu_{tokenizer.__class__.__name__}_exmp{num_example}_merge{merge_split}_conv{conv_generation}" )

    data_dirs = [ str(Path(_)) for _ in Path(dpath).absolute().glob("*") if Path(_).is_dir() ]      # absolute path of `subdir` of the directory of dpath

    for f in tqdm(data_dirs, desc="Iterating over files in tar"):
        dd = datasets.load_from_disk(f)                                                             # DatasetDict{'train' : (Dataset), 'validation' : (Datset), 'test' : (Dataset)}
        if merge_split:
            ds = datasets.concatenate_datasets([dd[split] for split in dd.keys()])
            ds_key = Path(f).stem                                                                   # task name (name of the subdir folder)
            ds_and_key = [(ds, ds_key)]                                                             # [(dataset, task name)]
        else:
            ds_and_key = [(ds, Path(f).stem + "__" + split) for split, ds in dd.items()]            # [(dataset, task name__train), (dataset, task name__validation), (dataset, task name__test)]

        for ds, ds_key in ds_and_key:
            if ds_key not in merged_datasets:
                merged_datasets[ds_key] = []
            chunk_cache = []
            if not conv_generation:
                for idx, row in tqdm(enumerate(ds), desc=f"Formatting {ds_key} dataset", leave=False):
                    chunk_cache.append(row)
                    if (idx + 1) % step_size == 0:
                        prompt_head = (
                            "You would be given a multiple-choice question paried with 4 choices (A-D). "
                            "Choose one of them using letter A, B, C, or D as the correct answer to the question. "
                            "Here are some examples: "
                        )
                        examples = "".join( [ (
                                    f"\n\n{row['input']}"
                                    f"\nA: {row['A']}"
                                    f"\nB: {row['B']}"
                                    f"\nC: {row['C']}"
                                    f"\nD: {row['D']}"
                                    f"\n\nAnswer: {row['target']}"
                                ) for row in chunk_cache[:-1]
                        ] )
                        examples += "\n\nNow answer the question:\n\n"
                        question = (
                            f"{chunk_cache[-1]['input']}"
                            f"\nA: {chunk_cache[-1]['A']}"
                            f"\nB: {chunk_cache[-1]['B']}"
                            f"\nC: {chunk_cache[-1]['C']}"
                            f"\nD: {chunk_cache[-1]['D']}"
                            f"\n\nAnswer: "
                        )
                        answer = f"{chunk_cache[-1]['target']}"

                        if tokenizer is not None and tokenizer.bos_token is not None:
                            prompt_head_tokens = tokenizer.encode(prompt_head)
                            examples_tokens = tokenizer.encode(examples)
                            question_tokens = tokenizer.encode(question)
                            answer_tokens = tokenizer.encode(answer)

                            if prompt_head_tokens[-1] == tokenizer.eos_token_id:
                                prompt_head_tokens = prompt_head_tokens[:-1]
                            if examples_tokens[-1] == tokenizer.eos_token_id:
                                examples_tokens = examples_tokens[:-1]
                            if question_tokens[-1] == tokenizer.eos_token_id:
                                question_tokens = question_tokens[:-1]

                            if examples_tokens[0] == tokenizer.bos_token_id:
                                examples_tokens = examples_tokens[1:]
                            if question_tokens[0] == tokenizer.bos_token_id:
                                question_tokens = question_tokens[1:]
                            if answer_tokens[0] == tokenizer.bos_token_id:
                                answer_tokens = answer_tokens[1:]

                            merged_datasets[ds_key].append(
                                {
                                    "tokenized_prompt": prompt_head_tokens
                                    + examples_tokens
                                    + question_tokens
                                    + answer_tokens,
                                    "question_token_start_idx": len(prompt_head_tokens)
                                    + len(examples_tokens),
                                    "answer_token_start_idx": len(prompt_head_tokens)
                                    + len(examples_tokens)
                                    + len(question_tokens),
                                    "answer_str": answer,
                                }
                            )

                        else:
                            merged_datasets[ds_key].append(
                                {
                                    "prompt": prompt_head + examples + question,
                                    "answer": answer,
                                }
                            )

                        chunk_cache = []                                                            # discard all the used IC ex & query 

            else:
                for idx, row in tqdm(
                    enumerate(ds), desc=f"Formatting {ds_key} dataset", leave=False
                ):
                    chunk_cache.append(row)
                    if len(chunk_cache) == step_size:
                        prompt_head = (
                            "You would be given a multiple-choice question paried with 4 choices (A-D). "
                            "Choose one of them using letter A, B, C, or D as the correct answer to the question. "
                            "Here are some examples: "
                        )
                        examples = "".join( [
                                (
                                    f"\n\n{row['input']}"
                                    f"\nA: {row['A']}"
                                    f"\nB: {row['B']}"
                                    f"\nC: {row['C']}"
                                    f"\nD: {row['D']}"
                                    f"\n\nAnswer: {row['target']}"
                                ) for row in chunk_cache[:-1]
                        ] )
                        examples += "\n\nNow answer the question:\n\n"
                        question = (
                            f"{chunk_cache[-1]['input']}"
                            f"\nA: {chunk_cache[-1]['A']}"
                            f"\nB: {chunk_cache[-1]['B']}"
                            f"\nC: {chunk_cache[-1]['C']}"
                            f"\nD: {chunk_cache[-1]['D']}"
                            f"\n\nAnswer: "
                        )
                        answer = f"{chunk_cache[-1]['target']}"

                        if tokenizer is not None and tokenizer.bos_token is not None:
                            prompt_head_tokens = tokenizer.encode(prompt_head)
                            examples_tokens = tokenizer.encode(examples)
                            question_tokens = tokenizer.encode(question)
                            answer_tokens = tokenizer.encode(answer)

                            if prompt_head_tokens[-1] == tokenizer.eos_token_id:
                                prompt_head_tokens = prompt_head_tokens[:-1]
                            if examples_tokens[-1] == tokenizer.eos_token_id:
                                examples_tokens = examples_tokens[:-1]
                            if question_tokens[-1] == tokenizer.eos_token_id:
                                question_tokens = question_tokens[:-1]

                            if examples_tokens[0] == tokenizer.bos_token_id:
                                examples_tokens = examples_tokens[1:]
                            if question_tokens[0] == tokenizer.bos_token_id:
                                question_tokens = question_tokens[1:]
                            if answer_tokens[0] == tokenizer.bos_token_id:
                                answer_tokens = answer_tokens[1:]

                            merged_datasets[ds_key].append(
                                {
                                    "tokenized_prompt": prompt_head_tokens
                                    + examples_tokens
                                    + question_tokens
                                    + answer_tokens,
                                    "question_token_start_idx": len(prompt_head_tokens)
                                    + len(examples_tokens),
                                    "answer_token_start_idx": len(prompt_head_tokens)
                                    + len(examples_tokens)
                                    + len(question_tokens),
                                    "answer_str": answer,
                                }
                            )

                        else:
                            merged_datasets[ds_key].append(
                                {
                                    "prompt": prompt_head + examples + question,
                                    "answer": answer,
                                }
                            )

                        chunk_cache.pop(0)                                                          # discard only the oldest IC ex, use the previous query as a new IC ex

    merged_datasets = {
        ("mmlu__" + k): v for k, v in merged_datasets.items() if len(v) > 0
    }                                                                                               # 'mmlu__task name__train' : {tokenizer prompt : ..., question token start idx : ..., ...}

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict                                                                       # DatasetDict{ 'mmlu__astronomy__train' : Dataset(...), ...}

def load_data(tokenizer: transformers.PreTrainedTokenizer,
              data_source: str,
              task_list: List,
              save_dir: str,
              cache: bool = True,
              num_example: int = 6,
              merge_split: bool = True,
              conv_generation: bool = True,
              cache_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
    data_total = None
    
    if cache:
        data_total = try_loading_cache(tokenizer,
                        num_example = num_example,
                        merge_split = merge_split,
                        conv_generation = conv_generation,
                        cache_local = cache_dir)
    
    if data_total is None:
        for task in task_list:
            data = load_dataset(data_source, task)
            data.save_to_disk(save_dir + task)

        data_total = mmlu_formatter(tokenizer, 
                                    dpath = save_dir, 
                                    num_example = num_example,
                                    cache = cache,
                                    merge_split = merge_split,
                                    conv_generation = conv_generation,
                                    cache_local = cache_dir)
    
    return data_total