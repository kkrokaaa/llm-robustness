from datasets import load_dataset, Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
import lib.attacks as attacks
import numpy as np
import pandas as pd

class PromptGeneratorForTruthfulQA:
    def __init__(self, target_model):
        self.target_model = target_model
        # contrary to <GCG> which has self.prompts, which is created by self.create_prompts based on data from self.logfile, <PromptGeneratorForTruthfulQA2> does not have data by itself
        # TODO : consistency
        
    def generate_prompt(self, data, max_new_len=100):
        # data : 1 sample from Dataset
        max_length = max(list( map(len, self.target_model.tokenizer(data['incorrect_answers'])) ))
        max_new_tokens = max( max_length +2, max_new_len )
        
        # wrapping the question with a template
        conv_template = self.target_model.conv_template
        #conv_template.append_message( conv_template.roles[0], f"{data['question']}" )
        conv_template.append_message( conv_template.roles[0], f"{data['question']}" )
        conv_template.append_message( conv_template.roles[1], f"" )
        prompt = conv_template.get_prompt()
        
        encoding = self.target_model.tokenizer(prompt)
        full_prompt = self.target_model.tokenizer.decode( encoding.input_ids ).replace('<s>','').replace('</s>','')
        conv_template.messages = []
        
        start_index = full_prompt.find(data['question'])
        end_index = start_index + len(data['question'])
        perturbable_prompt = full_prompt[start_index:end_index]
        
        prefix = self.target_model.tokenizer(full_prompt[:start_index])
        prefix_len = len(prefix.input_ids)
        # prefix_len = len(prefix.input_ids) - 1  # -1 for the eos token
        prefix_and_perturbable = self.target_model.tokenizer(full_prompt[:end_index])
        prefix_and_perturbable_len = len(prefix_and_perturbable.input_ids)
        # prefix_and_perturbable_len = len(prefix_and_perturbable.input_ids) - 1  # -1 for the eos token
        
        return attacks.Prompt(full_prompt, perturbable_prompt, max_new_tokens, prefix_len, prefix_and_perturbable_len)




    
class BatchPromptGeneratorForTruthfulQA:
    def __init__(self, target_model):
        self.target_model = target_model
        
    def generate_prompt(self, data, max_new_len=100):           # data : 1 sample
        max_length = max(list( map(len, self.target_model.tokenizer(data['incorrect_answers'])) ))          # data['incorrect_answers'] : list of strings
        max_new_tokens = max( max_length +2, max_new_len )      
        
        # wrapping the question with a template
        conv_template = self.target_model.conv_template
        #conv_template.append_message( conv_template.roles[0], f"{data['question']}" )
        conv_template.append_message( conv_template.roles[0], f"{data['question']}" )
        conv_template.append_message( conv_template.roles[1], f"" )
        prompt = conv_template.get_prompt()
        
        encoding = self.target_model.tokenizer(prompt)
        full_prompt = self.target_model.tokenizer.decode( encoding.input_ids ).replace('<s>','').replace('</s>','')
        conv_template.messages = []
        
        start_index = full_prompt.find(data['question'])
        end_index = start_index + len(data['question'])
        perturbable_prompt = full_prompt[start_index:end_index]
        
        prefix = self.target_model.tokenizer(full_prompt[:start_index])
        prefix_len = len(prefix.input_ids)
        # prefix_len = len(prefix.input_ids) - 1  # -1 for the eos token
        prefix_and_perturbable = self.target_model.tokenizer(full_prompt[:end_index])
        prefix_and_perturbable_len = len(prefix_and_perturbable.input_ids)
        # prefix_and_perturbable_len = len(prefix_and_perturbable.input_ids) - 1  # -1 for the eos token
        
        return attacks.Prompt(full_prompt, perturbable_prompt, max_new_tokens, prefix_len, prefix_and_perturbable_len)
    
    

class PromptGeneratorForSuperGlue:
    def __init__(self, target_model):
        self.target_model = target_model
        
    def generate_prompt(self, data, max_new_len=100):
        max_length = max(list( map(len, self.target_model.tokenizer(data['hypothesis'])) ))
        max_new_tokens = max( max_length +2, max_new_len )
        
        # wrapping the question with a template
        conv_template = self.target_model.conv_template
        #conv_template.append_message( conv_template.roles[0], f"{data['question']}" )
        conv_template.append_message( conv_template.roles[0], f"" )
        conv_template.append_message( conv_template.roles[1], f"Is the premise entailed by the hypothesis?\nPremise: {data['premise']}\nHypothesis: {data['hypothesis']}\nAnswer: " )
        prompt = conv_template.get_prompt()
        
        encoding = self.target_model.tokenizer(prompt)
        full_prompt = self.target_model.tokenizer.decode( encoding.input_ids ).replace('<s>','').replace('</s>','')
        conv_template.messages = []
        
        start_index = full_prompt.find(data['premise'])
        end_index = full_prompt.find(f"{data['hypothesis']}") + len(f"{data['hypothesis']}")
        perturbable_prompt = full_prompt[start_index:end_index]
        
        prefix = self.target_model.tokenizer(full_prompt[:start_index])
        prefix_len = len(prefix.input_ids)
        # prefix_len = len(prefix.input_ids) - 1  # -1 for the eos token
        prefix_and_perturbable = self.target_model.tokenizer(full_prompt[:end_index])
        prefix_and_perturbable_len = len(prefix_and_perturbable.input_ids)
        # prefix_and_perturbable_len = len(prefix_and_perturbable.input_ids) - 1  # -1 for the eos token
        
        return attacks.Prompt(full_prompt, perturbable_prompt, max_new_tokens, prefix_len, prefix_and_perturbable_len)






# batch version        
class BatchPromptGenerator:
    def __init__(self, target_model):
        # target_model : <LLM>,     data : list of strings
        self.target_model = target_model
        self.batches = None
    
    def make_batches(self, data, batch_size = 8):    
        # batch_size : number of <Prompt>s in a single chunk
        self.batches = divide_into_chunks(self._generate_whole_prompt(data), chunk_size = batch_size)            # list of lists of <Prompt>s
    
    def get_full_prompts(self, batch_prompt):
        #return [ [prompt.full_prompt for prompt in batch] for batch in self.batches]
        return [prompt.full_prompt for prompt in batch_prompt]
        
    def _generate_single_prompt(self, sample, max_new_len=100, set_system_prompt = False):                   
        # sample : 1 question (string)
        #max_length = max(list( map(len, self.target_model.tokenizer(data['incorrect_answers'])) ))             # data['incorrect_answers'] : list of strings
        max_length = len( self.target_model.tokenizer(sample) ) 
        max_new_tokens = max( max_length +2, max_new_len )
        
        if set_system_prompt:
            # setting the prompt;   with system prompt
            # wrapping the question with a template
            conv_template = self.target_model.conv_template
            #conv_template.set_system_message('an example system message')
            conv_template.append_message( conv_template.roles[0], f"{sample}" )
            conv_template.append_message( conv_template.roles[1], f"" )
            prompt = conv_template.get_prompt()
        
        else:
            # setting the prompt;   without system prompt
            prompt = f'{sample}'
        
        encoding = self.target_model.tokenizer(prompt)
        full_prompt = self.target_model.tokenizer.decode( encoding.input_ids ).replace('<s>','').replace('</s>','')
        if set_system_prompt:
            conv_template.messages = []
        
        start_index = full_prompt.find(sample)
        end_index = start_index + len(sample)
        perturbable_prompt = full_prompt[start_index:end_index]
        
        prefix = self.target_model.tokenizer(full_prompt[:start_index])
        prefix_len = len(prefix.input_ids)
        # prefix_len = len(prefix.input_ids) - 1  # -1 for the eos token
        prefix_and_perturbable = self.target_model.tokenizer(full_prompt[:end_index])
        prefix_and_perturbable_len = len(prefix_and_perturbable.input_ids)
        # prefix_and_perturbable_len = len(prefix_and_perturbable.input_ids) - 1  # -1 for the eos token
        
        return attacks.Prompt(full_prompt, perturbable_prompt, max_new_tokens, prefix_len, prefix_and_perturbable_len)
    
    def _generate_whole_prompt(self, question_list):
        # question : all the questions (list of strings)   ->  return a list of <Prompt>s
        prompt_list = [self._generate_single_prompt(s) for s in question_list]
        return prompt_list
    
    
    
def divide_into_chunks(object_list, chunk_size = 8):
    # given a list of objects, convert them into batches by returning a list of chunks(=list of objects)
    return list(map( 
                    lambda x : object_list[x*chunk_size : (x+1)*chunk_size], 
                    list(range( int(np.ceil( len(object_list) / chunk_size )) )) 
                    ))



# DatasetDict
class MMLUBatchPromptGenerator:
    def __init__(self, target_model):
        # target_model : <LLM>,     data : list of strings
        self.target_model = target_model
    
    def make_batches(self, dataset_dict, batch_size = 8):   
        """
        Args:
            dataset_dict : DatasetDict
            batch_size : number of <Prompt>s in a single chunk
        return:
            dict{dataset_name : dict{
                'prompt_batches' : list of lists of <Propmt>s
                'answer_batches' : list of lists of answer letters
            }}
        """
        return_dict = {}
        for k,v in dataset_dict.items():
            return_dict[k] = {}
            return_dict[k]['prompt_batches'] = divide_into_chunks(self._generate_whole_prompt(v['prompt']), chunk_size = batch_size)
            return_dict[k]['answer_batches'] = divide_into_chunks(v['answer'], chunk_size = batch_size)
        return return_dict
    
    def get_full_prompts(self, batch_prompt):
        #return [ [prompt.full_prompt for prompt in batch] for batch in self.batches]
        return [prompt.full_prompt for prompt in batch_prompt]
        
    def _generate_single_prompt(self, sample, max_new_len=100, set_system_prompt = False):                   
        # sample : 1 prompt string
        max_length = len( self.target_model.tokenizer(sample) ) 
        max_new_tokens = max( max_length +2, max_new_len )
        
        full_prompt = sample
        
        start_index = full_prompt.find(sample)
        end_index = start_index + len(sample)
        perturbable_prompt = full_prompt[start_index:end_index]
        
        prefix = self.target_model.tokenizer(full_prompt[:start_index])
        prefix_len = len(prefix.input_ids)
        # actual prefix length : len(prefix.input_ids) - 1  # -1 for the eos token
        prefix_and_perturbable = self.target_model.tokenizer(full_prompt[:end_index])
        prefix_and_perturbable_len = len(prefix_and_perturbable.input_ids)
        # actual prefix_and_perturbable length : len(prefix_and_perturbable.input_ids) - 1  # -1 for the eos token
        
        return attacks.Prompt(full_prompt, perturbable_prompt, max_new_tokens, prefix_len, prefix_and_perturbable_len)
    
    def _generate_whole_prompt(self, string_list):
        # string_list : all the prompt strings (list of strings)   ->  return a list of <Prompt>s
        prompt_list = [self._generate_single_prompt(s) for s in string_list]
        return prompt_list