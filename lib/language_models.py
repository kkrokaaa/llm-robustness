import torch
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy

class LLM:

    """Forward pass through a LLM."""

    def __init__(
        self, 
        model_path, 
        tokenizer_path, 
        conv_template_name,
        device
    ):

        # Language model
        torch_dtype = torch.float32 if 'vicuna' in model_path else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True,
            device_map='auto',
        ).eval()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.tokenizer.padding_side = 'left'
        if 'llama' in tokenizer_path or 'Llama' in tokenizer_path:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fastchat conversation template
        self.conv_template = get_conversation_template(
            conv_template_name
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()

    def __call__(self, batch, start_noise_idx, end_noise_idx, noise_scale, max_new_tokens=100, original_prompt=None):
        
        if original_prompt:
            #print("*** appending original prompt ***")
            batch.append(original_prompt)
        
        
        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(
            batch, 
            padding=True, 
            truncation=False, 
            return_tensors='pt'
        )
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)
    
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L982-L983
        # TODO: override generate() and add new parameter for inputs_embeds
        # https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/generation/utils.py#L1218 (1118)
        # TODO: override this block of code
        # https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/generation/utils.py#L4530C1-L4535C14
        
        # Forward pass through the LLM
        try:
            outputs = self.model.generate(
                batch_input_ids, 
                # inputs_embeds=inputs_embeds,
                attention_mask=batch_attention_mask, 
                max_new_tokens=max_new_tokens,
                start_noise_idx=start_noise_idx,
                end_noise_idx=end_noise_idx,
                noise_scale=noise_scale,
            )
        except RuntimeError as e:
            print(e)
            return []

        # Decode the outputs produced by the LLM
        batch_outputs = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        gen_start_idx = [
            len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)) 
            for i in range(len(batch_input_ids))
        ]
        batch_outputs = [
            output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)
        ]
        return batch_outputs
    
    
    def get_probs(self, batch, start_noise_idx, end_noise_idx, original_prompt=None):        
        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(
            batch, 
            padding=True, 
            truncation=False, 
            return_tensors='pt'
        )
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1238C16-L1238C38
        outputs = self.model(
            batch_input_ids, 
            attention_mask=batch_attention_mask, 
        )
        probs = torch.softmax(outputs.logits, dim=-1)
        
        # print(outputs.logits.shape) # torch.Size([b, xx, 32000])
        selected_probs = torch.ones_like(batch_input_ids, dtype=torch.float, device=self.model.device)       # [b, len]
        for i in range(len(batch_input_ids)):
            input_ids = batch_input_ids[i]
            prob = probs[i]
            for j in range(prob.shape[0]-1):
                token = self.tokenizer.decode(input_ids[j+1].item())
                # print(f'token: {token}, prob: {prob[j, input_ids[j+1]]}')
            # probs starting from the second token, set first token prob to 1
            selected_probs[i, 1:] = prob[range(prob.shape[0]-1), input_ids[1:]]
        
        return selected_probs



class BatchLLM:
    def __init__(self, model_path, tokenizer_path, conv_template_name, device):
        # Language model
        torch_dtype = torch.float32 if 'vicuna' in model_path else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype, 
            trust_remote_code=True,
            low_cpu_mem_usage=True, 
            use_cache=True, 
            device_map='auto'
        ).eval()
        if 'llama-2' in model_path or 'Llama-2' in model_path:                      #
            self.model = self.model.bfloat16().eval()#to(device).eval()             #error : probability tensor contains either `inf`, `nan` or element < 0

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, 
            trust_remote_code=True, 
            use_fast=False
        )
        self.tokenizer.padding_side = 'left'
        if 'llama' in tokenizer_path or 'Llama' in tokenizer_path:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fastchat conversation template
        self.conv_template = get_conversation_template(
            conv_template_name
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()

    def __call__(self, batch, start_noise_idx_list, end_noise_idx_list, noise_scale, max_new_tokens=100, original_prompt=None):
        
        if original_prompt:
            batch.append(original_prompt)
        
        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(
            batch, 
            padding=True, 
            truncation=False, 
            return_tensors='pt'
        )
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)
    
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L982-L983
        # TODO: override generate() and add new parameter for inputs_embeds
        # https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/generation/utils.py#L1218 (1118)
        # TODO: override this block of code
        # https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/generation/utils.py#L4530C1-L4535C14
            
        # Forward pass through the LLM
        try:
            outputs = self.model.generate(
                batch_input_ids, 
                # inputs_embeds=inputs_embeds,
                attention_mask=batch_attention_mask, 
                max_new_tokens=max_new_tokens,
                start_noise_idx_list=start_noise_idx_list,
                end_noise_idx_list=end_noise_idx_list,
                noise_scale=noise_scale,
            )
        except RuntimeError as e:
            print('cannot generate; error raised : ', e)
            return []

        # Decode the outputs produced by the LLM
        batch_outputs = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        gen_start_idx = [
            len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)) 
            for i in range(len(batch_input_ids))
        ]
        batch_outputs = [
            output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)
        ]
        return batch_outputs
    
    
    def get_probs(self, batch, start_noise_idx, end_noise_idx, original_prompt=None):        
        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(
            batch, 
            padding=True, 
            truncation=False, 
            return_tensors='pt'
        )
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1238C16-L1238C38
        outputs = self.model(
            batch_input_ids, 
            attention_mask=batch_attention_mask, 
        )
        probs = torch.softmax(outputs.logits, dim=-1)
        
        # print(outputs.logits.shape) # torch.Size([b, xx, 32000])
        selected_probs = torch.ones_like(batch_input_ids, dtype=torch.float, device=self.model.device)       # [b, len]
        for i in range(len(batch_input_ids)):
            input_ids = batch_input_ids[i]
            prob = probs[i]
            selected_probs[i, 1:] = prob[range(prob.shape[0]-1), input_ids[1:]]
        
        return selected_probs



class BatchLLMForward:
    def __init__(self, model_path, tokenizer_path, conv_template_name, device):
        # Language model
        torch_dtype = torch.float32 if 'vicuna' in model_path else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype, 
            trust_remote_code=True,
            low_cpu_mem_usage=True, 
            use_cache=True, 
            device_map='auto'
        ).eval()
        if 'llama-2' in model_path or 'Llama-2' in model_path:                      #
            #self.model = self.model.bfloat16().eval()#to(device).eval()             #error : probability tensor contains either `inf`, `nan` or element < 0
            pass

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, 
            trust_remote_code=True, 
            use_fast=False
        )
        self.tokenizer.padding_side = 'left'
        if 'llama' in tokenizer_path or 'Llama' in tokenizer_path:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fastchat conversation template
        self.conv_template = get_conversation_template(
            conv_template_name
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()

    def __call__(self, batch, start_noise_idx_list, end_noise_idx_list, noise_scale, max_new_tokens=100, original_prompt=None):
        
        if original_prompt:
            batch.append(original_prompt)
        
        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(
            batch, 
            padding=True, 
            truncation=False, 
            return_tensors='pt'
        )
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)
        
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L942
        # Forward pass through the LLM
        try:
            batch_outputs = self.model(
                batch_input_ids, 
                # inputs_embeds=inputs_embeds,
                attention_mask=batch_attention_mask,
                start_noise_idx_list=start_noise_idx_list,
                end_noise_idx_list=end_noise_idx_list,
                noise_scale=noise_scale,
            )
        except RuntimeError as e:
            print('cannot forward; error raised : ', e)
            return []

        return batch_outputs.logits
    
    
    def get_probs(self, batch, start_noise_idx, end_noise_idx, original_prompt=None):        
        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(
            batch, 
            padding=True, 
            truncation=False, 
            return_tensors='pt'
        )
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1238C16-L1238C38
        outputs = self.model(
            batch_input_ids, 
            attention_mask=batch_attention_mask, 
        )
        probs = torch.softmax(outputs.logits, dim=-1)
        
        # print(outputs.logits.shape) # torch.Size([b, xx, 32000])
        selected_probs = torch.ones_like(batch_input_ids, dtype=torch.float, device=self.model.device)       # [b, len]
        for i in range(len(batch_input_ids)):
            input_ids = batch_input_ids[i]
            prob = probs[i]
            selected_probs[i, 1:] = prob[range(prob.shape[0]-1), input_ids[1:]]
        
        return selected_probs



class LLMForward:
    def __init__(self, model_path, tokenizer_path, conv_template_name, device):
        # Language model
        torch_dtype = torch.float32 if 'vicuna' in model_path else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype, 
            trust_remote_code=True,
            low_cpu_mem_usage=True, 
            use_cache=True, 
            device_map='auto'
        ).eval()
        if 'llama-2' in model_path or 'Llama-2' in model_path:                      #
            #self.model = self.model.bfloat16().eval()#to(device).eval()             #error : probability tensor contains either `inf`, `nan` or element < 0
            pass

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, 
            trust_remote_code=True, 
            use_fast=False
        )
        self.tokenizer.padding_side = 'left'
        if 'llama' in tokenizer_path or 'Llama' in tokenizer_path:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fastchat conversation template
        self.conv_template = get_conversation_template(
            conv_template_name
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()

    def __call__(self, tokenized_input, start_noise_idx, end_noise_idx, noise_scale, max_new_tokens=100, num_copies = 8, noise_level = 1, original_prompt=None):
        
        #if original_prompt:
        #    batch.append(original_prompt)
        input_id = tokenized_input['input_ids'].to(self.model.device)
        attention_mask = tokenized_input['attention_mask'].to(self.model.device)
        input_ids = []
        attention_masks = []
        for _ in range(num_copies):
            id_copy = copy.deepcopy(input_id)
            attention_copy = copy.deepcopy(attention_mask)
            input_ids.append(id_copy)
            attention_masks.append(attention_copy)
        input_ids = torch.cat(input_ids, axis=0)
        attention_masks = torch.cat(attention_masks, axis=0)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L942
        # Forward pass through the LLM
        try:
            batch_outputs = self.model(
                input_ids, 
                # inputs_embeds=inputs_embeds,
                attention_mask=attention_masks,
                start_noise_idx=start_noise_idx,
                end_noise_idx=end_noise_idx,
                #noise_scale=noise_scale,
                noise_level=noise_level
            )
        except RuntimeError as e:
            print('cannot forward; error raised : ', e)
            return []

        return batch_outputs.logits
    
    
    def get_probs(self, tokenized_input):
        
        input_ids = tokenized_input['input_ids'].to(self.model.device)                                  # 1 x len
        attention_masks = tokenized_input['attention_mask'].to(self.model.device)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1238C16-L1238C38
        outputs = self.model(
            input_ids, 
            attention_mask=attention_masks,
        )
        probs = torch.softmax(outputs.logits, dim=-1)                                                   # 1 x len x vocab size
        
        # print(outputs.logits.shape) # torch.Size([b, xx, 32000])
        selected_probs = torch.ones_like(input_ids, dtype=torch.float, device=self.model.device)        # 1 x len
        selected_probs[:, 1:] = probs[0,range(probs.shape[1] - 1), input_ids[0, 1:]]
        
        return selected_probs