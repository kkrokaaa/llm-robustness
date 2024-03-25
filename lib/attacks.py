import json
import pandas as pd

class Prompt:
    def __init__(self, full_prompt, perturbable_prompt, max_new_tokens, start_noise_idx, end_noise_idx):
        self.full_prompt = full_prompt
        self.perturbable_prompt = perturbable_prompt
        self.max_new_tokens = max_new_tokens
        self.start_noise_idx = start_noise_idx
        self.end_noise_idx = end_noise_idx

    def perturb(self, perturbation_fn):
        perturbed_prompt = perturbation_fn(self.perturbable_prompt)
        self.full_prompt = self.full_prompt.replace(
            self.perturbable_prompt,
            perturbed_prompt
        )
        self.perturbable_prompt = perturbed_prompt

class Attack:
    def __init__(self, logfile, target_model):
        self.logfile = logfile
        self.target_model = target_model

class GCG(Attack):

    """Greedy Coordinate Gradient attack.

    Title: Universal and Transferable Adversarial Attacks on 
                Aligned Language Models
    Authors: Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson
    Paper: https://arxiv.org/abs/2307.15043
    """

    def __init__(self, logfile, target_model):
        super(GCG, self).__init__(logfile, target_model)

        with open(self.logfile, 'r') as f:
            log = json.load(f)

        self.goals = log['goal']
        self.targets = log['target']
        self.controls = log['controls']

        self.prompts = [
            self.create_prompt(g, c, t)
            for (g, c, t) in zip(self.goals, self.controls, self.targets)
        ]

    def create_prompt(self, goal, control, target, max_new_len=100):
        """Create GCG prompt."""

        max_new_tokens = max(
            len(self.target_model.tokenizer(target).input_ids) + 2,
            max_new_len
        )

        # Create full prompt for LLM
        conv_template = self.target_model.conv_template
        conv_template.append_message(
            conv_template.roles[0], f"{goal} {control}"
        )
        conv_template.append_message(conv_template.roles[1], f"")
        prompt = conv_template.get_prompt()

        # As per the GCG source code, we encode then decode the full prompt
        encoding = self.target_model.tokenizer(prompt)
        full_prompt = self.target_model.tokenizer.decode(
            encoding.input_ids
        ).replace('<s>','').replace('</s>','')

        # Clear the conv template
        conv_template.messages = []
        
        start_index = full_prompt.find(goal)
        end_index = full_prompt.find(control) + len(control)
        perturbable_prompt = full_prompt[start_index:end_index]
        
        # TODO: check defense encoding and verify the indices
        prefix = self.target_model.tokenizer(full_prompt[:start_index])
        prefix_len = len(prefix.input_ids)
        # prefix_len = len(prefix.input_ids) - 1  # -1 for the eos token
        prefix_and_perturbable = self.target_model.tokenizer(full_prompt[:end_index])
        prefix_and_perturbable_len = len(prefix_and_perturbable.input_ids)
        # prefix_and_perturbable_len = len(prefix_and_perturbable.input_ids) - 1  # -1 for the eos token
        
        # print(f'start_noise_idx: {prefix_len}')
        # print(f'end_noise_idx: {prefix_and_perturbable_len}')
        
        return Prompt(
            full_prompt, 
            perturbable_prompt, 
            max_new_tokens,
            prefix_len,
            prefix_and_perturbable_len
        )

class PAIR(Attack):

    """Prompt Automatic Iterative Refinement (PAIR) attack.

    Title: Jailbreaking Black Box Large Language Models in Twenty Queries
    Authors: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, 
                George J. Pappas, Eric Wong
    Paper: https://arxiv.org/abs/2310.08419
    """

    def __init__(self, logfile, target_model):
        super(PAIR, self).__init__(logfile, target_model)

        df = pd.read_pickle(logfile)
        jailbreak_prompts = df['jailbreak_prompt'].to_list()
        
        self.prompts = [
            self.create_prompt(prompt)
            for prompt in jailbreak_prompts
        ]
        
    def create_prompt(self, prompt):

        conv_template = self.target_model.conv_template
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        full_prompt = conv_template.get_prompt()

        # Clear the conv template
        conv_template.messages = []

        return Prompt(
            full_prompt,
            prompt,
            max_new_tokens=100
        )