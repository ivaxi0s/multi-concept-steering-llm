from openai import OpenAI
import openai

with open('/home/vr313/rds/hpc-work/Projects/Control/multi-concept-steering-llm/experiments/keys/openai.txt', 'r') as f:
    key_str = f.read()
API_KEY = key_str.strip('\n')
openai.api_key = API_KEY

OPENAI_MODELS = {
    "gpt3.5": "gpt-3.5-turbo",
    "gpt4": "gpt-4-turbo",
    "gpt4o": "gpt-4o",
}

class OpenAIModel:
    """Class wrapper for models that interact with a OpenAI API"""

    def __init__(self, model_name = "gpt4o", system_prompt=None):
        self.model_name = OPENAI_MODELS[model_name]
        self.client = OpenAI(api_key=API_KEY)
        self.system_prompt = system_prompt
    
    def predict(self, prompt):
        system_msg = {"role": "system", "content": self.system_prompt}
        user_msg = {"role": "user", "content": prompt}
        if self.system_prompt is None:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=[user_msg], temperature=0
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=[system_msg, user_msg], temperature=0
            )
        return response.choices[0].message.content


    def predict_with_logits(self, prompt, max_tokens=1, logit_bias={}):

        completion = self.client.chat.completions.create(
            model = self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens = max_tokens,
            logit_bias = logit_bias,
            seed = 1,
            logprobs = True,
            top_logprobs = 20,
        )

        sequence_token_logits = completion.choices[0].logprobs.content
        return sequence_token_logits