from .load_raw_data import select_data
from train_data.model.models import OpenAIModel

import random
from tqdm import tqdm
import json

random.seed(42)

def prep_squad(save_path, lim=1000):
    _, data = select_data('squad')

    # randomly select <lim> data points
    sampled_data = random.sample(data, lim)
    inputs = get_data_inputs('squad', sampled_data)

    # get concept outputs prepped data
    instruction = 'Given the input context, generate a question that can be answered using the context. Ensure you <DIRECTION> the <CONCEPT> of the question generated. Return only the output question and nothing else.'
    concepts = ['humour', 'difficulty', 'linguistic complexity']
    concept_descriptions = ['How funny the tone of the question is.', 'How challenging the question is based on how many logical inferences have to made to find the answer in the context.', 'The complexity of the language choice in the question, such as word choice, structure and grammar.']
    prepped_data = prep_concepts_data(inputs, instruction, concepts, concept_descriptions)

    # save prepped data
    with open(save_path, 'w') as f:
        json.dump(prepped_data, f)

def get_data_inputs(data_name, data):
    '''
        Get the inputs to generate outputs
    '''
    if data_name == 'squad':
        inputs = [d['context'] for d in data]
    return inputs

def prep_concepts_data(inputs, instruction, concepts, concept_descriptions, directions=['maximize', 'minimize']):
    llm = OpenAIModel('gpt4o')
    prepped_data = []
    for input in tqdm(inputs):
        output_concept = {}
        for concept, description in zip(concepts, concept_descriptions):
            for direction in directions:
                curr_instruction = instruction.replace('<CONCEPT>', concept).replace('<DIRECTION>', direction)
                prompt = f"{curr_instruction}\nDefinition of {concept}: {description}\nInput: {input}\nOutput:\n\n"
                out = llm.predict(prompt)
                if concept not in output_concept:
                    output_concept[concept] = {}
                output_concept[concept][direction] = out
        prepped_data.append({'input': input, 'output': output_concept})
    return prepped_data


