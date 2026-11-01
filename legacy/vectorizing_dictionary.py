import csv
import os
import torch
from transformers import BertTokenizer, BertModel
from math import sqrt
def normalize(vector):
    # Note : This modifies your input vector directly
    # We first find the magnitude of the given vector

    magnitude = 0
    for element in vector:
        magnitude += (element * element)
    magnitude = sqrt(magnitude)

    # We then go through each element and divide it by the magnitude
    for i, element in enumerate(vector):
        vector[i] = element / magnitude

    return vector

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertModel.from_pretrained('bert-base-uncased')

file_path = os.path.join(os.path.expanduser("~"), "Downloads", "symptomssingle.csv")

disease_symptoms_dict = {}  # Create an empty dictionary to store disease-symptoms associations

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)

    for column in reader:
        string = column[0]
        split_string = string[1:-1].split(',')
        symptoms = [entry.split('":"')[-1].strip('"}]') for entry in split_string]
        symptoms = [symptom for symptom in symptoms if not symptom.isdigit()]
        last_word = symptoms.pop()
        if '","' in last_word:
            last_word = last_word.split('","')[1]
            symptoms.append(last_word)
        output = symptoms

        input_query = tokenizer.encode_plus(" and ".join(symptoms),  # Combine symptoms into a single sentence
                                            add_special_tokens=True,
                                            return_attention_mask=True,
                                            return_tensors='pt')
        input_query = input_query['input_ids']

        output_query = model(input_query)
        output_CLS = output_query.last_hidden_state.squeeze(0)[0]

        # Add disease-symptoms association to the dictionary
        disease_symptoms_dict[last_word] = normalize(output_CLS.detach().numpy())


# Example
print("Disease: Panic disorder")
print("Vectorized Symptoms:", disease_symptoms_dict.get('Panic disorder'))