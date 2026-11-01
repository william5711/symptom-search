import torch
from transformers import BertTokenizer, BertModel
from math import sqrt
import csv
import os
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertModel.from_pretrained('bert-base-uncased')

# Normalize the vector
def normalize(vector):
    magnitude = 0
    for element in vector:
        magnitude += (element * element)
    magnitude = sqrt(magnitude)

    normalized_vector = [element / magnitude for element in vector]
    return normalized_vector


# Get user input for symptoms
user_input = input("Enter your symptoms: ")

# Tokenize user input (query is a string user_input)
input_query = tokenizer.encode_plus(user_input,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    return_tensors='pt')
input_query = input_query['input_ids']

with torch.no_grad():
    output_query = model(input_query)
    output_CLS = output_query.last_hidden_state.squeeze(0)[0]

normalized_output_CLS = normalize(output_CLS.tolist())

print("Vectorized Symptoms:", normalized_output_CLS)

# Load the symptoms-disease dictionary
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

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    return dot_product / (magnitude1 * magnitude2)

user_vector = normalized_output_CLS

# Find similarities and rank them
similarities = {}
for disease, vector2 in disease_symptoms_dict.items():
    similarity = cosine_similarity(user_vector, vector2)
    similarities[disease] = similarity

sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

# Get the top 10 most similar diseases
top_100_similarities = sorted_similarities[:10]

# Output the ranked similar diseases
print("User Input Symptoms:", user_input)
print("Top 10 Most Similar Diseases based on Similarity:")
for idx, (disease, similarity) in enumerate(top_100_similarities, 1):
    print(f"{idx}. Disease: {disease}, Similarity: {similarity:.4f}")
