from transformers import AutoTokenizer, AutoModel
import torch
import os
import torch.nn.functional as F
import csv
import numpy as np

# Normalize the vector
def normalize(vector):
    magnitude = np.linalg.norm(vector)
    normalized_vector = vector / magnitude
    return normalized_vector

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Get user input for symptoms
user_input = input("Enter your symptoms: ")

# Tokenize user input
encoded_input = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform mean pooling
sentence_embeddings = model_output.pooler_output
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1).squeeze().tolist()

# Normalize embeddings
normalized_output_CLS = normalize(sentence_embeddings)

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

        # Tokenize and get sentence embeddings for each disease's symptoms
        encoded_symptoms = tokenizer(" and ".join(symptoms), padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_symptoms)
        disease_embeddings = model_output.pooler_output
        disease_embeddings = F.normalize(disease_embeddings, p=2, dim=1).squeeze().tolist()

        # Add disease-symptoms association to the dictionary
        disease_symptoms_dict[last_word] = normalize(disease_embeddings)

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
