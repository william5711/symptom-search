import csv
import os

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
        output = [last_word] + symptoms

        # Assuming that the last_word represents the disease name
        disease_name = last_word
        disease_symptoms = symptoms

        # Add disease-symptoms association to the dictionary
        disease_symptoms_dict[disease_name] = disease_symptoms

# Function to predict the disease based on at least two similar symptoms
# Function to predict the disease based on at least two similar symptoms
def predict_disease(input_symptoms):
    for disease, symptoms in disease_symptoms_dict.items():
        common_symptoms_count = len(set(input_symptoms).intersection(symptoms))
        if common_symptoms_count >= 2:
            return disease

    return None  # If no disease with at least two similar symptoms is found


# Get user input for symptoms
user_input = input("Enter your symptoms (comma-separated): ")
user_input_symptoms = [symptom.strip() for symptom in user_input.split('""')]

predicted_disease = predict_disease(user_input_symptoms)

if predicted_disease:
    print("User symptoms:", user_input_symptoms)
    print("Predicted disease:", predicted_disease)
else:

    print("No matching disease found with at least two similar symptoms.")
