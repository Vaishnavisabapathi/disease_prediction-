import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the dataset
df = pd.read_csv('dataset.csv')

# Get the target column
y = df['Disease']

# Drop the 'Disease' column to get symptom columns
symptom_columns = df.columns[1:]

# Get all unique symptoms from the dataset
all_symptoms = set()
for col in symptom_columns:
    all_symptoms.update(df[col].dropna().unique())

# Create binary columns for each symptom
for symptom in all_symptoms:
    df[symptom] = df[symptom_columns].apply(lambda row: int(symptom in row.values), axis=1)

# Final features (binary symptom matrix)
X = df[list(all_symptoms)]

# Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the trained model to a pickle file
with open('disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully.")
