# import requests
# import pandas as pd
# import matplotlib.pyplot as plt
# # Define the IntAct API endpoint
# api_url = "http://www.ebi.ac.uk/Tools/webservices/psicquic/intact/webservices/current/search/query/species:human?firstResult=0&maxResults=100"

# # Make a request to the API
# response = requests.get(api_url)

# # Check if the request was successful (status code 200)
# if response.status_code == 200:
#     # Assuming the response is plain text, split lines to get individual records
#     data_lines = response.text.splitlines()

#     # Define columns for the dataset
#     columns = ["Gene1", "Gene2", "InteractionType", "Miscore"]

#     # Create an empty list to store data
#     data = []

#     # Extract relevant information from the API response and populate the list
#     for line in data_lines:
#         # Split line into fields based on the separator (adjust as needed)
#         fields = line.split('\t')

#         # Assuming you can identify the relevant fields from the response
#         gene1 = fields[0]
#         gene2 = fields[1]
#         interaction_type = fields[2]
#         miscore = fields[-1]  # Assuming miscore is the last field

#         # Append a new row to the data list
#         data.append({"Gene1": gene1, "Gene2": gene2, "InteractionType": interaction_type, "Miscore": miscore})

#     # Create a DataFrame from the list
#     df = pd.DataFrame(data, columns=columns)

#     # Display the first few rows of the DataFrame
#     print(df.head())

#     # Visualize the distribution of the target variable (InteractionType)
#     df["InteractionType"].value_counts().plot(kind="bar", title="Interaction Type Distribution")
#     plt.show()

# else:
#     print(f"Failed to retrieve data from the IntAct API. Status Code: {response.status_code}")

##**************************************************************************************************


# import pandas as pd

# # Replace 'your_file_path' with the actual path to your file
# file_path = "C:/Users/Gokmend/source/repos/PythonApplication1/uniprotKB.txt"

# # Define column names based on the structure of your data
# columns = ["Protein1", "Protein2", "Interaction1", "Interaction2", "GeneInfo1", "GeneInfo2", "InteractionType", "Authors", "Publication", "Taxonomy1", "Taxonomy2", "InteractionType2", "SourceDatabase", "InteractionID", "Miscore"]

# # Read the tab-separated file into a dataframe with specified column names
# df = pd.read_csv(file_path, sep='\t', header=None, names=columns, quoting=3, engine='python')

# # Display the dataframe
# print(df)

import pandas as pd
from IPython.display import display

# Replace 'your_file_path' with the actual path to your file
file_path = "C:/Users/Gokmend/source/repos/PythonApplication1/uniprotKB.txt"

# Define column names based on the structure of your data
columns = ["Protein1", "Protein2", "Interaction1", "Interaction2", "GeneInfo1", "GeneInfo2", "InteractionType", "Authors", "Publication", "Taxonomy1", "Taxonomy2", "InteractionType2", "SourceDatabase", "InteractionID", "Miscore"]

# Read the tab-separated file into a dataframe with specified column names
df = pd.read_csv(file_path, sep='\t', header=None, names=columns, quoting=3, engine='python')

# Display the entire DataFrame
for index, row in df.iterrows():
    print(row)
    print()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load your data
file_path = "C:/Users/Gokmend/source/repos/PythonApplication1/uniprotKB.txt"
columns = ["Protein1", "Protein2", "Interaction1", "Interaction2", "GeneInfo1", "GeneInfo2", "InteractionType", "Authors", "Publication", "Taxonomy1", "Taxonomy2", "InteractionType2", "SourceDatabase", "InteractionID", "Miscore"]
df = pd.read_csv(file_path, sep='\t', header=None, names=columns, quoting=3, engine='python')

# Extract relevant features and target
features = ["Protein1", "Protein2", "GeneInfo1", "GeneInfo2", "Taxonomy1", "Taxonomy2", "Miscore"]
target = "Disease"  # Replace with the actual column name indicating the disease label
def map_interaction_to_disease(interaction_type):
    # Replace this logic with your actual mapping rules
    diseased_interactions = ["psi-mi:\"MI:0096\"(pull down)", "psi-mi:\"MI:0914\"(association)"]
    
    if interaction_type in diseased_interactions:
        return "Diseased"
    else:
        return "Not Diseased"
# Assuming you have a function to map interactions to disease labels, create a new column "Disease"
df["Disease"] = df["InteractionType"].apply(lambda x: map_interaction_to_disease(x))  # Replace with your actual mapping function

# Extracting protein, gene names, and taxonomy from the GeneInfo columns
df["Protein1_Name"] = df["GeneInfo1"].str.extract(r'uniprotkb:(\w+)')
df["Protein2_Name"] = df["GeneInfo2"].str.extract(r'uniprotkb:(\w+)')
df["Gene1_Name"] = df["GeneInfo1"].str.extract(r'psi-mi:(\w+)_')
df["Gene2_Name"] = df["GeneInfo2"].str.extract(r'psi-mi:(\w+)_')
df["Taxonomy1_Num"] = df["Taxonomy1"].str.extract(r'taxid:(\d+)')
df["Taxonomy2_Num"] = df["Taxonomy2"].str.extract(r'taxid:(\d+)')

# Separate numerical and categorical features
numerical_features = ["Miscore"]
categorical_features = ["Protein1", "Protein2", "GeneInfo1", "GeneInfo2", "Taxonomy1", "Taxonomy2"]

# Create transformers
numerical_transformer = Pipeline(steps=[('dummy', 'passthrough')])  # No transformation for numerical features
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', dtype=int))])  # One-hot encode categorical features with specified dtype

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Append classifier to preprocessing pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression())])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)










