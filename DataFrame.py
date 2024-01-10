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
