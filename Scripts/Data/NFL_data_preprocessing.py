import pandas as pd
import os

def open_all_tsv(folder_path):
    dataframes = {}
    for file in os.listdir(folder_path):
        if file.endswith(".tsv"):
            df_name = os.path.splitext(file)[0]  # Get the file name without extension
            dataframes[df_name] = pd.read_csv(os.path.join(folder_path, file), sep="\t")
    print("Conversion completed for all TSV files in the folder.")        
    return dataframes





