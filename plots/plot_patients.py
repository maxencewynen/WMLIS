import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to handle missing values
def handle_missing_values(value):
    if pd.isnull(value):
        return "Unknown"
    return value

# Function to generate histograms
def generate_histogram(data, column_name, save_directory):
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=10, alpha=0.7)
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column_name}")
    plt.grid(True)
    plt.savefig(os.path.join(save_directory, f"{column_name}_histogram.png"))
    plt.close()

# Function to generate bar plots
def generate_bar_plot(data, column_name, save_directory):
    plt.figure(figsize=(10, 6))
    data.value_counts().plot(kind="bar")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title(f"Bar Plot of {column_name}")
    plt.grid(True)
    plt.savefig(os.path.join(save_directory, f"{column_name}_bar_plot.png"))
    plt.close()

# Specify the input Excel file and columns of interest
excel_file = "D:/R4/Instant_Segmentation_PatientDatabase_Variables.xlsx"
columns_of_interest = ["age", "sex", "BMI", "diagnosis", "clinical_site", "Number_of_Lesions", "total_lesion_volume", "number_PRL"]

# Specify the directory to save the plots
save_directory = "D:/R4/plots"

# Read the Excel file
df = pd.read_excel(excel_file)

# Handle missing values
df["sex"] = df["sex"].apply(handle_missing_values)
df["diagnosis"] = df["diagnosis"].apply(handle_missing_values)
df["clinical_site"] = df["clinical_site"].apply(handle_missing_values)

# Generate histograms and bar plots
for column in columns_of_interest:
    if column in df.columns:
        if column in ["age", "BMI", "Number_of_Lesions", "total_lesion_volume", "number_PRL"]:
            generate_histogram(df[column], column, save_directory)
        else:
            generate_bar_plot(df[column], column, save_directory)

