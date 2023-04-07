# CSV BUILDER FOR KAGGLE ANTI-SPOOFING DATASET

import csv
import os

# Header for the CSV File
header = ["sno", "name", "label"]

# Path to the folder of client images
# Change the Path depending upon the location
directory = "../csv_dataset_builder/LCC_FASD/"
development_directory = "../csv_dataset_builder/LCC_FASD/LCC_FASD_development/"
evaluation_directory = "../csv_dataset_builder/LCC_FASD/LCC_FASD_evaluation/"

# Opening the CSV File
real_count = 0
spoof_count = 0

with open("./dataset_csv/kaggle_images.csv", "w", newline="") as f:
  writer = csv.writer(f)

  # Write the header
  writer.writerow(header)

  # Value for the "label" Columnn
  # Real -> 1
  # Spoof -> 0

  # Looping over the main sub-directories: Devlopment, Evaluation and Training
  for folderName in os.listdir(directory):
    # Looping over the secondary sub-directory: real and spoof
    for subFolder in os.listdir(os.path.join(directory, folderName)):
      # Looping over the images and writing them
      for fileName in os.listdir(os.path.join(directory, folderName, subFolder)):

        if subFolder == "real":
          writer.writerow([real_count + spoof_count, os.path.join(directory, folderName, subFolder, fileName), 1])
          real_count += 1
        else:
          writer.writerow([real_count + spoof_count, os.path.join(directory, folderName, subFolder, fileName), 0])
          spoof_count += 1
  

with open("./dataset_csv/kaggle_dataset_details.csv", "w", newline="") as f:
  writer = csv.writer(f)

  writer.writerow(["Real Face Images", "Spoof Face Images", "Total Images"])
  writer.writerow([real_count + 1, spoof_count, real_count + spoof_count + 1])
  