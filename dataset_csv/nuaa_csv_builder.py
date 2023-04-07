# CSV BUILDER FOR NUAA DATASET

import csv
import os

# Header for the CSV File
header = ["sno", "name", "label"]

# Path to the folder of client images
# Change the Path depending upon the location
real_directory = "../csv_dataset_builder/Detectedface/ClientFace/"
spoof_directory = "../csv_dataset_builder/Detectedface/ImposterFace/"

# Opening the CSV File
real_count = 0
spoof_count = 0
with open("./dataset_csv/nuaa_images.csv", "w", newline="") as f:
  writer = csv.writer(f)

  # Wrie the header
  writer.writerow(header)

  # Value for the "label" Columnn
  # Real -> 1
  # Spoof -> 0
  
  # Looping over the subdirectories in the CLIENT Folder
  for folderName in os.listdir(real_directory):
    # Looping over the images in the subdirectory and write them
    for fileName in os.listdir(os.path.join(real_directory, folderName) + "/"):
      writer.writerow([real_count, os.path.join(real_directory, folderName, fileName), 1])
      real_count += 1

  # Looping over the subdirectoris in the IMPOSTER Folder
  for folderName in os.listdir(spoof_directory):
    # Looping over the images in the subdirectory and write them
    for fileName in os.listdir(os.path.join(spoof_directory, folderName) + "/"):
      writer.writerow([real_count + spoof_count, os.path.join(spoof_directory, folderName, fileName), 0])
      spoof_count += 1

# Write the number of files
with open("./dataset_csv//nuaa_dataset_details.csv", "w", newline="") as f:
  writer = csv.writer(f)

  writer.writerow(["Real Face Images", "Spoof Face Images", "Total Images"])
  writer.writerow([real_count + 1, spoof_count, real_count + spoof_count + 1])
