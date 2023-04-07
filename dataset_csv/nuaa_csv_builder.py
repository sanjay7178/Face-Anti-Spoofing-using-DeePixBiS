# CSV BUILDER FOR NUAA DATASET

import csv
import os

# Header for the CSV File
header = ["name", "label"]

# Path to the folder of client images
# Change the Path depending upon the location
client_directory = "./Detectedface/ClientFace/"
imposter_directory = "./Detectedface/ImposterFace/"

# Opening the CSV File
client_count = 0
imposter_count = 0
with open("nuaa_images.csv", "w", newline="") as f:
  writer = csv.writer(f)

  # Wrie the header
  writer.writerow(header)
  
  # Looping over the subdirectories in the CLIENT Folder
  for folderName in os.listdir(client_directory):
    # Looping over the images in the subdirectory and write them
    for fileName in os.listdir(os.path.join(client_directory, folderName) + "/"):
      writer.writerow([client_count, os.path.join(client_directory, folderName, fileName)])
      client_count += 1

  # Looping over the subdirectoris in the IMPOSTER Folder
  for folderName in os.listdir(imposter_directory):
    # Looping over the images in the subdirectory and write them
    for fileName in os.listdir(os.path.join(imposter_directory, folderName) + "/"):
      writer.writerow([client_count + imposter_count, os.path.join(imposter_directory, folderName, fileName)])
      imposter_count += 1

# Write the number of files
with open("nuaa_dataset_details.csv", "w", newline="") as f:
  writer = csv.writer(f)

  writer.writerow(["Client Face Images", "Imposter Face Images", "Total Images"])
  writer.writerow([client_count + 1, imposter_count, client_count + imposter_count + 1])
