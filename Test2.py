import os
import cv2 as cv
import torch
import torch.nn as nn
from torchvision import transforms
from Model import DeePixBiS
from Loss import PixWiseBCELoss
from Metrics import predict, test_accuracy, test_loss

# Load model and face classifier
model = DeePixBiS()
model.load_state_dict(torch.load('./DeePixBiS.pth'))
model.eval()

faceClassifier = cv.CascadeClassifier('Classifiers/haarface.xml')

# Define image processing pipeline
tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define input and output directories
input_dir = 'path/to/input/directory'
output_dir = 'path/to/output/directory'

# Loop over images in input directory
for filename in os.listdir(input_dir):
    # Read image and convert to grayscale
    img = cv.imread(os.path.join(input_dir, filename))
    grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Detect faces in image
    faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

    # Loop over faces and classify them
    for x, y, w, h in faces:
        faceRegion = img[y:y + h, x:x + w]
        faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)

        # Apply image processing pipeline
        faceRegion = tfms(faceRegion)
        faceRegion = faceRegion.unsqueeze(0)

        # Run classification model
        mask, binary = model.forward(faceRegion)
        res = torch.mean(mask).item()

        # Draw bounding box and label on image
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if res < 0.5:
            cv.putText(img, 'Fake', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            # Save fake image to output directory
            cv.imwrite(os.path.join(output_dir, filename), img)
        else:
            cv.putText(img, 'Real', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    # Display and wait for key press
    cv.imshow('Test', img)
    cv.waitKey(0)
