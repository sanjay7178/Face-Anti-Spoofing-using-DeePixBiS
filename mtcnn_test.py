import os
import cv2 as cv
import torch
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import MTCNN
from Model import DeePixBiS
from Loss import PixWiseBCELoss
from Metrics import predict, test_accuracy, test_loss

# Load model and face detector
model = DeePixBiS()
model.load_state_dict(torch.load('./DeePixBiS.pth'))
model.eval()

detector = MTCNN()

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
    # Read image and detect faces
    img = cv.imread(os.path.join(input_dir, filename))
    faces = detector.detect(img)

    # Loop over faces and classify them
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        faceRegion = img[y1:y2, x1:x2]

        # Apply image processing pipeline
        faceRegion = tfms(faceRegion)
        faceRegion = faceRegion.unsqueeze(0)

        # Run classification model
        mask, binary = model.forward(faceRegion)
        res = torch.mean(mask).item()

        # Draw bounding box and label on image
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if res < 0.5:
            cv.putText(img, 'Fake', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # Save fake image to output directory
            cv.imwrite(os.path.join(output_dir, filename), img)
        else:
            cv.putText(img, 'Real', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display and wait for key press
    cv.imshow('Test', img)
    cv.waitKey(0)
