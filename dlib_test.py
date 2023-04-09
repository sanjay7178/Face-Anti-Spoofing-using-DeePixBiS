import dlib
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from Model import DeePixBiS
from Loss import PixWiseBCELoss
from Metrics import predict, test_accuracy, test_loss
import os

# Helper function to get the bouding box coordinates from dlib rectangle

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

# Load model and face detector
model = DeePixBiS()
model.load_state_dict(torch.load("./DeePixBiS.pth"))
model.eval()

detector = dlib.get_frontal_face_detector()

# Define image processing pipeline
tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Input and Output Directories
input_dir = "./data/images/"
output_dir = "./data/output_images/"

# Loop over the images in input directory
count = 0
for filename in os.listdir(input_dir):

  # count == 10, IS TO CHECK IF THE CODE IS RUNNING PROPERLY
  # REMOVE IF NOT NEEDED
  if count == 10:
      break
  # ---
  
  img = cv2.imread(os.path.join(input_dir, filename))
  faces = detector(img)
  boxes = [convert_and_trim_bb(img, r) for r in faces]

  for (x1, y1, x2, y2) in boxes:

    faceRegion = img[y1:y2, x1:x2]

    # Apply image processing pipeline
    faceRegion = tfms(faceRegion)
    faceRegion = faceRegion.unsqueeze(0)

    # Run Classification Model
    mask, binary = model.forward(faceRegion)
    res = torch.mean(mask).item()

    # Bounding box and Label on image
    cv2.rectangle(img, (x1, y1), (x1 + x2, y1 + y2), (0, 0, 255), 2)
    if res < 0.5:
      cv2.putText(img, 'Fake', (x1, y1 + y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
      cv2.putText(img, 'Real', (x1, y1 + y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_dir, filename), img)

  # IF REMOVING THIS, REMOVE THE LINES AT THE START of THE FUNCTION
  count += 1
  #---