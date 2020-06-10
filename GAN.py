# Extract and resize the faces into a new dataset
from os import listdir
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

# load an image as an RGB numpy array
def load_image(filename):
  # load image from the file
  image = Image.open(filename)
  # convert to RGB if needed
  image = image.convert('RGB')
  # conver to array
  pixels = np.asarray(image)
  return pixels

# Extract the face from the loaded image and resize
def extract_face(model,pixels, required_size=(80,80)):
  # detect face in the image
  faces = model.detect_faces(pixels)
  # skip cases where we could not detect a face
  if len(faces) == 0:
    return None
  # extract details of the face
  x1,y1,width,height = faces[0]['box']
  # make sure the cooridinates are positive
  x1,y1 = abs(x1),abs(y1)
  # obtain second set of coordinates for convinience
  x2,y2 = x1+width,y1+height
  # retrive face pixels
  face_pixels = pixels[y1:y2,x1:x2]
  # resize the pixels to the model size
  image = Image.fromarray(face_pixels)
  image = image.resize(required_size)
  face_array = np.asarray(image)
  return face_array

# load images and extract faces for all images in a directory
def load_faces(directory,n_faces):
  model = MTCNN()
  faces = []
  # enumerate files
  for filename in listdir(directory):
    #load the image
    pixels = load_image(directory + filename)
    # get face
    face = extract_face(model,pixels)
    if face is None:
      continue
    # store
    faces.append(face)
    print(len(faces), face.shape)
    # stop once we have enough
    if len(faces) >= n_faces:
      break
  return np.asarray(faces)

# directory that contains all the images
directory = 'img_align_celeba/'
# load and extract all faces
all_faces = load_faces(directory,50000)
print('Loaded:',all_faces.shape)
# save in compressed format
np.savez_compressed('img_align_celeba.npz', all_faces)