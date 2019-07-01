import face_recognition
import os
import sys
import pickle
from PIL import Image
from tqdm import tqdm

folder_path = sys.argv[1]
print('Extracting faces from files in', folder_path)
included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']
# find all files with the above extensions in the given directory
image_files_path = [fn for fn in os.listdir(folder_path) if any(fn.endswith(ext) for ext in included_extensions)]
faces = list()
for image_file_path in tqdm(image_files_path):
    image = face_recognition.load_image_file(os.path.join(folder_path, image_file_path))
    # find all face locations in this image. model 'hog' or 'cnn' can be used
    locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model='hog')
    for location in locations:
        # crop face out of image and add it and it's original image to the faces list
        faces.append((image[location[0]:location[2], location[3]:location[1]], image_file_path))
print('Generating encodings')
encodings = list()
for face in tqdm(faces):
    # extract encodings of faces
    encoding = face_recognition.face_encodings(face[0])
    # sometimes no encodings are produced - this will check for this
    if len(encoding) == 0:
        continue
    encodings.append((encoding[0], ) + face)
print('Sorting encodings')
groups = list()
for encoding in tqdm(encodings):
    matched = False
    for group in groups:
        # get distances of the face from each encoding in the group
        distances = face_recognition.face_distance([item[0] for item in group], encoding[0])
        # see if half or more are less than a given value as tolerance threshold
        if sum(1 for distance in distances if distance < 0.35) >= len(distances)//2:
            group.append(encoding)
            matched = True
    if not matched:
        # if it does not fit into a group, create its own group
        groups.append([encoding])
print('Saving')
if not os.path.exists('output'):
    os.makedirs('output')
os.chdir('output')
for i in range(len(groups)):
    os.makedirs(str(i))
    for j in range(len(groups[i])):
        Image.fromarray(groups[i][j][1]).save(os.path.join(str(i), str(j)+'.jpg'))
print('Done!')