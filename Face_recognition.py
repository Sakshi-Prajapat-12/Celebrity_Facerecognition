import cv2 as cv
import face_recognition as fr
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
from concurrent.futures import ThreadPoolExecutor

# Create a Tkinter window and show a file dialog to select an image
root = Tk()
root.withdraw()
load_image = askopenfilename()

# Load the selected image
try:
    target_image = fr.load_image_file(load_image)
except Exception as e:
    print("Error loading the image:", e)
    exit()

# Load dataset images and encode faces
def encode_faces(folder):
    list_people_encoding = []
    for filename in os.listdir(folder):
        try:
            known_image = fr.load_image_file(os.path.join(folder, filename))
            known_encoding = fr.face_encodings(known_image)[0]  # Assuming only one face in each image
            list_people_encoding.append((known_encoding, filename))
        except Exception as e:
            print(f"Error encoding faces in {filename}: {e}")
    return list_people_encoding

dataset_encodings = encode_faces('project/')

# Find faces in the target image
def find_target_faces():
    face_locations = fr.face_locations(target_image, number_of_times_to_upsample=1, model='hog')
    match_found = False

    for location in face_locations:
        top, right, bottom, left = location
        target_encoding = fr.face_encodings(target_image, [location])[0]

        for known_encoding, filename in dataset_encodings:
            try:
                is_target_face = fr.compare_faces([known_encoding], target_encoding, tolerance=0.55)
                if is_target_face[0]:
                    match_found = True
                    cv.rectangle(target_image, (left, top), (right, bottom), (255, 0, 0), 2)
                    cv.rectangle(target_image, (left, bottom + 20), (right, bottom), (255, 0, 0), cv.FILLED)
                    cv.putText(target_image, filename, (left + 3, bottom + 14), cv.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
            except Exception as e:
                print(f"Error comparing faces with {filename}: {e}")

    if not match_found:
        print("No match found in the dataset")

# Render the image with faces highlighted
def render_image():
    rgb_img = cv.cvtColor(target_image, cv.COLOR_BGR2RGB)
    cv.imshow('face detect', rgb_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Execute the face recognition process
find_target_faces()
render_image()
