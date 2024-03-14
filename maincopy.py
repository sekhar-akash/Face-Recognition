import cv2
import numpy as np
import face_recognition
import os

class FaceRecognitionSystem:
    def __init__(self):
        self.classNames = []
        self.encoded_face_train = []
        new_face = input("Do you want to add a new face? (y/n): ")
        if new_face.lower() == "y":
            image_name = input("Please enter the image name: ") + ".jpg"
            cam = cv2.VideoCapture(0)
            result, image = cam.read()
            folder_name = "training"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            full_path = os.path.join(folder_name, image_name)
            if result:
                cv2.imwrite(full_path, image)
            cam.release()
            cv2.destroyAllWindows()

    def train(self, folder_path):
        mylist = os.listdir(folder_path)
        for cl in mylist:
            curImg = cv2.imread(f'{folder_path}/{cl}')
            self.classNames.append(os.path.splitext(cl)[0])

            curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
            encoded_face = face_recognition.face_encodings(curImg)[0]
            self.encoded_face_train.append(encoded_face)

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
            imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            faces_in_frame = face_recognition.face_locations(imgS)
            encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

            for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
                matches = face_recognition.compare_faces(self.encoded_face_train, encode_face)
                faceDist = face_recognition.face_distance(self.encoded_face_train, encode_face)
                matchIndex = np.argmin(faceDist)

                if matches[matchIndex]:
                    name = self.classNames[matchIndex].upper().lower()
                else:
                    name = "Unknown"

                y1, x2, y2, x1 = [coord * 4 for coord in faceloc]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('webcam', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



face_recognition_system = FaceRecognitionSystem()
face_recognition_system.train("training")
face_recognition_system.run()