import cv2
import numpy as np
import face_recognition
import os


new_face = input("do you want to add new face?(y/n)")
if new_face.lower()=="y":
    image_name = input("please enter the image name: ")
    image_name = image_name+".jpg"

    cam = cv2.VideoCapture(0)
    result, image = cam.read()

    folder_name = "training"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    full_path = os.path.join(folder_name, image_name)
    if result:
        cv2.imwrite(full_path,image)


    cam.release()
    cv2.destroyAllWindows()


path = 'training'
 
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        # when using face_recognition function the image should be in RGB format.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)



cap  = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    # below code  resizes the captured frame to a quarter of its original size and converts it from BGR to RGB format.
    #  This smaller, RGB image will be used for face detection and recognition, improving processing speed
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # below code detects face locations in the resized and converted frame using the 'face_recognition.face_locations()' function.
    #  This function returns a list of tuples containing coordinates (top, right, bottom, left) of detected faces in the image.
    faces_in_frame = face_recognition.face_locations(imgS)

    # It encodes the faces found in the frame using the 'face_recognition.face_encodings()' function. 
    # This function extracts facial features and computes a numerical encoding for each face detected.
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)


    for encode_face, faceloc in zip(encoded_faces,faces_in_frame):

        matches = face_recognition.compare_faces(encoded_face_train, encode_face)

        # If the distance is small, it indicates that the faces are similar.
        # If the distance is large, it indicates that the faces are dissimilar.
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)

        # 'np.argmin(faceDist)' is a NumPy function used to find the index of the minimum value in the array faceDist.
        matchIndex = np.argmin(faceDist)

        print(matchIndex)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            y1,x2,y2,x1 = faceloc

            #back to the original size. This line scales the coordinates by multiplying them by 4.
            # y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            y1, x2,y2,x1 =[coord * 4 for coord in faceloc]
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-5),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        else:
            name = "Unknown"
            y1,x2,y2,x1 = faceloc

            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            # for image
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            # for showing name and cv2.filled used to fill the box with same green color
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-5),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break