from cv2 import rectangle
from PIL import Image, ImageTk
from tkinter import Tk, Button, Label, Entry
import cv2
import numpy as np
import PIL
import threading
from mtcnn.mtcnn import MTCNN
import vggutils
import glob

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

detector = MTCNN()
register = False
cropped_frame = 0
frame = 0
train_set_dir = "/home/schevala/dl/wip/keras/face_recognition/vggface/trainset/"
image_data = []

for img in glob.glob(train_set_dir + "*.jpg"):
    n = cv2.imread(img)
    image_data.append(n)


def runFaceThread(name):
    if (e1.get() == ""):
        print("Enter the label to register face")
        return
    resized = cv2.resize(cropped_frame, (224, 224),
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(train_set_dir + e1.get() + ".jpg", resized)


def registerFace():
    # threading.Thread(target=runFaceThread).start()
    runFaceThread()


def DetectFace():
    for img in image_data:
        status = vggutils.verifyFace(img, cropped_frame)


root = Tk()

width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

btn = Button(root, text="Register face!", command=registerFace)
#btn.grid(row=1, column=0, sticky=W, pady=4)
btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

btn1 = Button(root, text="Detect face!", command=DetectFace)
#btn.grid(row=1, column=0, sticky=W, pady=4)
btn1.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

name = Label(root, text="Enter Name")
name.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

e1 = Entry(root)
# 1.grid(row=0, column=1)
e1.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

# predict_btn = Button(root, text="Predict face!", command=predictFace)
# #btn.grid(row=1, column=0, sticky=W, pady=4)
# predict_btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

root.bind('<Escape>', lambda e: root.quit())
lmain = Label(root)


def UseMTCNN(frame):
    global detector
    faces = detector.detect_faces(frame)
    for result in faces:
        # get coordinates
        x, y, width, height = result['box']
        x2, y2 = x + width, y + height
        # create the shape
        rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 1)
        return frame[y:y2, x:x2]


def show_frame():
    global cropped_frame, frame, image_data
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cropped_frame = UseMTCNN(frame)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


# show_frame()
lmain.pack()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
