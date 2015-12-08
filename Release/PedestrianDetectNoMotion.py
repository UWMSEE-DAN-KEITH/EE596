#EE596 Final Project
# Authors: Dan Sweet and Keith Mikoleit
# Date: December 8, 2015

#Pedestrian Detection Script, No Mtion
#    This script takes a movie (pre-recorded or from Pi) and attempts to find pedestrians
#    within each frame. It uses a pre-trained HOG model pedestrian classifier to do this.
#    It does not use any motion detection.
#
#    There is an optional test mode (MODE = 2) that uses the HAAR Cascade pre-trained 
#    model to look for things like faces, upper bodies, etc but we found performance of 
#    this to be pretty poor in most of our test data. 

# Import Packages
import argparse
import imutils
import time
#from picamera import PiCamera        #Comment this out if not Running on the Pi
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

MODE = 1                # 1= HOG PEDESTRIAN CLASSIFIER, 2=HAAR CASCADES CLASSIFIER
WINDOW_SIZE = 500

if MODE == 2:
    #Set this path to your local installation of OpenCV Haar Cascades Path
    HAAR_CASCADES_PATH = "A:\Programs\Python\Anaconda\Lib\site-packages\opencv\sources\data\haarcascades"

#Function that performs HOG Pedestrian search on one passed frame of the video
def Hog_Pedestrian(frame):
    
    #Copy Image
    orig = frame.copy()
    
    # Use HOG to detect people in image
    hog_start = time.time()
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
        padding=(8, 8), scale=1.2)
    hog_stop = time.time()
    
    # Draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Use non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    
    # Draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    #Add FPS
    cv2.putText(frame, "FPS: {}".format(fps), (frame.shape[1]-100, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # show the frame and record if the user presses a key
    cv2.imshow("No Max Suppression", orig)
    cv2.moveWindow("No Max Suppression", 0, 100)
    cv2.imshow("With Max Suppression", frame)
    cv2.moveWindow("With Max Suppression", WINDOW_SIZE+25, 100)
    
    #To track performance of HOG as we vary parameters
    print "HOG: " + str(hog_stop-hog_start)

# Function that performs Haar Cascade classifier. (NOTE This does not perform well on our test data, but 
# we left it as an option in the script for completeness). 
def Haar_Cascade(img):
    #Convert to Grayscale:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Detect faces using Haar multi-scale
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    print "Faces Detected: " + str(len(faces))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    #Add FPS
    cv2.putText(img, "FPS: {}".format(fps), (img.shape[1]-100, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow("HAAR", img)
    cv2.moveWindow("HAAR", WINDOW_SIZE+25, 100)
    
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=WINDOW_SIZE, help="minimum area size")
args = vars(ap.parse_args())
 
# if the video argument is None, then we are reading from Pi Camera
if args.get("video", None) is None:
    camera = PiCamera()
    camera.hflip = True
    camera.vflip = True
    camera.resolution = tuple([640,480])
    camera.framerate = 16
    time.sleep(0.25)
 
# Otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])
 
# Initialize the first frame in the video stream
firstFrame = None
avg = None
fps = 0

if MODE == 1:
    #HOG Descriptor for pedestrian detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
elif MODE == 2:
    #Select HAAR Face Descriptor
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADES_PATH + str('\haarcascade_frontalface_default.xml'))
#     face_cascade = cv2.CascadeClassifier(HAAR_CASCADES_PATH + str('\haarcascade_frontalface_alt2.xml'))
#     face_cascade = cv2.CascadeClassifier(HAAR_CASCADES_PATH + str('\haarcascade_upperbody.xml'))
#     face_cascade = cv2.CascadeClassifier(HAAR_CASCADES_PATH + str('\haarcascade_profileface.xml'))
#     face_cascade = cv2.CascadeClassifier(HAAR_CASCADES_PATH + str('\haarcascade_mcs_mouth.xml'))
#     face_cascade = cv2.CascadeClassifier(HAAR_CASCADES_PATH + str('\haarcascade_mcs_eyepair_small.xml'))

 
# MAIN LOOP: Repeat on all frames of the video
while True:
    #Get current time to track processing frame rate
    start_time = time.time()
    # Grab the current frame and initialize the occupied/unoccupied text
    (grabbed, frame) = camera.read()
    text = "Unoccupied"
 
    # If the frame could not be grabbed, then we have reached the end of the video
    if not grabbed:
        print "frame could not be grabbed"
        break
    
    # Resize the frame
    frame = imutils.resize(frame, width=WINDOW_SIZE)
    
    # Call specified model
    if MODE == 1:
        #Call the HOG Pedestrian Function
        Hog_Pedestrian(frame)
    elif MODE == 2:
        #Call the HAAR Cascade Function with whichever classifier was selected above
        Haar_Cascade(frame)
    
    #Get End Time and calculate FPS
    end_time=time.time()
    fps = str(int(1/(end_time-start_time)))
    
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
