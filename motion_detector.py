# import the necessary packages
from __future__ import print_function
import argparse
import datetime
import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths
import time
#from picamera import PiCamera
import numpy as np
import cv2

MODE = 2                # 1= Absolute Difference, 2=Weighted Average
WINDOW_SIZE = 500

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=WINDOW_SIZE, help="minimum area size")
args = vars(ap.parse_args())
 
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = PiCamera()
    camera.hflip = True
    camera.vflip = True
    camera.resolution = tuple([640,480])
    camera.framerate = 16
    time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])
 
# initialize the first frame in the video stream
firstFrame = None
avg = None
fps = 0

# setup kernal for difference frame countour opening
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# initialize people tracking variables
num_people = 0
motion_boxes_current = 0
motion_boxes_previous = 0

# loop over the frames of the video
while True:
    #Get current time to track processing frame rate
    start_time = time.time()
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    text = "Unoccupied"
 
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        print('frame could not be grabbed')
        break
    
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=WINDOW_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
	
    # if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...") 
        avg = gray.copy().astype("float")
        #rawCapture.truncate(0)
        continue
	
    #Absolute Difference with First Frame
    if MODE == 1:
        if firstFrame is None:
            firstFrame = gray
            continue
        
        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
    #Weighted Average difference between current frame and previous frames
    if MODE == 2:
        # only update the background if no people have entered the frame
        if(num_people == motion_boxes_current):
            # accumulate the weighted average between the current frame and previous frames
            cv2.accumulateWeighted(gray, avg, 0.5)
        # compute the difference between the current frame and running average
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
	   
    # dilate the thresholded image to fill in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    
    # set the number of motion boxes we fond
    motion_boxes_current = len(cnts)
	
	# if we found motion, run HOG to see if the motion includes a person
    if motion_boxes_current > 0:
        # if we have new motion check for people
        if(motion_boxes_current != motion_boxes_previous):
            # update number of motion boxes
            motion_boxes_previous = motion_boxes_current
            
            # detect people in the image
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.10)
            
            # apply non-maxima suppression to the bounding boxes using a
            # fairly large overlap threshold to try to maintain overlapping
            # boxes that are still people
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            
            # check to see if we have found people
            if len(pick) > 0:
                num_people = len(pick)
                text = "Occupied by " + str(num_people) + " people"
                # draw the final bounding boxes
                for (xA, yA, xB, yB) in pick:
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            else:
                num_people = 0
    
        # if there are no new motion boxes
        else:
            # draw motion boxes if we have previously found their to be people
    
            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < args["min_area"]:
                    continue
        
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            text = "Occupied by " + str(num_people) + " people"
	
    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(frame, "FPS: {}".format(fps), (frame.shape[1]-100, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.moveWindow("Security Feed", 0, 100)
    cv2.imshow("Thresh", thresh)
    cv2.moveWindow("Thresh", WINDOW_SIZE+25, 100)
    cv2.imshow("Frame Delta", frameDelta)
    cv2.moveWindow("Frame Delta", 2*WINDOW_SIZE+50, 100)
    
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
