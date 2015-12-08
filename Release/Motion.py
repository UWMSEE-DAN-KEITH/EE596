#EE596 Final Project
# Authors: Dan Sweet and Keith Mikoleit
# Date: December 8, 2015

#Motion Detection Test Script
#    This script takes a video (pre-recorded or from Pi) and tracks the motion within
#    each frame of the video. There are two modes that are supported:
#
#    MODE=1: Motion is tracked using an absolute difference between the first frame of
#    the video and the current frame. This runs efficiently but is overly sensitive and
#    tracks pretty much ALL motion, including small shadows.
#
#    MODE=2: Motion is tracked using a weighted average difference between the current
#    frame and the previous frame. This makes the motion detection less sensitive to 
#    insignificant motion (eg small shadows) but still track major motion.

# Import Packages
import argparse
import datetime
import imutils
import time
#from picamera import PiCamera        #Comment this out if not running on Pi
import cv2

MODE = 2                # 1= Absolute Difference, 2=Weighted Average
WINDOW_SIZE = 500

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=WINDOW_SIZE, help="minimum area size")
args = vars(ap.parse_args())
 
# If the video argument is None, then we are reading from Pi Camera
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

# Create Kernel shape used to dilate found contours
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# MAIN LOOP: Repeat for all frames of video
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
	
    # Resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=WINDOW_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
	
    # If the average frame is None, initialize it
    if avg is None:
        print "[INFO] starting background model..."
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
        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
	   
    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours on dilated image
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
	
	# For each contour, add a bounding box
    for c in cnts:
        # If the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
	
        # Compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
	
    # Draw the text, timestamp and FPS on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(frame, "FPS: {}".format(fps), (frame.shape[1]-100, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
    # Show the results
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
	
    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
