import picamera

with picamera.PiCamera() as camera:
    camera.hflip = True
    camera.vflip = True
    camera.resolution = (640, 480)
    camera.start_recording('my_video.h264')
    camera.wait_recording(15)
    camera.stop_recording()
