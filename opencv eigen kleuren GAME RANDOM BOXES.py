'''Dit is een spel gemaakt met opencv. De bedoeling is om met je eigen kleur zo veel mogelijk punten te scoren.
Je scoort punten door met je gekleurde bal/voorwerp in het kader met dezelfde kleur te houden. Elke speler heeft zijn eigen kleur,
deze zijn rood, groen, blauw en geel.

SETUP:
Bij het opstarten van het spel moet er eerst een snelle calibratie gebeuren met de kleuren zodat ze aangepast worden aan de belichting in de kamer.
Dit doe je met volgende stappen:

1) Run de code
2) Open het venster met het webcambeeld
3) Houdt de kleur die je wil calibreren in het midden van het scherm en druk de bijhorende toets in op basis van de kleur tot er een kader rond te zien is.
    b = blauw
    r = rood
    g = groen
    y = geel

4) Schakel over naar het scherm met de game en laat je gaan! (De eerste box die je moet aantikken staat linksboven, daarna komen ze in de zone van de speler te staan)
5) Reset de scores naar nul met de toets 'n' (wanneer je dit wil)
6) Sluit het spel af met de toets 'q'

Dit spel is gemaakt als proof of concept voor het tracken van gekleurde ballen/voorwerpen om dan te implementeren in Squeezi Dance.

'''
from collections import deque
from imutils.video import VideoStream
import cv2
import numpy as np
import argparse
import imutils
import time
import random

Webcam = cv2.VideoCapture(0)

new_width = 1280
new_height = 960

box_size = 40

#HITBOXES SETUP
class ColoredBox:
    def __init__(self, color):
        self.color = color
        self.position = (0, 0)
        self.size = box_size  # Adjust the size as needed

    def move_to_random_location(self, max_x, max_y):
        #self.position = (random.randint(0, max_x - self.size), random.randint(0, max_y - self.size))
        if self.color == "green":
            self.position = (random.randint(0, max_x // 4 - self.size), random.randint(0, max_y - self.size))
        elif self.color == "red":
            self.position = (random.randint(max_x // 4, max_x // 2 - self.size), random.randint(0, max_y - self.size))
        elif self.color == "blue":
            self.position = (random.randint(max_x // 2, 3 * max_x // 4 - self.size), random.randint(0, max_y - self.size))
        elif self.color == "yellow":
            self.position = (random.randint(3 * max_x // 4, max_x - self.size), random.randint(0, max_y - self.size))

# Create colored boxes for each color
green_box = ColoredBox(color="green")
yellow_box = ColoredBox(color="yellow")
blue_box = ColoredBox(color="blue")
red_box = ColoredBox(color="red")

# Set the maximum coordinates for the boxes (adjust as needed)
max_x, max_y = new_width - box_size, new_height - box_size

#setup scores
green_score = 0
red_score = 0
blue_score = 0
yellow_score = 0


#SETUP TRACKING
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
#original
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
#GREEN 
ap.add_argument("-bg", "--bufferg", type=int, default=32,
	help="max buffer size")
#YELLOW
ap.add_argument("-by", "--buffery", type=int, default=32,
	help="max buffer size")
#BLUE
ap.add_argument("-bb", "--bufferb", type=int, default=32,
	help="max buffer size")
#RED
ap.add_argument("-br", "--bufferr", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
#GREEN
ptstrackgreen = deque(maxlen=args["bufferg"])
countertrackgreen = 0
(dXG, dYG) = (0, 0)
directiontrackgreen = ""

#YELLOW
ptstrackyellow = deque(maxlen=args["buffery"])
countertrackyellow = 0
(dXY, dYY) = (0, 0)
directiontrackyellow = ""

#BLUE
ptstrackblue = deque(maxlen=args["bufferb"])
countertrackblue = 0
(dXB, dYB) = (0, 0)
directiontrackblue = ""

#RED
ptstrackred = deque(maxlen=args["bufferr"])
countertrackred = 0
(dXR, dYR) = (0, 0)
directiontrackred = ""

# allow the camera or video file to warm up
time.sleep(2.0)

# Initialize HSV values Rood, Groen, Geel en Blauw
HR, SR, VR = 20, 0, 0
HG, SG, VG = 20, 0, 0
HY, SY, VY = 20, 0, 0
HB, SB, VB = 20, 0, 0

def on_key_press(event):
    global HR, SR, VR
    global HG, SG, VG
    global HY, SY, VY
    global HB, SB, VB

    #ROOD
    if event == ord('r'):
        # Capture a frame from the webcam
        
        _, imageFrame = Webcam.read()

        # Convert the frame to HSV
        hsv_frame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        # Select the center pixel HSV values
        center_pixel_hsv = hsv_frame[imageFrame.shape[0]//2, imageFrame.shape[1]//2]

        # Store HSV values
        HR, SR, VR = center_pixel_hsv[0], center_pixel_hsv[1], center_pixel_hsv[2]

        print(f"HSV values: HR={HR}, SR={SR}, VR={VR}")
    
    #GROEN
    if event == ord('g'):
        # Capture a frame from the webcam
        _, imageFrame = Webcam.read()

        # Convert the frame to HSV
        hsv_frame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        # Select the center pixel HSV values
        center_pixel_hsv = hsv_frame[imageFrame.shape[0]//2, imageFrame.shape[1]//2]

        # Store HSV values
        HG, SG, VG = center_pixel_hsv[0], center_pixel_hsv[1], center_pixel_hsv[2]

        print(f"HSV values: HG={HG}, SG={SG}, VG={VG}")

    #YELLOW
    if event == ord('y'):
        # Capture a frame from the webcam
        _, imageFrame = Webcam.read()

        # Convert the frame to HSV
        hsv_frame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        # Select the center pixel HSV values
        center_pixel_hsv = hsv_frame[imageFrame.shape[0]//2, imageFrame.shape[1]//2]

        # Store HSV values
        HY, SY, VY = center_pixel_hsv[0], center_pixel_hsv[1], center_pixel_hsv[2]

        print(f"HSV values: HY={HY}, SY={SY}, VY={VY}")

    #BLUE
    if event == ord('b'):
        # Capture a frame from the webcam
        _, imageFrame = Webcam.read()

        # Convert the frame to HSV
        hsv_frame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        # Select the center pixel HSV values
        center_pixel_hsv = hsv_frame[imageFrame.shape[0]//2, imageFrame.shape[1]//2]

        # Store HSV values
        HB, SB, VB = center_pixel_hsv[0], center_pixel_hsv[1], center_pixel_hsv[2]

        print(f"HSV values: HB={HB}, SB={SB}, VB={VB}")


        
        
while True: 

    _, imageFrame = Webcam.read()

    white_screen = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
    #GREY background
    cv2.rectangle(white_screen, (0, 0), (1280, 960), (230, 230, 230), -1)
    cv2.rectangle(white_screen, (0, 0), (1280, 40), (200, 200, 200), -1)

    imageFrame = cv2.flip(imageFrame, 1)

    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    #RED UPPER AND LOWER
    red_lower = np.array([HR, SR-50, VR-50], np.uint8) 
    red_upper = np.array([HR+15, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

    
    #GREEN UPPER AND LOWER
    green_lower = np.array([HG-15, SG-50, VG-40], np.uint8) 
    green_upper = np.array([HG+15, 255, 255], np.uint8) 
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    
    #BLUE UPPER AND LOWER
    blue_lower = np.array([HB-15, SB-50, VB-50], np.uint8) 
    blue_upper = np.array([HB+15, 255, 255], np.uint8) 
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    #YELLOW UPPER AND LOWER
    yellow_lower = np.array([HY-10, SY-50, VY-50], np.uint8) 
    yellow_upper = np.array([HY+20, 255, 255], np.uint8) 
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    #KERNEL (geen idee wat dit doet)
    kernel = np.ones((5,5), "uint8")


    #MASK RED
    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask = red_mask)

    #MASK GREEN
    green_mask = cv2.dilate(green_mask, kernel) 
    res_green = cv2.bitwise_and(imageFrame, imageFrame, mask = green_mask) 

    #MASK BLUE
    blue_mask = cv2.dilate(blue_mask, kernel) 
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask = blue_mask) 

    #MASK YELLOW 
    yellow_mask = cv2.dilate(yellow_mask, kernel) 
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame, mask = yellow_mask) 
    
    #CONTOUR RED 
    contours, hierarchy = cv2.findContours(red_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 100): 
            x, y, w, h = cv2.boundingRect(contour) 
            imageFrame = cv2.rectangle(imageFrame, (x, y),  
                                       (x + w, y + h),  
                                       (0, 0, 255), 2) 
              
            cv2.putText(imageFrame, "Red Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                        (0, 0, 255))     
  
    #CONTOUR GREEN 
    contours, hierarchy = cv2.findContours(green_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
      
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 100): 
            x, y, w, h = cv2.boundingRect(contour) 
            imageFrame = cv2.rectangle(imageFrame, (x, y),  
                                       (x + w, y + h), 
                                       (0, 255, 0), 2) 
              
            cv2.putText(imageFrame, "Green Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX,  
                        1.0, (0, 255, 0)) 
  
    #CONTOUR BLUE 
    contours, hierarchy = cv2.findContours(blue_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 100): 
            x, y, w, h = cv2.boundingRect(contour) 
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h), 
                                       (255, 0, 0), 2) 
              
            cv2.putText(imageFrame, "Blue Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 0, 0)) 
              
    #CONTOUR YELLOW 
    contours, hierarchy = cv2.findContours(yellow_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 100): 
            x, y, w, h = cv2.boundingRect(contour) 
            imageFrame = cv2.rectangle(imageFrame, (x, y),  
                                       (x + w, y + h),  
                                       (0, 255, 255), 2) 
              
            cv2.putText(imageFrame, "Yellow Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                        (0, 255, 255))   



    #TRACKING LOOP ALGEMEEN
	# handle the frame from VideoCapture or VideoStream
    TrackFrame = imageFrame
    TrackFrame = cv2.resize(TrackFrame, (new_width, new_height))
	# resize the frame, blur it, and convert it to the HSV
	# color space
    #TrackFrame = imutils.resize(TrackFrame, width=600)
    blurred = cv2.GaussianBlur(TrackFrame, (11, 11), 0)
    hsvTrackFrame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask


    #GREEN
    masktrackgreen = cv2.inRange(hsvTrackFrame, green_lower, green_upper)
    masktrackgreen = cv2.erode(masktrackgreen, None, iterations=2)
    masktrackgreen = cv2.dilate(masktrackgreen, None, iterations=2)
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
    cntstrackgreen = cv2.findContours(masktrackgreen.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cntstrackgreen = imutils.grab_contours(cntstrackgreen)
    centertrackgreen = None

    #YELLOW
    masktrackyellow = cv2.inRange(hsvTrackFrame, yellow_lower, yellow_upper)
    masktrackyellow = cv2.erode(masktrackyellow, None, iterations=2)
    masktrackyellow = cv2.dilate(masktrackyellow, None, iterations=2)
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
    cntstrackyellow = cv2.findContours(masktrackyellow.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cntstrackyellow = imutils.grab_contours(cntstrackyellow)
    centertrackyellow = None

    #BLUE
    masktrackblue = cv2.inRange(hsvTrackFrame, blue_lower, blue_upper)
    masktrackblue = cv2.erode(masktrackblue, None, iterations=2)
    masktrackblue = cv2.dilate(masktrackblue, None, iterations=2)
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
    cntstrackblue = cv2.findContours(masktrackblue.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cntstrackblue = imutils.grab_contours(cntstrackblue)
    centertrackblue = None

    #RED
    masktrackred = cv2.inRange(hsvTrackFrame, red_lower, red_upper)
    masktrackred = cv2.erode(masktrackred, None, iterations=2)
    masktrackred = cv2.dilate(masktrackred, None, iterations=2)
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
    cntstrackred = cv2.findContours(masktrackred.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cntstrackred = imutils.grab_contours(cntstrackred)
    centertrackred = None
    
    
    # only proceed if at least one contour was found
    #GREEN
    if len(cntstrackgreen) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        ctrackgreen = max(cntstrackgreen, key=cv2.contourArea)
        ((xtrackgreen, ytrackgreen), radiustrackgreen) = cv2.minEnclosingCircle(ctrackgreen)
        Mtrackgreen = cv2.moments(ctrackgreen)
        centertrackgreen = (int(Mtrackgreen["m10"] / Mtrackgreen["m00"]), int(Mtrackgreen["m01"] / Mtrackgreen["m00"]))
		# only proceed if the radius meets a minimum size
        if radiustrackgreen > 2:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
            cv2.circle(white_screen, (int(xtrackgreen), int(ytrackgreen)), int(radiustrackgreen),
				(0, 255, 0), 2)
            cv2.circle(white_screen, centertrackgreen, 5, (0, 255, 0), -1)
            ptstrackgreen.appendleft(centertrackgreen)
    #YELLOW
    if len(cntstrackyellow) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        ctrackyellow = max(cntstrackyellow, key=cv2.contourArea)
        ((xtrackyellow, ytrackyellow), radiustrackyellow) = cv2.minEnclosingCircle(ctrackyellow)
        Mtrackyellow = cv2.moments(ctrackyellow)
        centertrackyellow = (int(Mtrackyellow["m10"] / Mtrackyellow["m00"]), int(Mtrackyellow["m01"] / Mtrackyellow["m00"]))
		# only proceed if the radius meets a minimum size
        if radiustrackyellow > 2:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
            cv2.circle(white_screen, (int(xtrackyellow), int(ytrackyellow)), int(radiustrackyellow),
				(0, 255, 255), 2)
            cv2.circle(white_screen, centertrackyellow, 5, (0, 255, 255), -1)
            ptstrackyellow.appendleft(centertrackyellow)
    
    #BLUE
    if len(cntstrackblue) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        ctrackblue = max(cntstrackblue, key=cv2.contourArea)
        ((xtrackblue, ytrackblue), radiustrackblue) = cv2.minEnclosingCircle(ctrackblue)
        Mtrackblue = cv2.moments(ctrackblue)
        centertrackblue = (int(Mtrackblue["m10"] / Mtrackblue["m00"]), int(Mtrackblue["m01"] / Mtrackblue["m00"]))
		# only proceed if the radius meets a minimum size
        if radiustrackblue > 2:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
            cv2.circle(white_screen, (int(xtrackblue), int(ytrackblue)), int(radiustrackblue),
				(255, 0, 0), 2)
            cv2.circle(white_screen, centertrackblue, 5, (255, 0, 0), -1)
            ptstrackblue.appendleft(centertrackblue)
    
    #RED
    if len(cntstrackred) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        ctrackred = max(cntstrackred, key=cv2.contourArea)
        ((xtrackred, ytrackred), radiustrackred) = cv2.minEnclosingCircle(ctrackred)
        Mtrackred = cv2.moments(ctrackred)
        centertrackred = (int(Mtrackred["m10"] / Mtrackred["m00"]), int(Mtrackred["m01"] / Mtrackred["m00"]))
		# only proceed if the radius meets a minimum size
        if radiustrackred > 2:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
            cv2.circle(white_screen, (int(xtrackred), int(ytrackred)), int(radiustrackred),
				(0, 0, 255), 2)
            cv2.circle(white_screen, centertrackred, 5, (0, 0, 255), -1)
            ptstrackred.appendleft(centertrackred)

    #KEEPTRACKING WHEN OUT OF VIEW?
    # GREEN
    if centertrackgreen is not None:
        ptstrackgreen.appendleft(centertrackgreen)
        # Limit the length of ptstrackgreen
        if len(ptstrackgreen) > args["bufferg"]:
            ptstrackgreen.pop()
    else:
        # Update with the last known location when the ball is not detected
        last_known_green = ptstrackgreen[0] if ptstrackgreen else (0, 0)
        ptstrackgreen.appendleft(last_known_green)
        # Limit the length of ptstrackgreen
        if len(ptstrackgreen) > args["bufferg"]:
            ptstrackgreen.pop()

    # YELLOW
    if centertrackyellow is not None:
        ptstrackyellow.appendleft(centertrackyellow)
        # Limit the length of ptstrackyellow
        if len(ptstrackyellow) > args["buffery"]:
            ptstrackyellow.pop()
    else:
        # Update with the last known location when the ball is not detected
        last_known_yellow = ptstrackyellow[0] if ptstrackyellow else (0, 0)
        ptstrackyellow.appendleft(last_known_yellow)
        # Limit the length of ptstrackyellow
        if len(ptstrackyellow) > args["buffery"]:
            ptstrackyellow.pop()

    # BLUE
    if centertrackblue is not None:
        ptstrackblue.appendleft(centertrackblue)
        # Limit the length of ptstrackblue
        if len(ptstrackblue) > args["bufferb"]:
            ptstrackblue.pop()
    else:
        # Update with the last known location when the ball is not detected
        last_known_blue = ptstrackblue[0] if ptstrackblue else (0, 0)
        ptstrackblue.appendleft(last_known_blue)
        # Limit the length of ptstrackblue
        if len(ptstrackblue) > args["bufferb"]:
            ptstrackblue.pop()

    # RED
    if centertrackred is not None:
        ptstrackred.appendleft(centertrackred)
        # Limit the length of ptstrackred
        if len(ptstrackred) > args["bufferr"]:
            ptstrackred.pop()
    else:
        # Update with the last known location when the ball is not detected
        last_known_red = ptstrackred[0] if ptstrackred else (0, 0)
        ptstrackred.appendleft(last_known_red)
        # Limit the length of ptstrackred
        if len(ptstrackred) > args["bufferr"]:
            ptstrackred.pop()


    # loop over the set of tracked points 
    #GREEN
    for i in np.arange(1, len(ptstrackgreen)):
		# if either of the tracked points are None, ignore
		# them
        if ptstrackgreen[i - 1] is None or ptstrackgreen[i] is None:
            continue
		# check to see if enough points have been accumulated in
		# the buffer
        if countertrackgreen >= 10 and i == 1 and len(ptstrackgreen) >= 10 and ptstrackgreen[-10] is not None:
			# compute the difference between the x and y
			# coordinates and re-initialize the direction
			# text variables
            dXG = ptstrackgreen[-10][0] - ptstrackgreen[i][0]
            dYG = ptstrackgreen[-10][1] - ptstrackgreen[i][1]
            (dirXG, dirYG) = ("", "")

        #GREEN
        thicknesstrackgreen = int(np.sqrt(args["bufferg"] / float(i + 1)) * 2.5)
        cv2.line(white_screen, ptstrackgreen[i - 1], ptstrackgreen[i], (0, 255, 0), thicknesstrackgreen)
        
    #YELLOW
    for l in np.arange(1, len(ptstrackyellow)):
		# if either of the tracked points are None, ignore
		# them
        if ptstrackyellow[l - 1] is None or ptstrackyellow[l] is None:
            continue
		# check to see if enough points have been accumulated in
		# the buffer
        if countertrackyellow >= 10 and l == 1 and len(ptstrackyellow) >= 10 and ptstrackyellow[-10] is not None:
			# compute the difference between the x and y
			# coordinates and re-initialize the direction
			# text variables
            dXY = ptstrackyellow[-10][0] - ptstrackyellow[l][0]
            dYY = ptstrackyellow[-10][1] - ptstrackyellow[l][1]
            (dirXY, dirYY) = ("", "")
        
        #YELLOW
        thicknesstrackyellow = int(np.sqrt(args["buffery"] / float(l + 1)) * 2.5)
        cv2.line(white_screen, ptstrackyellow[l - 1], ptstrackyellow[l], (0, 255, 255), thicknesstrackyellow)

    #BLUE
    for j in np.arange(1, len(ptstrackblue)):
		# if either of the tracked points are None, ignore
		# them
        if ptstrackblue[j - 1] is None or ptstrackblue[j] is None:
            continue
		# check to see if enough points have been accumulated in
		# the buffer
        if countertrackblue >= 10 and j == 1 and len(ptstrackblue) >= 10 and ptstrackblue[-10] is not None:
			# compute the difference between the x and y
			# coordinates and re-initialize the direction
			# text variables
            dXB = ptstrackblue[-10][0] - ptstrackblue[j][0]
            dYB = ptstrackblue[-10][1] - ptstrackblue[j][1]
            (dirXB, dirYB) = ("", "")
        
        #BLUE
        thicknesstrackblue = int(np.sqrt(args["buffery"] / float(j + 1)) * 2.5)
        cv2.line(white_screen, ptstrackblue[j - 1], ptstrackblue[j], (255, 0, 0), thicknesstrackblue)

    #RED
    for k in np.arange(1, len(ptstrackred)):
		# if either of the tracked points are None, ignore
		# them
        if ptstrackred[k - 1] is None or ptstrackred[k] is None:
            continue
		# check to see if enough points have been accumulated in
		# the buffer
        if countertrackred >= 10 and k == 1 and len(ptstrackred) >= 10 and ptstrackred[-10] is not None:
			# compute the difference between the x and y
			# coordinates and re-initialize the direction
			# text variables
            dXR = ptstrackred[-10][0] - ptstrackred[k][0]
            dYR = ptstrackred[-10][1] - ptstrackred[k][1]
            (dirXR, dirYR) = ("", "")
        
        #RED
        thicknesstrackred = int(np.sqrt(args["buffery"] / float(k + 1)) * 2.5)
        cv2.line(white_screen, ptstrackred[k - 1], ptstrackred[k], (0, 0, 255), thicknesstrackred)


	# show the movement deltas and the direction of movement on
	# the frame
    #cv2.putText(TrackFrame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
    #GREEN
    cv2.putText(TrackFrame, "dxG: {}, dyG: {}".format(dXG, dYG), (10, TrackFrame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    
    #YELLOW
    cv2.putText(TrackFrame, "dxY: {}, dyY: {}".format(dXY, dYY), (10, TrackFrame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    
    #BLUE
    cv2.putText(TrackFrame, "dxB: {}, dyB: {}".format(dXB, dYB), (10, TrackFrame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

    #RED
    cv2.putText(TrackFrame, "dxR: {}, dyR: {}".format(dXR, dYR), (10, TrackFrame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    #HITBOXES

    # Draw boxes on the tracking frame
    cv2.rectangle(white_screen, green_box.position, (green_box.position[0] + green_box.size, green_box.position[1] + green_box.size), (0, 255, 0), 2)
    cv2.rectangle(white_screen, yellow_box.position, (yellow_box.position[0] + yellow_box.size, yellow_box.position[1] + yellow_box.size), (0, 255, 255), 2)
    cv2.rectangle(white_screen, blue_box.position, (blue_box.position[0] + blue_box.size, blue_box.position[1] + blue_box.size), (255, 0, 0), 2)
    cv2.rectangle(white_screen, red_box.position, (red_box.position[0] + red_box.size, red_box.position[1] + red_box.size), (0, 0, 255), 2)

    # Check if colored balls are inside the boxes
    #GREEN
    if centertrackgreen is not None and (
        green_box.position[0] < centertrackgreen[0] < green_box.position[0] + green_box.size and green_box.position[1] < centertrackgreen[1] < green_box.position[1] + green_box.size):
        print("Green ball is inside the green box")
        # Green ball is inside the green box
        # Increase the score for the green ball and move the box
        green_score += 1
        green_box.move_to_random_location(max_x, max_y)
        print("Green box moved to:", green_box.position)

    #RED
    if centertrackred is not None and (
        red_box.position[0] < centertrackred[0] < red_box.position[0] + red_box.size and red_box.position[1] < centertrackred[1] < red_box.position[1] + red_box.size):
        print("Red ball is inside the Red box")
        # Red ball is inside the green box
        # Increase the score for the red ball and move the box
        red_score += 1
        red_box.move_to_random_location(max_x, max_y)
        print("Red box moved to:", red_box.position)

    #BLUE
    if centertrackblue is not None and (
        blue_box.position[0] < centertrackblue[0] < blue_box.position[0] + blue_box.size and blue_box.position[1] < centertrackblue[1] < blue_box.position[1] + blue_box.size):
        print("Blue ball is inside the blue box")
        # Blue ball is inside the blue box
        # Increase the score for the blue ball and move the box
        blue_score += 1
        blue_box.move_to_random_location(max_x, max_y)
        print("Blue box moved to:", blue_box.position)

    #YELLOW
    if centertrackyellow is not None and (
        yellow_box.position[0] < centertrackyellow[0] < yellow_box.position[0] + yellow_box.size and yellow_box.position[1] < centertrackyellow[1] < yellow_box.position[1] + yellow_box.size):
        print("Yellow ball is inside the yellow box")
        # Yellow ball is inside the yellow box
        # Increase the score for the yellow ball and move the box
        yellow_score += 1
        yellow_box.move_to_random_location(max_x, max_y)
        print("Yellow box moved to:", yellow_box.position)

    # Repeat the same logic for other colored balls and boxes
    

    # Display the score on the frame
        #GREEN
    cv2.putText(white_screen, f"Green Score: {green_score}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        #RED
    cv2.putText(white_screen, f"Red Score: {red_score}", (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #BLUE
    cv2.putText(white_screen, f"Blue Score: {blue_score}", (650, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #YELLOW
    cv2.putText(white_screen, f"Yellow Score: {yellow_score}", (970, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    

	# show the frame to our screen and increment the frame counter
    
    cv2.imshow("Tracking screen", TrackFrame)
    cv2.imshow('White Screen', white_screen)
    key = cv2.waitKey(1) & 0xFF
    countertrackgreen += 1
    countertrackyellow += 1
    countertrackblue += 1
    countertrackred += 1

    
    if cv2.waitKey(10) & 0xFF == ord('n'): 
        green_score = 0
        red_score = 0
        blue_score = 0
        yellow_score = 0

    # Program Termination 
    
    #cv2.imshow("Setup screen", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        Webcam.release() 
        cv2.destroyAllWindows() 
        break
    key = cv2.waitKey(10) & 0xFF
    on_key_press(key)

    
