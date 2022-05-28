# =====================================================================
#                      Problem 2: Straight Line Detection
# =====================================================================

# Import Libraries

import numpy as np
import cv2
import matplotlib.pyplot as plt


# =====================================================================
#                           Helper Functions
# =====================================================================

# Define Region of Interest 
def regionOfInterest(img):
    VerticesofInterest = [(100,540), (920, 540), (500, 310), (460, 310)] # coordinates passed as a tuple
    mask = np.zeros_like(img)
    matchMaskColor = (255,255,255)
    cv2.fillPoly(mask, np.array([VerticesofInterest], np.int32), matchMaskColor)
    maskedImg = cv2.bitwise_and(img, mask)
    return maskedImg

def getCanny(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(grey,(5,5), 0)
    CannyImg = cv2.Canny(blur_img, 250, 350)
    return(CannyImg)

def getHough(og_frame, edges):
    lines = cv2.HoughLinesP(edges,rho=1,threshold=20,theta=np.pi/180,
                            minLineLength=2,maxLineGap=300,lines=np.array([]))
    blank = np.zeros((og_frame.shape[0],og_frame.shape[1],3),dtype=np.uint8)
    
    # lines --> the output array of Probabilistic Hough Transform
    # find slope of longest line 
    for x1,y1,x2,y2 in lines[0]:
        LineSlope = (y2 - y1)/(x2 - x1)
    
    # compare slope and designate line colour accordingly
    for line in lines:
        x1,y1,x2,y2 = line[0]
        slope = (y2 - y1)/(x2 - x1)
        if (slope*LineSlope) > 0:
            cv2.line(blank,(x1,y1),(x2,y2),(0,255,0),4)
        else:
            cv2.line(blank,(x1,y1),(x2,y2),(0,0,255),4)
        
           
    final_img = cv2.addWeighted(og_frame,0.8,blank,1,0.0)
    return final_img

# =====================================================================
#                           Main Function
# =====================================================================

if __name__== "__main__":
    # read video
    cap = cv2.VideoCapture('Q2_StraightLine.mp4')
    # result = cv2.VideoWriter('Q2_solution.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, (960,540))
    
    if (cap.isOpened() == False):
        print("Error opening video file")
    
    while(cap.isOpened()):
        ret , frame = cap.read()
        # frame = cv2.flip(image, -1)
        
        if ret == True:
            CannyImg = getCanny(frame)  # get edges
            # kernel = np.ones((3,3),np.uint8) # to dilate the images
            # CannyImg = cv2.dilate(CannyImg, kernel, iterations=2)
            maskedImg = regionOfInterest(CannyImg)  # define Area of Interest
            Lanes = np.copy(frame)
            hough = getHough(Lanes, maskedImg) # perform hough transform
            
            cv2.imshow(' Video output ', hough)
            
            if cv2.waitKey(45) & 0xFF == ord('q'):
                break
            
        else:
            break
        
    cap.release()
    # result.release() # Saving the Video
    cv2.destroyAllWindows()









