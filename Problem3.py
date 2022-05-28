# =====================================================================
#                      Problem 3: Predict Turns
# =====================================================================

# Import Libraries

import numpy as np
import cv2
import matplotlib.pyplot as plt

# =====================================================================
#                           Helper Functions
# =====================================================================

def masks(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    # yellow mask
    yLow = np.array([0, 100, 30], dtype='uint8')
    yUpp = np.array([50, 255, 255], dtype='uint8')
    yellowMask = cv2.inRange(hls, yLow, yUpp)
    yellowLine = cv2.bitwise_and(hls, hls, mask = yellowMask).astype(np.uint8)

    # white mask
    wLow = np.array([50, 50, 50], dtype='uint8')
    wUpp = np.array([255, 255, 255], dtype='uint8')
    whiteMask = cv2.inRange(hls, wLow, wUpp)
    whiteLine = cv2.bitwise_and(hls, hls, mask = whiteMask).astype(np.uint8)

    processedImg = cv2.bitwise_or(yellowLine, whiteLine)
    bgrImg = cv2.cvtColor(processedImg, cv2.COLOR_HLS2BGR)
    
    return(bgrImg)
    
# Define Region of Interest 
def regionOfInterest(img):
    # VerticesofInterest = [(150,700), (580, 450), (750, 450), (1200,700), (1000,700), (650, 500), (350, 700)] # coordinates passed as a tuple
    VerticesofInterest = [(150,700), (600, 450), (740, 450), (1200, 700)] # coordinates passed as a tuple
    maskOut = np.zeros_like(img)
    matchMaskColor = (255,255,255)
    cv2.fillPoly(maskOut, np.array([VerticesofInterest], np.int32), matchMaskColor)
    maskedImg = cv2.bitwise_and(img, maskOut)
    return maskedImg

def getCanny(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(grey,(5,5), 0)
    CannyImg = cv2.Canny(blur_img, 250, 350)
    return(CannyImg)

def fillLaneSpace(straight, dashed, img):

    LineAB = straight[0][0]
    LineCD = dashed[1][0]  
    pt1, pt2 = (LineAB[0],LineAB[1]), (LineAB[2],LineAB[3])
    pt3, pt4 = (LineCD[0],LineCD[1]), (LineCD[2],LineCD[3])
    VerticesofInterest = [pt1, pt2, pt3, pt4]
    # build a mask
    cv2.fillPoly(img, np.array([VerticesofInterest], np.int32), (222,50,50))
    return(img)

def getHough(og_frame, edges): 
    lines = cv2.HoughLinesP(edges,rho=1,threshold=20,theta=np.pi/180,
                            minLineLength=2,maxLineGap=300,lines=np.array([]))
    blank = np.zeros((og_frame.shape[0],og_frame.shape[1],3),dtype=np.uint8)
    blank1 = np.zeros((og_frame.shape[0],og_frame.shape[1],3),dtype=np.uint8)

    laneDash,laneStr = [], []
    slopeDash, slopeStr = [], []

    for x1,y1,x2,y2 in lines[0]:
        LineSlope = (y2 - y1)/(x2 - x1)
        
    for line in lines:
        x1,y1,x2,y2 = line[0]
        slope = (y2 - y1)/(x2 - x1)
        if abs(slope) > 2.5 :
            if (slope*LineSlope) > 0:
                cv2.line(blank1,(x1,y1),(x2,y2),(0,255,0),4)
            else:
                cv2.line(blank1,(x1,y1),(x2,y2),(0,0,255),4)

    for line in lines:
        x1,y1,x2,y2 = line[0]
        slope = (y2 - y1)/(x2 - x1)
        if abs(slope) > 2.5 :
            if (slope*LineSlope) > 0:
                cv2.line(blank,(x1,y1),(x2,y2),(0,255,0),4)
                laneStr.append(line)
                slopeStr.append(slope)
            else:
                cv2.line(blank,(x1,y1),(x2,y2),(0,0,255),4)
                laneDash.append(line)
                slopeDash.append(slope)

    final_img = cv2.addWeighted(og_frame,1,blank,0.5, 0.0)
    justLines = cv2.addWeighted(og_frame,1,blank1,1,0.0)
    return final_img, justLines

def getHistogram(img):
    histogram = np.sum(img, axis=0)
    midPoint = int(histogram.shape[0]/2)
    
    # Pixel Range for Lanes
    left = np.argmax(histogram[:midPoint]) # the solid lane
    right = np.argmax(histogram[midPoint:]) + midPoint # the dashed lane
    imgCenter = int(img.shape[1]/2)
    laneCenter = left + (right - left)/2 
    x, y = [], []
    for i in range(len(histogram)):
        x.append(i)
        y.append(histogram[i])
        
    # plt.plot(x,y)
    # plt.show()
    
    # Turn prediction for lanes --> assume camera in center of car
    if (laneCenter - imgCenter < 0):
        prediction = "Right Turn"
    elif (laneCenter - imgCenter < 8):
        prediction = "Straight"
    else:
    	prediction = "left Turn"
    
    return(prediction,left,right)

def slidingWindows(hough, img, left_X,right_X):
    left, right = left_X, right_X
    n = 10 # number of windows
    m = 25 # margin of 100 pixels
    windowHeight = int(img.shape[0]/n)
    # get indices of pixesl that are not black
    nonZero = img.nonzero()
    nonZeroY, nonZeroX = np.array(nonZero[0]), np.array(nonZero[1])
    # list of indices of lane pixels
    leftInd, rightInd = [], []
    imgHeight = img.shape[0]
    boxes = hough.copy()
    # Step through the windows one by one
    for window in range(0,n):
		# Identify window boundaries in x and y (and right and left)
        winMinY = imgHeight - (window+1)*windowHeight
        winMaxY = imgHeight - (window)*windowHeight
        winMinX_Left, winMaxX_Left = left - m, left + m
        winMinX_Right, winMaxX_Right = right - m, right + m
        # cv2.rectangle(boxes,(winMinX_Left,winMinY),(winMaxX_Left,winMaxY), (250,15,20), 2)
        # cv2.rectangle(boxes,(winMinX_Right,winMinY),(winMaxX_Right,winMaxY), (250,15,20), 2)
  		# Identify the nonzero pixels in x and y within the window
        nonZeroLeft = ((nonZeroY >= winMinY) & (nonZeroY < winMaxY) 
                       & (nonZeroX >= winMinX_Left) & (nonZeroX < winMaxX_Left)).nonzero()[0]
        nonZeroRight = ((nonZeroY >= winMinY) & (nonZeroY < winMaxY) 
                        & (nonZeroX >= winMinX_Right) & (nonZeroX < winMaxX_Right)).nonzero()[0]
  		# Append these indices to the list
        leftInd.append(nonZeroLeft)
        rightInd.append(nonZeroRight)
        minPixels = 25
  		# If found > minpix pixels, move to next window
        if len(nonZeroLeft) > minPixels:
            left = int(np.mean(nonZeroX[nonZeroLeft]))
        if len(nonZeroRight) > minPixels:        
            right = int(np.mean(nonZeroX[nonZeroRight]))
    # Concatenate the arrays of indices into single list
    # plt.imshow()
    leftInd = np.concatenate(leftInd)
    rightInd = np.concatenate(rightInd)
    
    # Extract left and right line pixel positions
    leftx = nonZeroX[leftInd]
    lefty = nonZeroY[leftInd] 
    rightx = nonZeroX[rightInd]
    righty = nonZeroY[rightInd] 
    
    if leftx.size == 0 or rightx.size == 0 or lefty.size == 0 or righty.size == 0:
   		return(None)
       
    return(boxes, leftx,lefty,rightx,righty)

def getCurvature(height, leftx, lefty, rightx, righty):
    y_eval = np.max(height)
    left_fit_cr = np.polyfit(lefty*(30/720), leftx*(370/7), 2)
    right_fit_cr = np.polyfit(righty*(30/720), rightx*(370/7), 2)
    # Calculate radius of curvature
    leftCurve = ((1 + (2*left_fit_cr[0]*y_eval*(30/720) + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    rightCurve = ((1 + (2*right_fit_cr[0]*y_eval*(30/720) + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return(leftCurve, rightCurve)

def polyfit(img,leftx,lefty,rightx,righty):
    # Perform Polynomial fit, get 2nd order equation
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        height = np.linspace(0, img.shape[0]-1, img.shape[0] )
        output = np.dstack((img,img,img))*255	
        output = output.copy()	
        output[lefty, leftx] = [255, 0, 0]
        output[righty, rightx] = [255, 0, 0]
    	# Fit a second order polynomial to each 
        left_bot_X = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
        right_bot_X = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
        # Extract points from fit
        ptsLeft = np.array([np.transpose(np.vstack([left_bot_X, height]))])
        ptsRight = np.array([np.flipud(np.transpose(np.vstack([right_bot_X, height])))])
        pts = np.hstack((ptsLeft, ptsRight))
        pts = np.array(pts, dtype=np.int32)
        
        leftCurve, rightCurve = getCurvature(height,leftx,lefty,rightx,righty)
        print(leftCurve, 'metres', rightCurve, 'metres')


    except:
        pass
    
    # get curvature from polyfit lines

    return(leftCurve, rightCurve, pts)




# =====================================================================
#                           Warping
# =====================================================================

sourcePts = np.array([[120,650], [500, 450], [825, 450], [1250, 650]])
destinationPts = np.array([[0,800],[0, 0],[350, 0],[350,800]])
hMat, status = cv2.findHomography(sourcePts, destinationPts)
# invhMat, status = cv2.getPerspectiveTransform(destinationPts, sourcePts)
invhMat = np.linalg.inv(hMat)

# =====================================================================
#                           Main Function
# =====================================================================

global_vars = {
    'leftCurve': None,
    'rightCurve': None,
    }

count = 0

if __name__== "__main__":
    # read video
    cap = cv2.VideoCapture('Q3_PredictTurn.mp4')

    if (cap.isOpened() == False):
        print("Error opening video file")
    
    while(cap.isOpened()):
        ret , frame = cap.read()
        if ret == True:
            laneMasks = masks(frame)
            maskedImg = regionOfInterest(laneMasks)  # define Area of Interest
            # -----------------------------------------------------
            Lanes = np.copy(frame)
            warped = cv2.warpPerspective(maskedImg, hMat,  (350,800))
            CannyImg = getCanny(warped)  # get edges
            hough, justLines = getHough(warped, CannyImg)
            # -----------------------------------------------------
            prediction, strCord, dshCord = getHistogram(CannyImg)
            # print("prediction:", prediction)
            img, leftX,leftY,rightX,rightY = slidingWindows(hough, CannyImg,strCord,dshCord)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
            leftCurve, rightCurve, points = polyfit(CannyImg, leftX,leftY,rightX,rightY)
            CurveRadius = (leftCurve + rightCurve)/2
            global_vars['leftCurve'] = leftCurve
            global_vars['rightCurve'] = rightCurve
            # plt.imshow(img)
            # plt.show()
            # break
            color_blend = np.zeros_like(warped).astype(np.uint8)
            cv2.fillPoly(color_blend, points, (222,150,100))
            toUnWarp = cv2.addWeighted(hough,1,color_blend,0.5,0.0)
            # -----------------------------------------------------
            unWarped = cv2.warpPerspective(toUnWarp, invhMat,  (1280, 720))
            final_img = cv2.addWeighted(frame,0.8,unWarped,1,0.0)
            # -----------------------------------------------------
            cv2.putText(final_img, prediction, (80,100),cv2.FONT_HERSHEY_DUPLEX, 1.5,(120,255,120),3, cv2.LINE_AA)
            cv2.putText(final_img,'Curve Radius: '+str((global_vars['leftCurve']+global_vars['rightCurve'])/2)[:7]+' m',(80,200), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5,(120,255,120),3, cv2.LINE_AA)

            
            cv2.imshow(' Video output ', final_img)
            count += 1
            
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
            
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    

