import numpy as np
import cv2
import copy
import math
from enum import Enum
import sys

class Color(Enum):
        RED = (0,0,255)
        GREEN = (0,255,0)
        BLUE = (255,0,0)

class Position(Enum):
        TOP = 0
        BOTTOM = 1
        LEFT = 2
        RIGHT = 3

class GoalLine:
    def __init__(self, p1, p2, position):
        self.p1 = p1
        self.p2 = p2
        self.position = position
        self.enter_blobs = 0
        self.exit_blobs = 0

    def draw(self, img, line_color):
        cv2.line(img, self.p1, self.p2 ,line_color.value, 5)

    def intersect(self, blob):
        x1, y1 = self.p1
        x2, y2 = self.p2
        rect_x, rect_y = blob.getLastShape().getUpperLeftPoint()

        if blob.marked and blob.marked == self.position:
            # do not consider blob that
            # have already passed through this line
            return
        
        if self.position == Position.LEFT or self.position == Position.RIGHT:
            max_y  = max(y1, y2)
            min_y  = min(y1, y2)
            pred_rect_x, _ = blob.shapeHistory[-2].getUpperLeftPoint()
            
            # if the x is the same
            # and the rect_y is between the two y
            # means that the line is crossing the rect
            if rect_x in range(x1-5, x1+5) and rect_y > min_y and rect_y < max_y:
                # there is an intersection
                # check if the blob is entering or not
                blob.marked = self.position
                blob_movement_x = rect_x - pred_rect_x
                if self.position == Position.LEFT:
                    if  blob_movement_x > 0:
                        self.enter_blobs += 1
                    else:
                        self.exit_blobs += 1
                else:
                    if blob_movement_x < 0:
                        self.enter_blobs += 1
                    else:
                        self.exit_blobs += 1

        if self.position == Position.TOP or self.position == Position.BOTTOM:
            max_x  = max(x1, x2)
            min_x  = min(x1, x2)
            _, pred_rect_y = blob.shapeHistory[-2].getUpperLeftPoint()
            if rect_y in range(y1-5, y1+5) and rect_x > min_x and rect_x < max_x:
                # there is an intersection
                # check if the blob is entering or not
                blob.marked = self.position
                blob_movement_y = rect_y - pred_rect_y
                if self.position == Position.TOP:
                    if  blob_movement_y > 0:
                        self.enter_blobs += 1
                    else:
                        self.exit_blobs += 1
                else:
                    if blob_movement_y < 0:
                        self.enter_blobs += 1
                    else:
                        self.exit_blobs += 1             

    def drawScore(self, image):
        x, y = self.p1
        cv2.putText(image, str(self.enter_blobs) + '/' + str(self.exit_blobs), (x - 50, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 3, cv2.LINE_AA)

class Rect:
    """
    This class contains helper functions to work with opencv rectangles
    """
    def __init__(self, x, y, w, h):
        """
        where "x" and "y" are the coords of the upper left point
        "w" is the width and "h" the height of the rectangle
        """
        self.shape = (x, y, w, h)
    
    def getHeight(self):
        """
        Returns the Height
        """
        return self.shape[3]

    def getWidth(self):
        """
        Returns the Width
        """
        return self.shape[2]

    def getUpperLeftPoint(self):
        """
        Returns a tuple (x, y) which represents the coords
        of the upper left point of the rectangle
        """
        return (self.shape[0], self.shape[1])

    def getArea(self):
        """
        Returns the area of the rectangle
        """
        return self.getWidth()*self.getHeight()

    def getRawShape(self):
        """
        Returns a tuple (x, y, w, h)
        where "x" and "y" are the coords of the upper left point
        "w" is the
        """
        return self.shape

    def intersect(self, rect):
        """
        Return the intersection rectangle between the current object
        and the object "rect" passed as parameters of the function.
        If there is no intersection, it returns None
        """
        a = self.getRawShape()
        b = rect.getRawShape()
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w<0 or h<0: return None # or (0,0,0,0) ?
        return Rect(x, y, w, h)

    def merge(self, rect_to_merge):
        current_shape = self.getRawShape()
        new_shape = rect_to_merge.getRawShape()
        merged_x = min(current_shape[0], new_shape[0])
        merged_y = min(current_shape[1], new_shape[1])
        merged_w = max(current_shape[2], new_shape[2])
        merged_h = max(current_shape[3], new_shape[3])
        return Rect(merged_x, merged_y, merged_w, merged_h)

    def draw(self, img, color):

        if not color:
            color = (0, 0, 255)

        x, y, w, h = self.getRawShape()
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)

class Blob:
    def __init__(self, rect, frame):
        """
        Rect is a object of class Rectangle
        Frame is number of the number of frame in which this object
        is created.
        """
        self.shapeHistory = [rect]
        self.lastUpdateFrame = frame
        self.marked = False

    def getLastShape(self):
        """
        Returns the most updated rectangle of the blob
        """
        return self.shapeHistory[-1]

    def getAverageArea(self):
        areas = [x.getArea() for x in self.shapeHistory]
        return sum(areas) / len(areas)

    def predictShape(self):
        last_shape = self.shapeHistory[-1]
        pred_shape = self.shapeHistory[-2]
        x_last, y_last, w, h = last_shape.getRawShape()
        x_pred, y_pred = pred_shape.getUpperLeftPoint()
        deltaX = x_last - x_pred
        deltaY = y_last - y_pred
        predicted_x = x_last + deltaX
        predicted_y = y_last + deltaY
        return Rect(predicted_x, predicted_y, w, h)

def detectObjects(frame, fgbg):
    detected_list = []
    
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    
    t_mask = fgbg.apply(y, 0.01)

    small_img = cv2.resize(t_mask, (0,0), fx=0.5, fy=0.5) 
    #cv2.imshow('bs_thr', small_img)

    # median blur
    blurred = cv2.medianBlur(t_mask, 3)

    # dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    dilation = cv2.dilate(closing, kernel, iterations = 1)

    # remove shadoes
    small_img = cv2.resize(dilation, (0,0), fx=0.5, fy=0.5) 
    #cv2.imshow('after_operations', small_img)

    im2, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        x,y,w,h = cv2.boundingRect(hull)
        founded_rect = Rect(x, y, w, h)
        area =  founded_rect.getArea()
        if area < 100 or area > 10000 or w < 5 or h < 5: continue
        detected_list.append(founded_rect)
    
    return detected_list

def addNewBlob(current_blob, blobs, frame_number):
    # find if there is a matching blobs
    founds = []
    rates = []

    # is there an intersection with the existing blob in the list?
    for blob in blobs:
        
        if len(blob.shapeHistory) > 1:
            shape = blob.predictShape()
        else:
            shape = blob.getLastShape()

        shape2 = current_blob.getLastShape()
        intersection = shape.intersect(shape2)

        # found an intersection = there is a good possibility that that shape
        # belongs to the same frame Blob object, use the "intersection_over_union" rate
        # to decide wether associate the shape to that blobs or not.
        # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        # 0.2 because it gives me some robustness
        if intersection != None:
            union_area = shape.getArea() + shape2.getArea() - intersection.getArea()
            intersection_over_union_rate = intersection.getArea() / union_area
            if intersection_over_union_rate > 0.2:
                founds.append(blob)
                rates.append(intersection_over_union_rate)
        
    if len(founds) > 0:
        found = founds[rates.index(max(rates))]

        # found an intersection
        new_shape = current_blob.getLastShape()
        if found.lastUpdateFrame == frame_number:
            # multiple shape overlap in the same frame
            # maybe this two object refers to the same
            # object, do a merge
            new_shape = found.shapeHistory.pop().merge(new_shape)
        
        found.shapeHistory.append(new_shape)
        found.lastUpdateFrame = frame_number
    else:
        blobs.append(current_blob)
    
def removeOldBlobs(blobs, frame_number):
    rate = 1
    return [x for x in blobs if abs(frame_number - x.lastUpdateFrame) <= rate]

if __name__ == "__main__":

    video_path = 'res/video.mov'
    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    cap = cv2.VideoCapture(video_path)
    
    # Exit if video not opened.
    if not cap.isOpened():
        print("Could not open video")
        sys.exit()

    frame_height = 1916
    frame_width = 1409

    line_left = GoalLine((100,250), (100,frame_height), Position.LEFT)
    line_right = GoalLine((290,250), (290, frame_height), Position.RIGHT)
    line_top = GoalLine((70,350), (600, 350), Position.TOP)
    line_bottom = GoalLine((150,400), (300, 400), Position.BOTTOM)

    fgbg_left = cv2.createBackgroundSubtractorMOG2(varThreshold=25, detectShadows=False)
    fgbg_right = cv2.createBackgroundSubtractorMOG2(varThreshold=120, detectShadows=False)
    fgbg_top = cv2.createBackgroundSubtractorMOG2(varThreshold=25, detectShadows=False)
    fgbg_bottom = cv2.createBackgroundSubtractorMOG2(varThreshold=200, detectShadows=False)

    blobs_list = [[], [], [], []]
    fgbg_top_list = [fgbg_left, fgbg_right, fgbg_top, fgbg_bottom]

    frame_counter = 1

    while(1):

        ok, frame = cap.read()
        lines = [line_left, line_right, line_top, line_bottom]

        if not ok:
            print("End of the video")
            for l, line in enumerate(lines):
                final_score = (line.enter_blobs, line.exit_blobs)
                print(l, final_score)
            break 
        
        height, width, channels = frame.shape

        # destra
        # cropped = frame[int(height / 3): int(height / 3) + 500, width-500:width]
        cropped_left = frame[int(height / 3): int(height / 3) + 500, 0:500]
        cropped_right = frame[int(height / 3): int(height / 3) + 500, width-500:width]
        cropped_top = frame[0:500, int(width / 3)-100:int(width / 3)+500]
        cropped_bottom = frame[height - 500:height, int(width / 3)-100:int(width / 3)+500]

        portions = [cropped_left, cropped_right, cropped_top, cropped_bottom]

        frame_counter += 1

        for i, cropped in enumerate(portions):

            cropped_to_draw = cropped.copy()

            line = lines[i]
            blobs = blobs_list[i]
            fgbg = fgbg_top_list[i]
 
            # call detector using frame difference
            detected = detectObjects(cropped, fgbg)
            
            # add new blobs
            for d in detected:
                addNewBlob(Blob(d, frame_counter), blobs, frame_counter)
    
            # remove old blobs
            blobs = removeOldBlobs(blobs, frame_counter)

            # predict position of untracked blobs
            # predictPositionOfUntrackedBlobs(blobs, frame_counter)

            for blob in blobs:
                if blob.lastUpdateFrame == frame_counter:
                    color = (0, 0, len(blob.shapeHistory)*50)
                    blob.getLastShape().draw(cropped_to_draw, color)

            line.draw(cropped_to_draw, Color.GREEN)
            for blob in blobs:
                if len(blob.shapeHistory) > 2 and blob.lastUpdateFrame == frame_counter:
                    # can check this blob
                    line.intersect(blob)

            line.drawScore(cropped_to_draw)
            small_img = cv2.resize(cropped_to_draw, (0,0), fx=0.5, fy=0.5) 
            cv2.imshow('frame' + str(i), small_img)

        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
