#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun July  25 02:55:07 2021

@author: nathan
"""

import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser(description = 'This is the default value')
ap.add_argument('-i', '--video', required=True,
                help = 'path to input video')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

# YOLO functions

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h): # draws rectangle over the given predicted region and writes class name over the box

    label = str(classes[class_id])+" "+str(round(confidence,2))

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Lane detection functions

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # creating a greyscale image
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # applying Gaussian Blur with a 5X5 kernal to reduce noise
    canny = cv2.Canny(blur, 50, 150) # computing the strongest gradients 
    return canny 

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(205, height), (1173, height), (565, 266)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines): # drawing lines on black image
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (np.int64(x1), np.int64(y1)), (np.int64(x2), np.int64(y2)), (255, 0, 0), 10) # last 2 arguments are colour and line thickness
    return line_image

def make_coordinates(image, line_parameters, direction):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0
    y1 = image.shape[0]
    y2 = int(y1*(3/5)) # lines will start from the bottom and go 3/5th upwards
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    if direction == "right" and (x1 > 1600 or x1 < 640): # lower x value
        x1 = 1000
    if direction == "right" and (x2 > x1 or x2 < 640): # upper x value
        x2 = 700
    if direction == "left" and (x1 < -200 or x1 > 640): # lower x value
        x1 = 300
    if direction == "left" and (x2 > 650 or x2 < x1): # upper x value
        x2 = 500
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = [] # average of the lines on the left
    right_fit = [] # average of the lines on the right
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1) # this gives you slope and parameters
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average, "left")
    right_line = make_coordinates(image, right_fit_average, "right")
    return np.array([left_line, right_line])

# importing and showing video
# analysing the video

cap = cv2.VideoCapture(args.video)

frame_array = []

while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)

# detecting lines in cropped gradient image
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # precision of 2 pixels with with 1 degree i.i pi/180 with a threshold of 100

# averaging out the lines to display a smooth line
    averaged_lines = average_slope_intercept(frame, lines)

    line_image = display_lines(frame, averaged_lines)

# combining black image with lines with original image
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # 0.8 reduces the intensity of lane_image

# IMPLEMENTING YOLO

    Width = combo_image.shape[1]
    Height = combo_image.shape[0]
    scale = 0.00392

    classes = None

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    np.random.seed(130)
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(args.weights, args.config)  #  reads the weights and config file and creates the network.

    blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False) # preparing each frame of the input video to run through the deep neural network.

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(combo_image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

# Exporting frame to video
        
    #reading each files
    img = combo_image
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)
    
# video frame merging ends here

    cv2.imshow('Result', combo_image)
    #cv2.waitKey(1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # displays each frame for specified amount of mili-seconds
        break
    
out = cv2.VideoWriter('tests/'+args.video+'_test.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 40, size) # second last argument is the number of frames per second
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()

cap.release()