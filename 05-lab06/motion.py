#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from datetime import datetime


BLUR_KERNEL_P0 = (21, 21)
BLUR_KERNEL_P1 = (11, 11)
LEARNING_RATE = 0.001
MIN_CONTOUR_AREA = 625

cap = cv2.VideoCapture("lab_images/people-walking.mp4")

#backSub = cv2.createBackgroundSubtractorKNN()
backSub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_blured = cv2.GaussianBlur(frame, BLUR_KERNEL_P0, 0)
    #backSubmask = backSub.apply(frame_blured, learningRate=LEARNING_RATE)
    backSubmask = backSub.apply(frame_blured)

    # thresh = cv2.GaussianBlur(backSubmask, BLUR_KERNEL_P1, 0)
    # thresh = cv2.erode(backSubmask, None, iterations=3)
    # thresh = cv2.dilate(thresh, None, iterations=6)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    thresh = cv2.morphologyEx(backSubmask, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, None, iterations=3)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_movimento = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if  area < MIN_CONTOUR_AREA:
            continue

        area_movimento += area
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 255, 255), 2)

    # cv2.putText(
    #     frame, datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
    #     (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    # Show results
    cv2.imshow("Feed", frame)
    cv2.imshow("Thresh", thresh)

    # Wait for key 'ESC' to quit
    key = cv2.waitKey(90) & 0xFF
    if key == 27:
        break

# That's how you exit
cap.release()
cv2.destroyAllWindows()