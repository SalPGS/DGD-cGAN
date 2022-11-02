"""
 > Split the image into four equal sizes  
"""
#Libraries
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt

#Folder with the images
input_path = "/your/folder"
os.chdir(input_path)


for image in os.listdir():
    img = cv2.imread(image)
    # Vertical image
    height = img.shape[0]
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    left1 = img[:, :width_cutoff]
    right1 = img[:, width_cutoff:]
    
    #Rotate 90° to left side (horizontal image)
    img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
     # Vertical image
    height = img.shape[0]
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    l1 = img[:, :width_cutoff]
    l2 = img[:, width_cutoff:]
   
    #Rotate 90° to left side (horizontal image)
    l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #save image l1
    cv2.imwrite("/your/path/l1_{}".format(image), l1)
    #Rotate 90° to left side (horizontal image)
    l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #save image l2
    cv2.imwrite("/your/path/l2_{}".format(image), l2)
    # Other half
    #Rotate 90° to right side 
    img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
    height = img.shape[0]
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    r1 = img[:, :width_cutoff]
    r2 = img[:, width_cutoff:]
    # finish vertical devide image
    #Rotate 90° to right side 
    r1 = cv2.rotate(r1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #save image r1
    cv2.imwrite("/your/path/r1_{}".format(image), r1)
    #Rotate 90° to right side 
    r2 = cv2.rotate(r2, cv2.ROTATE_90_COUNTERCLOCKWISE)    
    cv2.imwrite("/your/path/r2_{}".format(image), r2)

