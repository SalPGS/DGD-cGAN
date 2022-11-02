"""
 > Underwater image formation process
 Save all image pairs with the same name in different folders
""
#Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import glob

def Gray_world(image): 
    '''
    Calculates the airlight with gray world assumption dividing each colour channel R, G, B by its average value.
    img: image image
    r: red channel
    g: green channel
    b: blue channel
    '''
    img = image

    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]


    dr = 1.0 / np.mean(r[r != 0])
    dg = 1.0 / np.mean(g[g != 0])
    db = 1.0 / np.mean(b[b != 0])
    # Gray colour dsum*3
    dsum = dr + dg + db

    dr = dr / dsum * 3.
    dg = dg / dsum * 3.
    db = db / dsum * 3.
    
    r_cast =  img[:, :, 0] * dr
    g_cast =  img[:, :, 1] * dg
    b_cast =  img[:, :, 2] * db

    r = r_cast
    g = g_cast
    b = b_cast
    
    return img, dr, dg, db
  
  
  
# Airlight or veiling light

def Air_light(img, ar,ag,ab):
    '''
    This function cast the gray world vector into the image size giving the Airlight matrix. 
    Takes each r g b channel as a scalar from gray world assumption
    Parameters:
        img: is the image image ui_matrix
        ar: is the transmission tran function
    Returns:
        the underwater image
    '''
    height,width,_ = img.shape
    matrix_size = np.zeros((height,width,_))

    #Fill the matrix with the r_gray, gr_gray, b_gray gray world values

    matrix_size[:,:,0].fill(ar)
    matrix_size[:,:,1].fill(ag)
    matrix_size[:,:,2].fill(ab)
    
    return matrix_size

  
#Save images  
'''
This for loop will apply the above functions to the images inside the folder and save it in another one
'''

#Folder with images
input_folder = "/your/folder"
#Folder were there resized images will be stored
output_airlight = "/your/folder"

os.chdir(input_folder)
#Counter d 
d=1
for img in os.listdir():
    underwater_image = Image.open(img)
    ui_matrix = np.array(underwater_image).astype("float64")
    gw_image, r_gray, gr_gray, b_gray = Gray_world(image=ui_matrix)
    al = Air_light(img=ui_matrix, ar=r_gray, ag=gr_gray,ab= b_gray)
    image_name = (img.split('.')[-2] +'%d.jpg'%d)
    file_path = os.path.join(output_airlight,image_name)
    al_image = Image.fromarray(al, 'RGB')
    al_image.save(file_path)
    d+=1
