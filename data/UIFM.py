"""
Underwater image formation process
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

  

'''
This for loop will apply the above functions to the images inside and save their airlight pair
'''

#Folder with images
input_folder = "/your/same_folder"
#Folder were there airlight images will be stored here using the same name.jpg you can change it modifying image_name variable
output_airlight = "/your/same_folder"

os.chdir(input_folder)

for img in os.listdir():
    underwater_image = Image.open(img)
    #underwater_image = underwater_image.resize((256, 256))
    ui_matrix = np.array(underwater_image).astype("float64")
    gw_image, r_gray, gr_gray, b_gray = Gray_world(image=ui_matrix)
    al = Air_light(img=ui_matrix, ar=r_gray, ag=gr_gray,ab= b_gray)
    #image_name = (img.split('.')[-2] +'%d.jpg'%d)
    image_name = (img.split('.')[-2]+'.jpg') #image's original name
    file_path = os.path.join(output_airlight,image_name)
    al_image = Image.fromarray(al, 'RGB')
    al_image = al_image.resize((1807, 1355))
    al_image.save(file_path)
   


#Maximum intensity ground truth in the paper we did not use it but it is optional
def Ground_truth(J):
    '''
    Function that calculates the maximum value of the ground truth image
    Parameters:
        J: underwater image matrix
        r,g,b channels
    Returns:
        The J matrix with the r, g ,b  maximum values
    '''
    #r, g, b channels
    r = J[:,:,0]
    g = J[:,:,1]
    b = J[:,:,2]
    

    # maximum value of each channel
    r_max = np.max(r)
    g_max = np.max(g)
    b_max = np.max(b)

    # filling matrix with rgb maximum values
    J[:,:,0] = r_max
    J[:,:,1] = g_max
    J[:,:,2] = b_max


    return J
   
#Transmission   
def Transmission(I, A, J):
    '''
    Function for returning the transmission of light in the underwatre image
    Folowing equation T = I-A/J-A

    Parameters:
        I: is the image image ui_matrix
        A: is airlight which is al function 
        J: is the ground truth image 
    Returns:
        Matrix with the transmission of light
    '''

    T = (I-A) / (J-A)
    
    #thos works T = (I-A)/(255-A)

    return T
    
   
   
   
'''
This for loop will apply the above functions to the images inside the folder and save it in another one
Remeber to check all the images with the same name.jpg for create the paired data
'''

input_folder2 = "/your/ underwater images folder"
under_path=  glob.glob(input_folder2 + "\\*.jpg") # Grabbing all the image file names
input_folder3 = f"/your/ ground truth images folder"
groundtruth_path = glob.glob(input_folder3 + "\\*.jpg")
output_transmission = "E:\\My_Python\\GAN\\TURBID\\Test2"

os.chdir(input_folder2)
#Counter d 
d=1
#for img2 in os.listdir():
for img2, img3 in zip(under_path, groundtruth_path):
    underwater_image = Image.open(img2)
    g_truth = Image.open(img3)    
    ui_matrix = np.array(underwater_image).astype("float64")
    gt_matrix = np.array(g_truth).astype("float64")
    
    al = Air_light(img = ui_matrix, ar = r_gray, ag = gr_gray, ab = b_gray)
    gt_matrix = np.array(g_truth).astype("float64")
    #j_max = Ground_truth(J=gt_matrix) #for use maximum intensity set J to j_max
    tran = Transmission(I = ui_matrix, A = al, J = gt_matrix)
    image_name2 = (img.split('.')[-2] +'%d.jpg'%d)
    file_path2 = os.path.join(output_transmission,image_name2)
    tran_image = Image.fromarray(tran, 'RGB')
    tran_image.save(file_path2)
    d+=1
