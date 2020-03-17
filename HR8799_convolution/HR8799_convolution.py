import os
import matplotlib.pyplot as plt
from astropy.io import fits
from glob import glob
from scipy import ndimage
import numpy as np
import time
from skimage import draw, data
import math
# load data here
def ImgLoader(fn_list):
    imgcube = []
    HA=np.zeros(len(fn_list))
    
    for ind, fn in enumerate(fn_list):
        with fits.open(fn) as img:
            imgcube.append(img[0].data)
            HA[ind]=math.radians(img[0].header["HA"])
            
    print ("{} files are loaded".format(len(fn_list)))

    return np.array(imgcube), HA
# path to the directory where the fits files are in
route = r"D:\DirectImaging\HR8799\HR8799\N2.20080918\calibrated/fits"
# a list containing all the fits file
fn_list = glob(os.path.join(route, "*.fits"))
# load data here
imgcube, HA = ImgLoader(fn_list)

# there are total 136 images, each image has 1024 * 1024 pixels
print (imgcube.shape)
img_size = imgcube[0, :, :].shape
side=img_size[0]
def Center_of_Gravity(img):
    centroid_x = np.arange(img_size[1]).reshape(1, -1) * img
    centroid_y = np.arange(img_size[0]).reshape(-1, 1) * img
    centroid_x = centroid_x.sum()
    centroid_y = centroid_y.sum()
    return centroid_x / img.sum(), centroid_y / img.sum()

def convolution(img):
    N=np.zeros((100,100))
    for i in range(25):
        X=i*4
        for j in range(25):
            Y=j*4
            foo_1=foo[y-150+50-Y:y+150+50-Y, x-150+50-X:x+150+50-X]
            N[Y,X]=np.sum(np.multiply(foo_1,foo_0))
    """
    plt.ion()
    plt.figure(2)
    plt.imshow(N,cmap="jet")
    plt.colorbar()
    plt.show()
    plt.pause(1)
    """
    raw, column=N.shape
    position=np.argmax(N)
    #print(position)
    m, n=divmod(position, column)
    #print("(m,n)=",m,n)
    m_=m+2
    n_=n+2
    foo_1=foo[y-150+50-(m_):y+150+50-(m_), x-150+50-(n_):x+150+50-(n_)]
    N[m_,n_]=np.sum(np.multiply(foo_1,foo_0))
    m_=m-2
    n_=n+2
    foo_1=foo[y-150+50-(m_):y+150+50-(m_), x-150+50-(n_):x+150+50-(n_)]
    N[m_,n_]=np.sum(np.multiply(foo_1,foo_0))
    m_=m+2
    n_=n-2
    foo_1=foo[y-150+50-(m_):y+150+50-(m_), x-150+50-(n_):x+150+50-(n_)]
    N[m_,n_]=np.sum(np.multiply(foo_1,foo_0))
    m_=m-2
    n_=n-2
    foo_1=foo[y-150+50-(m_):y+150+50-(m_), x-150+50-(n_):x+150+50-(n_)]
    N[m_,n_]=np.sum(np.multiply(foo_1,foo_0))
    """
    plt.ion()
    plt.figure(2)
    plt.clf()
    plt.imshow(N,cmap="jet")
    plt.colorbar()
    plt.show()
    plt.pause(1)
    """
    raw, column=N.shape
    position=np.argmax(N)
    #print(position)
    m, n=divmod(position, column)
    #print("(m,n)=",m,n)

    foo_1=foo[y-150+50-m:y+150+50-m, x-150+50-n:x+150+50-n]
    for i in range(5):
        for j in range(5):
            foo_1=foo[y-150+50-(m+j-2):y+150+50-(m+j-2), x-150+50-(n+i-2):x+150+50-(n+i-2)]
            N[m+j-2,n+i-2]=np.sum(np.multiply(foo_1,foo_0))
    
    raw, column=N.shape
    position=np.argmax(N)
    #print(position)
    m, n=divmod(position, column)
    print("(m,n)=",m,n)
    
    plt.ion()
    plt.figure(2)
    plt.clf()
    plt.imshow(N,cmap="jet")
    plt.colorbar()
    plt.show()
    plt.pause(40)
    
    return m, n

for ind, img in enumerate(imgcube):
    foo=imgcube[ind,:,:]
    """
    plt.ion()
    plt.figure(0)
    plt.title("the original image")
    plt.imshow(foo, cmap="gray")
    plt.show()
    plt.pause(1)
    """
    centroid = Center_of_Gravity(foo)
    x=int(round(centroid[0]))
    y=int(round(centroid[1]))
    foo_0=foo[y-150:y+150,x-150:x+150]
    """
    plt.ion()
    plt.figure(1)
    plt.title("the shifted image")
    plt.imshow(foo_0, cmap="gray")
    plt.show()
    plt.pause(1)
    """
    p=convolution(foo)
    print(ind+1,"/136")
    #print("(x,y)=",512-(x-50+p[1]),",", 512-(y-50+p[0]))
    shift_value = 512-(y-50+p[0]), 512-(x-50+p[1])
    imgcube[ind] = ndimage.shift(img ,shift_value)
    """
    plt.ion()
    plt.figure(3)
    plt.clf()
    plt.imshow(imgcube[ind],cmap="gray")
    plt.scatter(512, 512, c="r", marker="x" )
    plt.show()
    plt.pause(1)
    plt.close('all')
    """
    
print("registration done")

# construct the PSF using cADI method
# we will subtract each image by this PSF later
PSF_cADI = np.median(imgcube, axis = 0)

img_cADI = np.zeros(PSF_cADI.shape)

#Now we calculate the correct angle 
print("Find the correct paralltic angle")
# https://en.wikipedia.org/wiki/HR_8799
Declination=21+8*(1/60)+0.302*(1/3600)
print("Declination of HR8799; 21° 08' 03.302''")
print("Declination=",Declination,"°")
Declination=math.radians(Declination)
# https://www.ifa.hawaii.edu/mko/coordinates.shtml
Geographical_latitude=19+49*(1/60)+35.61788*(1/3600)
print("Geographical latitude of Keck 2: 19° 49' 35.61788''")
print("Geographical latitude=",Geographical_latitude,"°")
Geographical_latitude=math.radians(Geographical_latitude)

para_angle_thm=np.zeros(len(fn_list))
for i in range(len(fn_list)):
    
    x=(math.sin(HA[i]))/(math.tan(Geographical_latitude)*math.cos(Declination)-math.sin(Declination)*math.cos(HA[i]))
    para_angle_thm[i]=math.atan(x)
    
start=time.time()
for img, angle in zip(imgcube, para_angle_thm):
    img_cADI += ndimage.rotate(img - PSF_cADI,  # subtract the PSF before rotationg
                               -angle,          # in degree. rotate the image counter-clockwisely if the value is positive
                               reshape = False) # retain the same shape as the input image

plt.ion()
plt.figure(figsize = (6, 6))
plt.imshow(img_cADI, cmap = "gray")
plt.xlim(344, 680)
plt.ylim(344, 680)
plt.show()
plt.pause(10)
    



