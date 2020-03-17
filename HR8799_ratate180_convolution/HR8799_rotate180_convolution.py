import os
import matplotlib.pyplot as plt
from astropy.io import fits
from glob import glob
from scipy import ndimage
import numpy as np
import time
from skimage import draw, data
import math

# Load data here
#we get the image data and the hour angel for each image.
def ImgLoader(fn_list):
    imgcube = []
    HA=np.zeros(len(fn_list))
    
    for ind, fn in enumerate(fn_list):
        with fits.open(fn) as img:
            imgcube.append(img[0].data)
            HA[ind]=math.radians(img[0].header["HA"])
            
    print ("{} files are loaded".format(len(fn_list)))

    return np.array(imgcube), HA

def user_input(foo):
    plt.ion()
    plt.figure(0)
    plt.title("the original image")
    plt.imshow(foo, cmap="gray")
    plt.show()
    plt.pause(1)

    print("center(x,y)")
    x=input("x=")
    y=input("y=")
    print("center(x,y)=","(",x,",",y,")")
    x=round(float(x))
    y=round(float(y))
    sure='N'
    while sure!='Y':
        a=input("Searching area a x a ,a=")
        a=300
        a=int(a)
        print("Searching area:",a,"x",a)
        print(a*a,"pixels")
        foo_shift=foo[int(round(y-a/2)):int(round(y+a/2)), int(round(x-a/2)):int(round(x+a/2))]
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.title("the shifted image")
        plt.imshow(foo_shift, cmap="gray")
        plt.scatter(a/2, a/2, c="r", marker="x")
        plt.show()
        plt.pause(1)
        sure_1=input("Is the center OK？[Y/N]")
        while sure_1!='Y':
            print("center(x,y)")
            x=input("x=")
            y=input("y=")
            print("center(x,y)=","(",x,",",y,")")
            x=round(float(x))
            y=round(float(y))
            foo_shift=foo[int(round(y-a/2)):int(round(y+a/2)), int(round(x-a/2)):int(round(x+a/2))]
            plt.ion()
            plt.figure(1)
            plt.clf()
            plt.title("the shifted image")
            plt.imshow(foo_shift, cmap="gray")
            plt.scatter(a/2, a/2, c="r", marker="x")
            plt.show()
            plt.pause(1)
            sure_1=input("Is the center OK ？[Y/N]")
        sure=input("Is the area fine for searching？[Y/N]")
    search_ok='N'
    while search_ok!='Y':
        search=input("Searching range a x a, a:")
        search=int(search)
        plt.figure(1)
        plt.clf()
        plt.title("the shifted image")
        plt.imshow(foo_shift, cmap="gray")
        plt.scatter(a/2, a/2, c="r", marker="x")
        plt.scatter(a/2-search/2,a/2-search/2, c="w", marker="+")
        plt.scatter(a/2-search/2,a/2+search/2, c="w", marker="+")
        plt.scatter(a/2+search/2,a/2-search/2, c="w", marker="+")
        plt.scatter(a/2+search/2,a/2+search/2, c="w", marker="+")
        plt.show()
        plt.pause(1)
        search_ok=input("Is Searching range Ok?[Y/N]")
    print("We will start the centeroid measurement. ")
    Start=input("OK?[Y/N]")

    return a, search, x, y

def Center_of_Gravity(img):
    centroid_x = np.arange(img_size[1]).reshape(1, -1) * img
    centroid_y = np.arange(img_size[0]).reshape(-1, 1) * img
    centroid_x = centroid_x.sum()
    centroid_y = centroid_y.sum()
    return centroid_x / img.sum(), centroid_y / img.sum()

def rotation(img,search,a, x,y):
    N_rot=np.zeros((search,search))
    for i in range(search):
        x_check=x-search/2+i
        for j in range(search):
            y_check=y-search/2+j
            foo_shift_check=img[int(round(y_check-a/2)):int(round(y_check+a/2)), int(round(x_check-a/2)):int(round(x_check+a/2))]
            foo_shift_check_rot=np.rot90(np.rot90(foo_shift_check))
            N_rot[j,i]=np.sum(np.absolute(foo_shift_check-foo_shift_check_rot))
    raw, column=N_rot.shape
    position=np.argmin(N_rot)
    #print(position)
    m, n=divmod(position, column)
    print("(m,n)=",m,n)
    """
    plt.ion()
    plt.figure(2)
    plt.clf()
    plt.imshow(N_rot, cmap="jet")
    plt.colorbar()
    plt.scatter(n, m, c="r", marker="x" )
    plt.show()
    plt.pause(2)
    
    plt.ion()
    plt.figure(3)
    plt.clf()
    plt.title("the shifted image")
    plt.imshow(img, cmap="gray")
    plt.scatter((x-a/2)+(a-search)/2+n, (y-a/2)+(a-search)/2+m, c="r", marker="x")
    plt.show()
    plt.pause(1)
    """
    return m, n, np.amin(N_rot)

def convolution(img, Standard, a, x, y):
    N=np.zeros((a, a))
    for i in range (int (a/4)):
        X=i*4
        for j in range(int(a/4)):
            Y=j*4
            foo=img[int(y-a/2-a/2+Y):int(y-a/2+a/2+Y), int(x-a/2-a/2+X):int(x-a/2+a/2+X)]
            N[Y,X]=np.sum(np.multiply(foo,Standard))
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
    print("(m,n)=",m,n)
    m_=m+2
    n_=n+2
    foo=img[int(y-a/2-a/2+m_):int(y-a/2+a/2+m_), int(x-a/2-a/2+n_):int(x-a/2+a/2+n_)]
    N[m_,n_]=np.sum(np.multiply(foo,Standard))
    m_=m-2
    n_=n+2
    foo=img[int(y-a/2-a/2+m_):int(y-a/2+a/2+m_), int(x-a/2-a/2+n_):int(x-a/2+a/2+n_)]
    N[m_,n_]=np.sum(np.multiply(foo,Standard))
    m_=m+2
    n_=n-2
    foo=img[int(y-a/2-a/2+m_):int(y+a/2-a/2+m_), int(x-a/2-a/2+n_):int(x-a/2+a/2+n_)]
    N[m_,n_]=np.sum(np.multiply(foo,Standard))
    m_=m-2
    n_=n-2
    foo=img[int(y-a/2-a/2+m_):int(y-a/2+a/2+m_), int(x-a/2-a/2+n_):int(x-a/2+a/2+n_)]
    N[m_,n_]=np.sum(np.multiply(foo,Standard))
    """
    plt.ion()
    plt.figure(2)
    plt.clf()
    plt.imshow(N,cmap="jet")
    plt.colorbar()
    plt.show()
    plt.pause(3)
    """
    raw, column=N.shape
    position=np.argmax(N)
    #print(position)
    m, n=divmod(position, column)
    print("(m,n)=",m,n)

    for i in range(5):
        for j in range(5):
            foo=img[int(y-a/2-a/2+(m+j-2)):int(y-a/2+a/2+(m+j-2)), int(x-a/2-a/2+(n+i-2)):int(x-a/2+a/2+(n+i-2))]
            N[m+j-2,n+i-2]=np.sum(np.multiply(foo,Standard))
    
    raw, column=N.shape
    position=np.argmax(N)
    #print(position)
    m, n=divmod(position, column)
    print("(m,n)=",m,n)
    """
    plt.ion()
    plt.figure(2)
    plt.clf()
    plt.imshow(N,cmap="jet")
    plt.colorbar()
    plt.show()
    plt.pause(1)
    """
    return m, n

# path to the directory where the HR8799 fits files are in
route = r"G:\DirectImaging\HR8799\HR8799\N2.20080918\calibrated/fits"
# a list containing all the fits file
fn_list = glob(os.path.join(route, "*.fits"))
# load data here
imgcube, HA = ImgLoader(fn_list)
# there are total 136 images, each image has 1024 * 1024 pixels
print(imgcube.shape)
img_size = imgcube[0, :, :].shape
side=img_size[0]

foo=imgcube[0,:,:]
'''
#User input:
a, search, x, y=user_input(foo)
print("x=", x, "\ny=", y, "\nSearching range:", search, "\nimage size", a)
'''
#=============
#我使用的初始條件
a=300
search=200
#=============

#(1) rotate 180
rot_record=np.zeros((136))
m=np.zeros((136))
n=np.zeros((136))
for ind, img in enumerate(imgcube):
    x, y=Center_of_Gravity(img) #function: Center_of_Gravity( )
    m[ind], n[ind], rot_record[ind]=rotation(img, search ,a, x, y) #function: rotation( )
    ind_0=ind+1
    print(ind_0,"/136")
print("registration done")
# let the ith image be standard image
i=np.argmin(rot_record)
m=int(m[i])
n=int(n[i])

Standard=np.array(imgcube[i,:,:])
x, y=Center_of_Gravity(Standard)

x=x-search/2+n
y=y-search/2+m
Standard=Standard[int(y-a/2):int(y+a/2), int(x-a/2):int(x+a/2)]

# (2) convolution
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
    p=convolution(foo, Standard, a, x, y)  #function: convolution( )
    print(ind+1,"/136")
    #print("(x,y)=",512-(x-50+p[1]),",", 512-(y-50+p[0]))
    shift_value=512-y,512-x
    imgcube[ind] = ndimage.shift(img ,shift_value)
    shift_value = a/2-p[0], a/2-p[1]
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
"""
程式所需時間：
start=time.time()
end = time.time()
elapsed = end - start
print("Time taken: ", elapsed, "seconds.")
"""
