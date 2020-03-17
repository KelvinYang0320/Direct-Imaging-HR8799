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
foo=imgcube[0,:,:]
plt.ion()
plt.figure(0)
plt.title("the original image")
plt.imshow(foo, cmap="gray")
plt.show()
plt.pause(3)

def Center_of_Gravity(img):
    centroid_x = np.arange(img_size[1]).reshape(1, -1) * img
    centroid_y = np.arange(img_size[0]).reshape(-1, 1) * img
    centroid_x = centroid_x.sum()
    centroid_y = centroid_y.sum()
    return centroid_x / img.sum(), centroid_y / img.sum()
centroid = Center_of_Gravity(foo)
x=round(centroid[0])
y=round(centroid[1])
print("center(",x,",",y,")")
"""
print("center(x,y)")
x=input("x=")
y=input("y=")
print("center(x,y)=","(",x,",",y,")")
x=round(float(x))
y=round(float(y))
"""
sure='N'
while sure!='Y':
    a=input("Searching area a x a ,a=")
    a=int(a)
    print("Searching area:",a,"x",a)
    print(a*a,"pixels")
    foo_shift=foo[int(round(y-a/2)):int(round(y+a/2)), int(round(x-a/2)):int(round(x+a/2))]
    plt.ion()
    plt.figure(1)
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
Check=np.zeros((search,search))
for i in range(search):
    for j in range(search):
        x_check=x-search/2+i
        y_check=y-search/2+j
        foo_shift_check=foo[int(round(y_check-search/2)):int(round(y_check+search/2)), int(round(x_check-search/2)):int(round(x_check+search/2))]
        foo_shift_check_rot=np.rot90(np.rot90(foo_shift_check))
        Check[j,i]=np.sum(np.absolute(foo_shift_check-foo_shift_check_rot))
index_min=np.argmin(Check)
index_min_y=index_min//search
index_min_x=index_min%search
plt.ion()
plt.figure(2)
plt.clf()
plt.imshow(Check, cmap="jet")
plt.colorbar()
plt.scatter(index_min_x, index_min_y, c="r", marker="x" )
plt.show()
plt.pause(1)
plt.ion()
plt.figure(3)
plt.clf()
plt.title("the shifted image")
plt.imshow(foo_shift, cmap="gray")
plt.scatter((a-search)/2+index_min_x, (a-search)/2+index_min_y, c="r", marker="x")
plt.show()
plt.pause(1)


def Center(img,search,a):
    Check=np.zeros((search,search))
    centroid = Center_of_Gravity(img)
    x=round(centroid[0])
    y=round(centroid[1])
    for i in range(search):
        x_check=x-search/2+i
        for j in range(search):
            y_check=y-search/2+j
            foo_shift_check=img[int(round(y_check-search/2)):int(round(y_check+search/2)), int(round(x_check-search/2)):int(round(x_check+search/2))]
            foo_shift_check_rot=np.rot90(np.rot90(foo_shift_check))
            Check[j,i]=np.sum(np.absolute(foo_shift_check-foo_shift_check_rot))
    index_min=np.argmin(Check)
    index_min_y=index_min//search
    index_min_x=index_min%search
    X=(x-a/2)+((a-search)/2+index_min_x)
    Y=(y-a/2)+((a-search)/2+index_min_y)
    return X, Y   

plt.figure()
for ind, img in enumerate(imgcube):
    """
    center=x,y
    """
    center = Center(img, search ,a)
    shift_value = 512-center[1], 512-center[0]
    imgcube[ind] = ndimage.shift(img ,shift_value)
    ind_0=ind+1
    print(ind_0,"/136")
    plt.imshow(imgcube[ind],cmap="gray")
    plt.scatter(512, 512, c="r", marker="x" )
    plt.show()
    plt.pause(1)    
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
end=time.time()   
print ("Derotation and Combination done")
elapsed = end - start
print ("Time taken: ", elapsed, "seconds.")

plt.ion()
plt.figure(figsize = (6, 6))
plt.imshow(img_cADI, cmap = "gray")
plt.xlim(344, 680)
plt.ylim(344, 680)
plt.show()
plt.pause(10)

#start=time.time()

#end=time.time()
#elapsed = end - start
#print ("Time taken: ", elapsed, "seconds.")
