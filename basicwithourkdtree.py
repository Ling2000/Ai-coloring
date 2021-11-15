import threading
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import math
import imageio
import itertools
import numpy.ma as ma
from PIL import Image

def BW(picture,w,h):
    img_array = np.array(picture)
    poto=np.zeros((h,w))
    for x in range(w):
        for y in range(h):
            r1, g1, b1 = img_array[y,x]
            poto[y,x]= 0.21*r1 + 0.72*g1 + 0.07*b1
    img2 = Image.fromarray(np.uint8(poto))
    return img2

def takep(elem):
    return elem[-1]



def coooo(zb,cl):
    i=0
    for x in cl:
        q,z,h=x
        r,g,b=z
        r1,g1,b1=zb
        if r==r1 and g1==g and b1==b:
            i=i+1
            pass
    return i
def dis(a,b):
    c=a-b
    i=0
    dis=0
    while i<3 :
        j=0
        while j<3:
            dis=dis+c[i,j]*c[i,j]
            j=j+1
        i=i+1
    return math.sqrt(dis)

def basicAgent(bw, km_a,w,h,zb):
    bw_array = np.array(bw)
    km = np.array(km_a)
    kmc=km.copy()
    print(len(kmc))
    fla=0
    i=101
    while i<102:
        j=1
        while j<h-1:
            ri=np.matlib.zeros((3,3))
            m=-1
            while m<2 :
                n=-1
                while n<2:
                    ri[m+1,n+1]=bw_array[j+m,i+n]
                    n=n+1
                m=m+1
            si=[]
            a=1
            while a<w/2-1 :
                b=1
                while b<h-1:
                    le=np.matlib.zeros((3,3))
                    m=-1
                    while m<2 :
                        n=-1
                        while n<2:
                            le[m+1,n+1]=bw_array[b+m,a+n]
                            n=n+1
                        m=m+1
                    dist=dis(ri,le)
                    if len(si)>=6:
                        if fla==0:
                            si.sort(key=takep,reverse=False)
                            fla=1
                            pass
                        zz,zzz,di=si[5]
                        if dist<di:
                            si.append((le,km[b,a],dist))
                    else:
                        si.append((le,km[b,a],dist))

                    b=b+1
                a=a+1
            si.sort(key=takep,reverse=False)
            si=si[0:5]
            print(si)
            cl=[]
            cl0=coooo(zb[0],si)
            cl.append(("cl0",cl0/6))
            cl1=coooo(zb[1],si)
            cl.append(("cl1",cl1/6))
            cl2=coooo(zb[2],si)
            cl.append(("cl2",cl2/6))
            cl3=coooo(zb[3],si)
            cl.append(("cl3",cl3/6))
            cl4=coooo(zb[4],si)
            cl.append(("cl4",cl4/6))
            cl.sort(key=takep,reverse=True)
            print(cl)
            na,cee=cl[0]
            if na=="cl0":
                kmc[j,i]=zb[0]
            elif na=="cl1":
                kmc[j,i]=zb[1]
            elif na=="cl2":
                kmc[j,i]=zb[2]
            elif na=="cl3":
                kmc[j,i]=zb[3]
            elif na=="cl4":
                kmc[j,i]=zb[4]
            j=j+1
        i=i+1
        im = Image.fromarray(np.uint8(kmc))
    return im


def recolor(zb,picture,w,h):
    img_array = np.array(picture)
    for x in range(w):
        for y in range(h):
            r1, g1, b1 = img_array[y,x]
            r,g,b=zb[0]
            cd1=math.sqrt((r1-r)*(r1-r)+(g1-g)*(g1-g)+(b1-b)*(b1-b))
            r,g,b=zb[1]
            cd2=math.sqrt((r1-r)*(r1-r)+(g1-g)*(g1-g)+(b1-b)*(b1-b))
            r,g,b=zb[2]
            cd3=math.sqrt((r1-r)*(r1-r)+(g1-g)*(g1-g)+(b1-b)*(b1-b))
            r,g,b=zb[3]
            cd4=math.sqrt((r1-r)*(r1-r)+(g1-g)*(g1-g)+(b1-b)*(b1-b))
            r,g,b=zb[4]
            cd5=math.sqrt((r1-r)*(r1-r)+(g1-g)*(g1-g)+(b1-b)*(b1-b))
            azs=[]
            azs.append(cd1)
            azs.append(cd2)
            azs.append(cd3)
            azs.append(cd4)
            azs.append(cd5)
            azs.sort()
            if azs[0]==cd1:
                r,g,b=zb[0]
                img_array[y,x]=(r,g,b)
                pass
            elif azs[0]==cd2:
                r,g,b=zb[1]
                img_array[y,x]=(r,g,b)
                pass
            elif azs[0]==cd3:
                r,g,b=zb[2]
                img_array[y,x]=(r,g,b)
                pass
            elif azs[0]==cd4:
                r,g,b=zb[3]
                img_array[y,x]=(r,g,b)
                pass
            elif azs[0]==cd5:
                r,g,b=zb[4]
                img_array[y,x]=(r,g,b)
                pass
        img2 = Image.fromarray(np.uint8(img_array))
    return img2


def mean(cl):
    if len(cl)==0:
        ran1=random.random()
        ran2=random.random()
        ran3=random.random()
        return (int(255*ran1),int(255*ran2),int(255*ran3))#(float(int(255*ran1)),float(int(255*ran2)),float(int(255*ran3)))
        pass
    avgr=0
    avgg=0
    avgb=0
    i=0
    while i<len(cl):
        r,g,b=cl[i]
        avgr=avgr+r
        avgg=avgg+g
        avgb=avgb+b
        i=i+1
    return (int(avgr/len(cl)),int(avgg/len(cl)),int(avgb/len(cl)))#(float(int(avgr/len(cl))),float(int(avgg/len(cl))),float(int(avgb/len(cl))))


def kmeans(zb,picture,w,h):
    cl1=[]
    cl2=[]
    cl3=[]
    cl4=[]
    cl5=[]
    for x in range(w):
        for y in range(h):
            r,g,b=zb[0]
            cd1=math.sqrt((picture[y,x,0]-r)*(picture[y,x,0]-r)+(picture[y,x,1]-g)*(picture[y,x,1]-g)+(picture[y,x,2]-b)*(picture[y,x,2]-b))
            r,g,b=zb[1]
            cd2=math.sqrt((picture[y,x,0]-r)*(picture[y,x,0]-r)+(picture[y,x,1]-g)*(picture[y,x,1]-g)+(picture[y,x,2]-b)*(picture[y,x,2]-b))
            r,g,b=zb[2]
            cd3=math.sqrt((picture[y,x,0]-r)*(picture[y,x,0]-r)+(picture[y,x,1]-g)*(picture[y,x,1]-g)+(picture[y,x,2]-b)*(picture[y,x,2]-b))
            r,g,b=zb[3]
            cd4=math.sqrt((picture[y,x,0]-r)*(picture[y,x,0]-r)+(picture[y,x,1]-g)*(picture[y,x,1]-g)+(picture[y,x,2]-b)*(picture[y,x,2]-b))
            r,g,b=zb[4]
            cd5=math.sqrt((picture[y,x,0]-r)*(picture[y,x,0]-r)+(picture[y,x,1]-g)*(picture[y,x,1]-g)+(picture[y,x,2]-b)*(picture[y,x,2]-b))
            zas=[]
            zas.append(cd1)
            zas.append(cd2)
            zas.append(cd3)
            zas.append(cd4)
            zas.append(cd5)
            zas.sort()

            r=picture[y,x,0]
            g=picture[y,x,1]
            b=picture[y,x,2]

            if zas[0]==cd1:
                cl1.append((r,g,b))
                pass
            elif zas[0]==cd2:
                cl2.append((r,g,b))
                pass
            elif zas[0]==cd3:
                cl3.append((r,g,b))
                pass
            elif zas[0]==cd4:
                cl4.append((r,g,b))
                pass
            elif zas[0]==cd5:
                cl5.append((r,g,b))
                pass
    zb1=mean(cl1)
    zb2=mean(cl2)
    zb3=mean(cl3)
    zb4=mean(cl4)
    zb5=mean(cl5)
    if zb1==zb[0] and zb2==zb[1] and zb3==zb[2] and zb4==zb[3] and zb5==zb[4]:

        return zb
        pass
    else:
        arr=[]
        arr.append(zb1)
        arr.append(zb2)
        arr.append(zb3)
        arr.append(zb4)
        arr.append(zb5)
        return kmeans(arr,picture,w,h)




if __name__ == "__main__":
    im = Image.open('c5f0e38c65398aadd31181d0570fac26.jpeg')
    pix = im.load()
    width = im.size[0]
    height = im.size[1]
    poto=np.zeros((height,width,3))
    print(width)
    print(height)
    for x in range(width):
        for y in range(height):
            r, g, b = pix[x, y]
            poto[y,x,0]=r
            poto[y,x,1]=g
            poto[y,x,2]=b
    img2 = Image.fromarray(np.uint8(poto))
    #img2.show()
    img2.save("yuan.png","png")
    zb=[(4, 3, 3), (162, 44, 45), (205, 158, 156), (251, 250, 249), (241, 5, 6)]
    zb=kmeans(zb,poto,width,height)
    print(zb)
    imz=im.copy()
    imgbw=BW(imz,width,height)
    #imgbw.show()
    imgbw.save("bw.png","png")
    img=recolor(zb, imz ,width,height)
    #img.show()
    img.save("5mean.png","png")
    imgtest=basicAgent(imgbw, img,width,height,zb)
    imgtest.show()
    imgtest.save("final.png","png")
    print("Done.")
