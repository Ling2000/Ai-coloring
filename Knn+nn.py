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
import skimage.color
from PIL import Image
from sklearn.neighbors import KDTree


def ReLu(x):
    return max(x,0)

def dRelu(x):
    if x>0:
        return 1
        pass
    if x<=0:
        return 0

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

        r,g,b=x
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




def noden(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,p):
    net=p[0]+p[1]*x1+p[2]*x2+p[3]*x3+p[4]*x4+p[5]*x5+p[6]*x6+p[7]*x7+p[8]*x8+p[9]*x9
    return ReLu(net)



def nodes(x0,x1,x2,x3,x4,x5,p):
    net=p[0]+p[1]*x1+p[2]*x2+p[3]*x3+p[4]*x4+p[5]*x5
    return ReLu(net)




def eNN(bw, km_a,w,h,co):
    bw_array = np.array(bw)/255
    km = np.array(km_a)
    kmc=km.copy()/255
    #srcArray = np.asarray(km_a)/255
    #km = skimage.color.rgb2lab(srcArray)
    #kmc=km.copy()/128
    fla=0
    a=1
    zuobianzon=[]
    tag={}
    while a<w/2-1 :
        b=1
        while b<h-1:
            le=()
            m=-1
            while m<2 :
                n=-1
                while n<2:
                    le=le+(bw_array[b+m,a+n],)
                    n=n+1
                m=m+1
            lie=list(le)
            zuobianzon.append(lie)
            tag[le]=kmc[b,a]
            b=b+1
        a=a+1
    i=0

    w1=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    w2=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    w3=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    w4=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    w5=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

    wr=[0.15,0.15,0.15,0.15,0.15,0.15]
    wg=[0.15,0.15,0.15,0.15,0.15,0.15]
    wb=[0.15,0.15,0.15,0.15,0.15,0.15]

    #for zb in zuobianzon:
    while i<50000:#train
        zbi=random.random()*len(zuobianzon)
        zbi=int(zbi)
        zb=zuobianzon.pop(zbi)
        colort=tag[tuple(zb)]

        f1=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w1)
        f2=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w2)
        f3=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w3)
        f4=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w4)
        f5=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w5)

        r=nodes(1,f1,f2,f3,f4,f5,wr)
        g=nodes(1,f1,f2,f3,f4,f5,wg)
        b=nodes(1,f1,f2,f3,f4,f5,wb)

        #Backpropagation
        step=0.01
        lam=0.0003#parament of the Regularization
        print(colort[0]-r)#*(colort[0]-r))
        er=-2*(colort[0]-r)/3
        eg=-2*(colort[1]-g)/3
        eb=-2*(colort[2]-b)/3

        re0=step*er*(dRelu(r))*1
        re1=step*er*(dRelu(r))*f1
        re2=step*er*(dRelu(r))*f2
        re3=step*er*(dRelu(r))*f3
        re4=step*er*(dRelu(r))*f4
        re5=step*er*(dRelu(r))*f5
        re=[re0,re1,re2,re3,re4,re5]

        ge0=step*eg*(dRelu(g))*1
        ge1=step*eg*(dRelu(g))*f1
        ge2=step*eg*(dRelu(g))*f2
        ge3=step*eg*(dRelu(g))*f3
        ge4=step*eg*(dRelu(g))*f4
        ge5=step*eg*(dRelu(g))*f5
        ge=[ge0,ge1,ge2,ge3,ge4,ge5]

        be0=step*eb*(dRelu(b))*1
        be1=step*eb*(dRelu(b))*f1
        be2=step*eb*(dRelu(b))*f2
        be3=step*eb*(dRelu(b))*f3
        be4=step*eb*(dRelu(b))*f4
        be5=step*eb*(dRelu(b))*f5
        be=[be0,be1,be2,be3,be4,be5]

        swe0=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(f1)*1
        swe1=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(f1)*zb[0]
        swe2=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(f1)*zb[1]
        swe3=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(f1)*zb[2]
        swe4=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(f1)*zb[3]
        swe5=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(f1)*zb[4]
        swe6=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(f1)*zb[5]
        swe7=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(f1)*zb[6]
        swe8=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(f1)*zb[7]
        swe9=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(f1)*zb[8]
        we1=[swe0,swe1,swe2,swe3,swe4,swe5,swe6,swe7,swe8,swe9]

        swe0=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(f2)*1
        swe1=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(f2)*zb[0]
        swe2=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(f2)*zb[1]
        swe3=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(f2)*zb[2]
        swe4=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(f2)*zb[3]
        swe5=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(f2)*zb[4]
        swe6=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(f2)*zb[5]
        swe7=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(f2)*zb[6]
        swe8=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(f2)*zb[7]
        swe9=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(f2)*zb[8]
        we2=[swe0,swe1,swe2,swe3,swe4,swe5,swe6,swe7,swe8,swe9]

        swe0=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(f3)*1
        swe1=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(f3)*zb[0]
        swe2=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(f3)*zb[1]
        swe3=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(f3)*zb[2]
        swe4=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(f3)*zb[3]
        swe5=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(f3)*zb[4]
        swe6=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(f3)*zb[5]
        swe7=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(f3)*zb[6]
        swe8=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(f3)*zb[7]
        swe9=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(f3)*zb[8]
        we3=[swe0,swe1,swe2,swe3,swe4,swe5,swe6,swe7,swe8,swe9]

        swe0=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(f4)*1
        swe1=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(f4)*zb[0]
        swe2=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(f4)*zb[1]
        swe3=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(f4)*zb[2]
        swe4=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(f4)*zb[3]
        swe5=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(f4)*zb[4]
        swe6=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(f4)*zb[5]
        swe7=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(f4)*zb[6]
        swe8=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(f4)*zb[7]
        swe9=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(f4)*zb[8]
        we4=[swe0,swe1,swe2,swe3,swe4,swe5,swe6,swe7,swe8,swe9]

        swe0=step*(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(f5)*1
        swe1=step*(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(f5)*zb[0]
        swe2=step*(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(f5)*zb[1]
        swe3=step*(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(f5)*zb[2]
        swe4=step*(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(f5)*zb[3]
        swe5=step*(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(f5)*zb[4]
        swe6=step*(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(f5)*zb[5]
        swe7=step*(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(f5)*zb[6]
        swe8=step*(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(f5)*zb[7]
        swe9=step*(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(f5)*zb[8]
        we5=[swe0,swe1,swe2,swe3,swe4,swe5,swe6,swe7,swe8,swe9]



        w1=np.mat(w1)-np.mat(we1)-step*lam*np.mat(w1)#l1 Regularization
        w1=w1.tolist()
        w1=w1[0]
        w2=np.mat(w2)-np.mat(we2)-step*lam*np.mat(w2)
        w2=w2.tolist()
        w2=w2[0]
        w3=np.mat(w3)-np.mat(we3)-step*lam*np.mat(w3)
        w3=w3.tolist()
        w3=w3[0]
        w4=np.mat(w4)-np.mat(we4)-step*lam*np.mat(w4)
        w4=w4.tolist()
        w4=w4[0]
        w5=np.mat(w5)-np.mat(we5)-step*lam*np.mat(w5)
        w5=w5.tolist()
        w5=w5[0]

        wr=np.mat(wr)-np.mat(re)-step*lam*np.mat(wr)
        wr=wr.tolist()
        wr=wr[0]
        wg=np.mat(wg)-np.mat(ge)-step*lam*np.mat(wg)
        wg=wg.tolist()
        wg=wg[0]
        wb=np.mat(wb)-np.mat(be)-step*lam*np.mat(wb)
        wb=wb.tolist()
        wb=wb[0]
        i=i+1
        #if i==10000:
        #    break;
        #    pass
        pass

    print("train finish")
    print(w1)
    print(w2)
    print(w3)
    print(w4)
    print(w5)

    print(wr)
    print(wg)
    print(wb)

    i=int(w/2)
    while i<w-1:
        j=1
        while j<h-1:
            ri=()
            m=-1
            while m<2 :
                n=-1
                while n<2:
                    ri=ri+(bw_array[j+m,i+n],)
                    n=n+1
                m=m+1
            zb=list(ri)
            #print(zb)
            f1=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w1)
            f2=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w2)
            f3=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w3)
            f4=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w4)
            f5=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w5)

            r=nodes(1,f1,f2,f3,f4,f5,wr)*255
            g=nodes(1,f1,f2,f3,f4,f5,wg)*255
            b=nodes(1,f1,f2,f3,f4,f5,wb)*255
            #print(r)
            #print((r,g,b))
            #hui=km[j,i]
            #print([hui[0],g,b])
            #km[j,i]=np.matrix([r,g,b])
            coo=co[0]
            cd1=math.sqrt((coo[0]-r)*(coo[0]-r)+(coo[1]-g)*(coo[1]-g)+(coo[2]-b)*(coo[2]-b))
            coo=co[1]
            cd2=math.sqrt((coo[0]-r)*(coo[0]-r)+(coo[1]-g)*(coo[1]-g)+(coo[2]-b)*(coo[2]-b))
            coo=co[2]
            cd3=math.sqrt((coo[0]-r)*(coo[0]-r)+(coo[1]-g)*(coo[1]-g)+(coo[2]-b)*(coo[2]-b))
            coo=co[3]
            cd4=math.sqrt((coo[0]-r)*(coo[0]-r)+(coo[1]-g)*(coo[1]-g)+(coo[2]-b)*(coo[2]-b))
            coo=co[4]
            cd5=math.sqrt((coo[0]-r)*(coo[0]-r)+(coo[1]-g)*(coo[1]-g)+(coo[2]-b)*(coo[2]-b))
            zas=[]
            zas.append(cd1)
            zas.append(cd2)
            zas.append(cd3)
            zas.append(cd4)
            zas.append(cd5)
            zas.sort()

            if zas[0]==cd1:
                km[j,i]=np.matrix(co[0])
                pass
            elif zas[0]==cd2:
                km[j,i]=np.matrix(co[1])
                pass
            elif zas[0]==cd3:
                km[j,i]=np.matrix(co[2])
                pass
            elif zas[0]==cd4:
                km[j,i]=np.matrix(co[3])
                pass
            elif zas[0]==cd5:
                km[j,i]=np.matrix(co[4])
                pass
            j=j+1
        i=i+1

        #end = skimage.color.lab2rgb(km)*255
        #end = end.astype(np.uint8)
        ima = Image.fromarray(np.uint8(km))
    return ima






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
    im = Image.open('微信图片_20210527120123.jpg')
    pix = im.load()
    width = im.size[0]
    height = im.size[1]
    poto=np.zeros((height,width,3))
    print(width)
    print(height)
    for x in range(width):
        for y in range(height):
            r, g, b= pix[x, y]
            poto[y,x,0]=r
            poto[y,x,1]=g
            poto[y,x,2]=b
    img2 = Image.fromarray(np.uint8(poto))

    zb=[(70, 95, 52), (80, 145, 212), (206, 242, 236), (128, 161, 137), (147, 196, 217)]
    zb=kmeans(zb,poto,width,height)
    #print(zb)
    imz=img2.copy()
    imgbw=BW(imz,width,height)
    imgbw.save("bw.png","png")
    #img=recolor(zb, imz ,width,height)
    #img.save("5mean.png","png")
    imgtest=eNN(imgbw, img2,width,height,zb)
    imgtest.show()
    imgtest.save("final.png","png")
    print("Done.")
