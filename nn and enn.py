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
from sklearn.neighbors import KDTree
from PIL import ImageCms



def ReLu(x):
    return max(x,0)

def dRelu(x):
    if x>0:
        return 1
        pass
    if x<=0:
        return 0

def node(xf,xs,p):
    net=p[0]*xf+p[1]*xs
    return ReLu(net)

def nodef(xy,xe,xs,xf,p):
    net=p[0]*xy+p[1]*xe+p[2]*xs+p[3]*xf
    return ReLu(net)

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

def half(m):
    m[0]=int(m[0]/4)
    m[1]=int(m[1]/4)
    m[2]=int(m[2]/4)
    return m

def double(m):
    m[0]=int(m[0]*4)
    m[1]=int(m[1]*4)
    m[2]=int(m[2]*4)
    return m


def noden(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,p):

    net=p[0]+p[1]*x1+p[2]*x2+p[3]*x3+p[4]*x4+p[5]*x5+p[6]*x6+p[7]*x7+p[8]*x8+p[9]*x9
    return ReLu(net)

def nodenn(x,zb,p):
    net=p[0]+p[1]*zb[0]+p[2]*zb[1]+p[3]*zb[2]+p[4]*zb[3]+p[5]*zb[4]+p[6]*zb[5]+p[7]*zb[6]+p[8]*zb[7]+p[9]*zb[8]
    return ReLu(net)

def nodes(x0,x1,x2,x3,x4,x5,p):
    net=p[0]+p[1]*x1+p[2]*x2+p[3]*x3+p[4]*x4+p[5]*x5
    return ReLu(net)

def nodess(x0,x1,x2,x3,x4,x5,p):
    net=p[0]+p[1]*x1+p[2]*x2+p[3]*x3+p[4]*x4+p[5]*x5
    return ReLu(net)


def fNN(bw, km_a,w,h):
    #bw_array = np.array(bw)
    km = np.array(km_a)
    i=0
    while i<w:
        j=0
        while j<h:
            km[j,i]=rgb2lab(km[j,i])
            pass
            j=j+1
        i=i+1
        pass
    kmc=km.copy()/128
    print(len(kmc))
    fla=0
    a=7
    zuobianzon=[]
    tag={}
    while a<w/2-7 :
        b=7
        while b<h-7:
            le=()
            m=-7
            while m<8 :
                n=-7
                while n<8:
                    if b+m<0 or a+n<0 or a+n>w/2-1 or b+m>h-1:
                        le=le+(0,)
                        pass
                    else:
                        le=le+(km[b+m,a+n][0],)
                    n=n+1
                m=m+1
            lie=list(le)
            zuobianzon.append(lie)
            tag[le]=kmc[b,a][1:]
            b=b+1
        a=a+1
    i=0
    w1=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w2=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w3=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w4=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w5=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w6=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w7=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w8=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w9=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w10=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w11=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w12=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w13=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w14=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w15=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w16=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w17=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w18=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w19=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w20=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w21=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w22=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w23=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w24=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    w25=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]

    sw1=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    sw2=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    sw3=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    sw4=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]
    sw5=[(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2),(random.random()*2-1)*math.sqrt(1/2)]


    wr=[(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7)]
    wg=[(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7)]
    wb=[(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7),(random.random()*2-1)*math.sqrt(6/7)]

    print("ini")
    #for zb in zuobianzon:
    while i<1000:#train
        zbi=random.random()*len(zuobianzon)
        zbi=int(zbi)
        zb=zuobianzon.pop(zbi)
        colort=tag[tuple(zb)]

        f0=nodenn(1,zb[0:9],w1)
        f1=nodenn(1,zb[9:18],w2)
        f2=nodenn(1,zb[18:27],w3)
        f3=nodenn(1,zb[27:36],w4)
        f4=nodenn(1,zb[36:45],w5)

        f5=nodenn(1,zb[45:54],w6)
        f6=nodenn(1,zb[54:63],w7)
        f7=nodenn(1,zb[63:72],w8)
        f8=nodenn(1,zb[72:81],w9)
        f9=nodenn(1,zb[81:90],w10)

        f10=nodenn(1,zb[90:99],w11)
        f11=nodenn(1,zb[99:108],w12)
        f12=nodenn(1,zb[108:117],w13)
        f13=nodenn(1,zb[117:126],w14)
        f14=nodenn(1,zb[126:135],w15)

        f15=nodenn(1,zb[135:144],w16)
        f16=nodenn(1,zb[144:153],w17)
        f17=nodenn(1,zb[153:162],w18)
        f18=nodenn(1,zb[162:171],w19)
        f19=nodenn(1,zb[171:180],w20)

        f20=nodenn(1,zb[180:189],w21)
        f21=nodenn(1,zb[189:198],w22)
        f22=nodenn(1,zb[198:207],w23)
        f23=nodenn(1,zb[207:216],w24)
        f24=nodenn(1,zb[216:],w25)

        s1=[f0,f1,f2,f5,f6,f7,f10,f11,f12]
        s2=[f2,f3,f4,f7,f8,f9,f12,f13,f14]
        s3=[f10,f11,f12,f15,f16,f17,f20,f21,f22]
        s4=[f12,f13,f14,f17,f18,f19,f22,f23,f24]
        s5=[f6,f7,f8,f11,f12,f13,f16,f17,f18]

        t1=nodenn(1,s1,sw1)
        t2=nodenn(1,s2,sw2)
        t3=nodenn(1,s3,sw3)
        t4=nodenn(1,s4,sw4)
        t5=nodenn(1,s5,sw4)

        r=nodes(1,t1,t2,t3,t4,t5,wr)
        g=nodes(1,t1,t2,t3,t4,t5,wg)
        b=nodes(1,t1,t2,t3,t4,t5,wb)

        #Backpropagation
        step=0.00000006
        lam=0.0000003
        #print(colort[0]-g)
        #print((r,g,b))
        er=0#-2*(color[0]-r)/3
        eg=-2*(colort[0]-g)/3
        eb=-2*(colort[1]-b)/3

        re0=step*er*(dRelu(r))*1
        re1=step*er*(dRelu(r))*t1
        re2=step*er*(dRelu(r))*t2
        re3=step*er*(dRelu(r))*t3
        re4=step*er*(dRelu(r))*t4
        re5=step*er*(dRelu(r))*t5
        re=[re0,re1,re2,re3,re4,re5]

        ge0=step*eg*(dRelu(g))*1
        ge1=step*eg*(dRelu(g))*t1
        ge2=step*eg*(dRelu(g))*t2
        ge3=step*eg*(dRelu(g))*t3
        ge4=step*eg*(dRelu(g))*t4
        ge5=step*eg*(dRelu(g))*t5
        ge=[ge0,ge1,ge2,ge3,ge4,ge5]

        be0=step*eb*(dRelu(b))*1
        be1=step*eb*(dRelu(b))*t1
        be2=step*eb*(dRelu(b))*t2
        be3=step*eb*(dRelu(b))*t3
        be4=step*eb*(dRelu(b))*t4
        be5=step*eb*(dRelu(b))*t5
        be=[be0,be1,be2,be3,be4,be5]


        swe1=[1]
        swe1=swe1+s1
        we1=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(t1)*1*np.mat(swe1)

        swe1=[1]
        swe1=swe1+s2
        we2=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(t2)*1*np.mat(swe1)

        swe1=[1]
        swe1=swe1+s3
        we3=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(t3)*1*np.mat(swe1)

        swe1=[1]
        swe1=swe1+s4
        we4=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(t4)*1*np.mat(swe1)

        swe1=[1]
        swe1=swe1+s5
        we5=step*(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(t5)*1*np.mat(swe1)



        swe1=[1]
        swe1=swe1+zb[0:9]
        e1=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(t1)*sw1[1]*dRelu(f0)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[9:18]
        e2=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(t1)*sw1[2]*dRelu(f1)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[18:27]
        e3=step*((er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(t1)*sw1[3]+(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(t2)*sw2[1])*dRelu(f2)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[27:36]
        e4=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(t2)*sw2[2]*dRelu(f3)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[36:45]
        e5=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(t2)*sw2[3]*dRelu(f4)*np.mat(swe1)




        swe1=[1]
        swe1=swe1+zb[45:54]
        e6=step*(er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(t1)*sw1[4]*dRelu(f5)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[54:63]
        e7=step*((er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(t1)*sw1[5]+(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(t5)*sw5[1])*dRelu(f6)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[63:72]
        e8=step*((er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(t1)*sw1[6]+(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(t1)*sw2[4]+(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(t5)*sw5[2])*dRelu(f7)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[72:81]
        e9=step*((er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(t1)*sw2[5]+(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(t5)*sw5[3])*dRelu(f8)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[81:90]
        e10=step*(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(t1)*sw2[6]*dRelu(f9)*np.mat(swe1)




        swe1=[1]
        swe1=swe1+zb[90:99]
        e11=step*((er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(t1)*sw1[7]+(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(t3)*sw3[1])*dRelu(f10)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[99:108]
        e12=step*((er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(t1)*sw1[8]+(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(t3)*sw3[2]+(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(t5)*sw5[4])*dRelu(f11)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[108:117]
        e13=step*((er*(dRelu(r))*wr[1]+eg*(dRelu(g))*wg[1]+eb*(dRelu(b))*wb[1])*dRelu(t1)*sw1[9]+(er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(t2)*sw2[7]+(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(t3)*sw3[3]+(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(t4)*sw4[1]+(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(t5)*sw5[5])*dRelu(f12)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[117:126]
        e14=step*((er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(t1)*sw2[8]+(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(t4)*sw4[2]+(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(t5)*sw5[6])*dRelu(f13)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[126:135]
        e15=step*((er*(dRelu(r))*wr[2]+eg*(dRelu(g))*wg[2]+eb*(dRelu(b))*wb[2])*dRelu(t1)*sw2[9]+(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(t4)*sw4[3])*dRelu(f14)*np.mat(swe1)




        swe1=[1]
        swe1=swe1+zb[135:144]
        e16=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(t3)*sw3[4]*dRelu(f15)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[144:153]
        e17=step*((er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(t3)*sw3[5]+(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(t5)*sw5[7])*dRelu(f16)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[153:162]
        e18=step*((er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(t3)*sw3[6]+(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(t4)*sw4[4]+(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(t5)*sw5[8])*dRelu(f17)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[162:171]
        e19=step*((er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(t4)*sw4[5]+(er*(dRelu(r))*wr[5]+eg*(dRelu(g))*wg[5]+eb*(dRelu(b))*wb[5])*dRelu(t5)*sw5[9])*dRelu(f18)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[171:180]
        e20=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(t4)*sw4[6]*dRelu(f19)*np.mat(swe1)



        swe1=[1]
        swe1=swe1+zb[180:189]
        e21=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(t3)*sw3[7]*dRelu(f20)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[189:198]
        e22=step*(er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(t3)*sw3[8]*dRelu(f21)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[198:207]
        e23=step*((er*(dRelu(r))*wr[3]+eg*(dRelu(g))*wg[3]+eb*(dRelu(b))*wb[3])*dRelu(t3)*sw3[9]+(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(t4)*sw4[7])*dRelu(f22)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[207:216]
        e24=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(t4)*sw4[8]*dRelu(f23)*np.mat(swe1)

        swe1=[1]
        swe1=swe1+zb[216:]
        e25=step*(er*(dRelu(r))*wr[4]+eg*(dRelu(g))*wg[4]+eb*(dRelu(b))*wb[4])*dRelu(t4)*sw4[9]*dRelu(f24)*np.mat(swe1)



        w1=np.mat(w1)-e1-step*lam*np.mat(w1)
        w1=w1.tolist()
        w1=w1[0]
        w2=np.mat(w2)-e2-step*lam*np.mat(w2)
        w2=w2.tolist()
        w2=w2[0]
        w3=np.mat(w3)-e3-step*lam*np.mat(w3)
        w3=w3.tolist()
        w3=w3[0]
        w4=np.mat(w4)-e4-step*lam*np.mat(w4)
        w4=w4.tolist()
        w4=w4[0]
        w5=np.mat(w5)-e5-step*lam*np.mat(w5)
        w5=w5.tolist()
        w5=w5[0]

        w6=np.mat(w6)-e6-step*lam*np.mat(w6)
        w6=w6.tolist()
        w6=w6[0]
        w7=np.mat(w7)-e7-step*lam*np.mat(w7)
        w7=w7.tolist()
        w7=w7[0]
        w8=np.mat(w8)-e8-step*lam*np.mat(w8)
        w8=w8.tolist()
        w8=w8[0]
        w9=np.mat(w9)-e9-step*lam*np.mat(w9)
        w9=w9.tolist()
        w9=w9[0]
        w10=np.mat(w10)-e10-step*lam*np.mat(w10)
        w10=w10.tolist()
        w10=w10[0]

        w11=np.mat(w11)-e11-step*lam*np.mat(w11)
        w11=w11.tolist()
        w11=w11[0]
        w12=np.mat(w12)-e12-step*lam*np.mat(w12)
        w12=w12.tolist()
        w12=w12[0]
        w13=np.mat(w13)-e13-step*lam*np.mat(w13)
        w13=w13.tolist()
        w13=w13[0]
        w14=np.mat(w14)-e14-step*lam*np.mat(w14)
        w14=w14.tolist()
        w14=w14[0]
        w15=np.mat(w15)-e15-step*lam*np.mat(w15)
        w15=w15.tolist()
        w15=w15[0]

        w16=np.mat(w16)-e16-step*lam*np.mat(w16)
        w16=w16.tolist()
        w16=w16[0]
        w17=np.mat(w17)-e17-step*lam*np.mat(w17)
        w17=w17.tolist()
        w17=w17[0]
        w18=np.mat(w18)-e18-step*lam*np.mat(w18)
        w18=w18.tolist()
        w18=w18[0]
        w19=np.mat(w19)-e19-step*lam*np.mat(w19)
        w19=w19.tolist()
        w19=w19[0]
        w20=np.mat(w20)-e20-step*lam*np.mat(w20)
        w20=w20.tolist()
        w20=w20[0]

        w21=np.mat(w21)-e21-step*lam*np.mat(w21)
        w21=w21.tolist()
        w21=w21[0]
        w22=np.mat(w22)-e22-step*lam*np.mat(w22)
        w22=w22.tolist()
        w22=w22[0]
        w23=np.mat(w23)-e23-step*lam*np.mat(w23)
        w23=w23.tolist()
        w23=w23[0]
        w24=np.mat(w24)-e24-step*lam*np.mat(w24)
        w24=w24.tolist()
        w24=w24[0]
        w25=np.mat(w25)-e25-step*lam*np.mat(w25)
        w25=w25.tolist()
        w25=w25[0]

        sw1=np.mat(sw1)-we1-step*lam*np.mat(sw1)
        sw1=sw1.tolist()
        sw1=sw1[0]
        sw2=np.mat(sw2)-we2-step*lam*np.mat(sw2)
        sw2=sw2.tolist()
        sw2=sw2[0]
        sw3=np.mat(sw3)-we3-step*lam*np.mat(sw3)
        sw3=sw3.tolist()
        sw3=sw3[0]
        sw4=np.mat(sw4)-we4-step*lam*np.mat(sw4)
        sw4=sw4.tolist()
        sw4=sw4[0]
        sw5=np.mat(sw5)-we5-step*lam*np.mat(sw5)
        sw5=sw5.tolist()
        sw5=sw5[0]



        wr=np.mat(wr)-re-step*lam*np.mat(wr)
        wr=wr.tolist()
        wr=wr[0]
        wg=np.mat(wg)-ge-step*lam*np.mat(wg)
        wg=wg.tolist()
        wg=wg[0]
        wb=np.mat(wb)-be-step*lam*np.mat(wb)
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
    kmc=kmc*128
    i=int(w/2)+7
    while i<w-7:
        j=7
        while j<h-7:
            ri=()
            m=-7
            while m<8 :
                n=-7
                while n<8:
                    #if j+m<0 or i+n<0 or i+n>w/2-1 or j+m>h-1:
                    #    ri=ri+(0,)
                    #    pass
                    #else:
                    hui=km[j+m,i+n]
                    ri=ri+(hui[0],)
                    n=n+1
                m=m+1
            zb=list(ri)
            #print(zb)
            f0=nodenn(1,zb[0:9],w1)
            f1=nodenn(1,zb[9:18],w2)
            f2=nodenn(1,zb[18:27],w3)
            f3=nodenn(1,zb[27:36],w4)
            f4=nodenn(1,zb[36:45],w5)

            f5=nodenn(1,zb[45:54],w6)
            f6=nodenn(1,zb[54:63],w7)
            f7=nodenn(1,zb[63:72],w8)
            f8=nodenn(1,zb[72:81],w9)
            f9=nodenn(1,zb[81:90],w10)

            f10=nodenn(1,zb[90:99],w11)
            f11=nodenn(1,zb[99:108],w12)
            f12=nodenn(1,zb[108:117],w13)
            f13=nodenn(1,zb[117:126],w14)
            f14=nodenn(1,zb[126:135],w15)

            f15=nodenn(1,zb[135:144],w16)
            f16=nodenn(1,zb[144:153],w17)
            f17=nodenn(1,zb[153:162],w18)
            f18=nodenn(1,zb[162:171],w19)
            f19=nodenn(1,zb[171:180],w20)

            f20=nodenn(1,zb[180:189],w21)
            f21=nodenn(1,zb[189:198],w22)
            f22=nodenn(1,zb[198:207],w23)
            f23=nodenn(1,zb[207:216],w24)
            f24=nodenn(1,zb[216:],w25)

            s1=[f0,f1,f2,f5,f6,f7,f10,f11,f12]
            s2=[f2,f3,f4,f7,f8,f9,f12,f13,f14]
            s3=[f10,f11,f12,f15,f16,f17,f20,f21,f22]
            s4=[f12,f13,f14,f17,f18,f19,f22,f23,f24]
            s5=[f6,f7,f8,f11,f12,f13,f16,f17,f18]

            t1=nodenn(1,s1,sw1)
            t2=nodenn(1,s2,sw2)
            t3=nodenn(1,s3,sw3)
            t4=nodenn(1,s4,sw4)
            t5=nodenn(1,s5,sw4)

            #r=nodes(1,t1,t2,t3,t4,t5,wr)*255
            g=nodes(1,t1,t2,t3,t4,t5,wg)*128
            b=nodes(1,t1,t2,t3,t4,t5,wb)*128

            #print((r,g,b))
            hui=km[j,i]
            kmc[j,i]=np.matrix([hui[0],g,b])
            j=j+1
        i=i+1

        i=0
        while i<w:
            j=0
            while j<h:
                kmc[j,i]=lab2rgb(kmc[j,i])
                pass
                j=j+1
            i=i+1
            pass

        im = Image.fromarray(np.uint8(kmc))

        #cov=np.uint8(cov)
        #cov.save("final.png","png")
    return im




def eNN(bw, km_a,w,h):
    bw_array = np.array(bw)
    km = np.array(km_a)
    kmc=km.copy()
    print(len(kmc))
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
            tag[le]=km[b,a]
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
    while i<100:#train
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
        step=0.0000001
        lam=0.0000003
        print(colort[0]-r)
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



        w1=np.mat(w1)-np.mat(we1)-step*lam*np.mat(w1)
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
        #if i==50000:
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


            f1=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w1)
            f2=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w2)
            f3=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w3)
            f4=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w4)
            f5=noden(1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8],w5)

            r=nodes(1,f1,f2,f3,f4,f5,wr)
            g=nodes(1,f1,f2,f3,f4,f5,wg)
            b=nodes(1,f1,f2,f3,f4,f5,wb)
            #print((r,g,b))

            kmc[j,i]=np.matrix([r,g,b])
            j=j+1
        i=i+1
        im = Image.fromarray(np.uint8(kmc))
    return im





def NN(bw, km_a,w,h):
    bw_array = np.array(bw)
    km = np.array(km_a)
    kmc=km.copy()
    print(len(kmc))
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
            tag[le]=km[b,a]
            b=b+1
        a=a+1
    i=0
    w1=0.5
    w2=0.5
    w3=0.5
    w4=0.5
    w5=0.5
    w6=0.5
    w7=0.5
    w8=0.5
    w9=0.5
    w10=0.5
    w11=0.5
    w12=0.5
    w13=0.5
    w14=0.5
    w15=0.5
    w16=0.5
    sw1=0.5
    sw2=0.5
    sw3=0.5
    sw4=0.5
    sw5=0.5
    sw6=0.5
    sw7=0.5
    sw8=0.5
    r1=0.25
    r2=0.25
    r3=0.25
    r4=0.25
    g1=0.25
    g2=0.25
    g3=0.25
    g4=0.25
    b1=0.25
    b2=0.25
    b3=0.25
    b4=0.25
    for zb in zuobianzon:
    #while i<5000:
        #train
        zbi=random.random()*len(zuobianzon)
        zbi=int(zbi)
        zb=zuobianzon.pop(zbi)
        colort=tag[tuple(zb)]

        f1=node(zb[4],zb[0],[w1,w2])
        f2=node(zb[4],zb[1],[w3,w4])
        f3=node(zb[4],zb[2],[w5,w6])
        f4=node(zb[4],zb[3],[w7,w8])
        f5=node(zb[4],zb[5],[w9,w10])
        f6=node(zb[4],zb[6],[w11,w12])
        f7=node(zb[4],zb[7],[w13,w14])
        f8=node(zb[4],zb[8],[w15,w16])


        s1=node(f1,f2,[sw1,sw2])
        s2=node(f3,f4,[sw3,sw4])
        s3=node(f5,f6,[sw5,sw6])
        s4=node(f7,f8,[sw7,sw8])


        r=nodef(s1,s2,s3,s4,[r1,r2,r3,r4])
        g=nodef(s1,s2,s3,s4,[g1,g2,g3,g4])
        b=nodef(s1,s2,s3,s4,[b1,b2,b3,b4])
        #Backpropagation
        step=0.000006
        lam=0.0000001
        print(colort[0]-r)
        er=-(colort[0]-r)/3
        eg=-(colort[1]-g)/3
        eb=-(colort[2]-b)/3

        re1=step*er*(dRelu(r))*s1
        re2=step*er*(dRelu(r))*s2
        re3=step*er*(dRelu(r))*s3
        re4=step*er*(dRelu(r))*s4

        ge1=step*eg*(dRelu(g))*s1
        ge2=step*eg*(dRelu(g))*s2
        ge3=step*eg*(dRelu(g))*s3
        ge4=step*eg*(dRelu(g))*s4

        be1=step*eb*(dRelu(b))*s1
        be2=step*eb*(dRelu(b))*s2
        be3=step*eb*(dRelu(b))*s3
        be4=step*eb*(dRelu(b))*s4

        swe1=step*(er*(dRelu(r))*r1+eg*(dRelu(g))*g1+eb*(dRelu(b))*b1)*dRelu(s1)*f1
        swe2=step*(er*(dRelu(r))*r1+eg*(dRelu(g))*g1+eb*(dRelu(b))*b1)*dRelu(s1)*f2
        swe3=step*(er*(dRelu(r))*r2+eg*(dRelu(g))*g2+eb*(dRelu(b))*b2)*dRelu(s2)*f3
        swe4=step*(er*(dRelu(r))*r2+eg*(dRelu(g))*g2+eb*(dRelu(b))*b2)*dRelu(s2)*f4
        swe5=step*(er*(dRelu(r))*r3+eg*(dRelu(g))*g3+eb*(dRelu(b))*b3)*dRelu(s3)*f5
        swe6=step*(er*(dRelu(r))*r3+eg*(dRelu(g))*g3+eb*(dRelu(b))*b3)*dRelu(s3)*f6
        swe7=step*(er*(dRelu(r))*r4+eg*(dRelu(g))*g4+eb*(dRelu(b))*b4)*dRelu(s4)*f7
        swe8=step*(er*(dRelu(r))*r4+eg*(dRelu(g))*g4+eb*(dRelu(b))*b4)*dRelu(s4)*f8

        we2=step*(er*(dRelu(r))*r1+eg*(dRelu(g))*g1+eb*(dRelu(b))*b1)*dRelu(s1)*sw1*dRelu(f1)*zb[0]
        we4=step*(er*(dRelu(r))*r1+eg*(dRelu(g))*g1+eb*(dRelu(b))*b1)*dRelu(s1)*sw2*dRelu(f2)*zb[1]
        we6=step*(er*(dRelu(r))*r2+eg*(dRelu(g))*g2+eb*(dRelu(b))*b2)*dRelu(s2)*sw3*dRelu(f3)*zb[2]
        we8=step*(er*(dRelu(r))*r2+eg*(dRelu(g))*g2+eb*(dRelu(b))*b2)*dRelu(s2)*sw4*dRelu(f4)*zb[3]
        we10=step*(er*(dRelu(r))*r3+eg*(dRelu(g))*g3+eb*(dRelu(b))*b3)*dRelu(s3)*sw5*dRelu(f5)*zb[5]
        we12=step*(er*(dRelu(r))*r3+eg*(dRelu(g))*g3+eb*(dRelu(b))*b3)*dRelu(s3)*sw6*dRelu(f6)*zb[6]
        we14=step*(er*(dRelu(r))*r4+eg*(dRelu(g))*g4+eb*(dRelu(b))*b4)*dRelu(s4)*sw7*dRelu(f7)*zb[7]
        we16=step*(er*(dRelu(r))*r4+eg*(dRelu(g))*g4+eb*(dRelu(b))*b4)*dRelu(s4)*sw8*dRelu(f8)*zb[8]

        we1=step*(er*(dRelu(r1))*r1+eg*(dRelu(g1))*g1+eb*(dRelu(b1))*b1)*dRelu(s1)*sw1*dRelu(f1)*zb[4]
        we3=step*(er*(dRelu(r1))*r1+eg*(dRelu(g1))*g1+eb*(dRelu(b1))*b1)*dRelu(s1)*sw2*dRelu(f2)*zb[4]
        we5=step*(er*(dRelu(r2))*r2+eg*(dRelu(g1))*g2+eb*(dRelu(b1))*b2)*dRelu(s2)*sw3*dRelu(f3)*zb[4]
        we7=step*(er*(dRelu(r2))*r2+eg*(dRelu(g1))*g2+eb*(dRelu(b1))*b2)*dRelu(s2)*sw4*dRelu(f4)*zb[4]
        we9=step*(er*(dRelu(r3))*r3+eg*(dRelu(g3))*g3+eb*(dRelu(b3))*b3)*dRelu(s3)*sw5*dRelu(f5)*zb[4]
        we11=step*(er*(dRelu(r3))*r3+eg*(dRelu(g3))*g3+eb*(dRelu(b3))*b3)*dRelu(s3)*sw6*dRelu(f6)*zb[4]
        we13=step*(er*(dRelu(r4))*r4+eg*(dRelu(g4))*g4+eb*(dRelu(b4))*b4)*dRelu(s4)*sw7*dRelu(f7)*zb[4]
        we15=step*(er*(dRelu(r4))*r4+eg*(dRelu(g4))*g4+eb*(dRelu(b4))*b4)*dRelu(s4)*sw8*dRelu(f8)*zb[4]

        w1=w1-we1-step*w1*lam
        w2=w2-we2-step*w2*lam
        w3=w3-we3-step*w3*lam
        w4=w4-we4-step*w4*lam
        w5=w5-we5-step*w5*lam
        w6=w6-we6-step*w6*lam
        w7=w7-we7-step*w7*lam
        w8=w8-we8-step*w8*lam
        w9=w9-we9-step*w9*lam
        w10=w10-we10-step*w10*lam
        w11=w11-we11-step*w11*lam
        w12=w12-we12-step*w12*lam
        w13=w13-we13-step*w13*lam
        w14=w14-we14-step*w14*lam
        w15=w15-we15-step*w15*lam
        w16=w16-we16-step*w16*lam

        sw1=sw1-swe1-step*sw1*lam
        sw2=sw2-swe2-step*sw2*lam
        sw3=sw3-swe3-step*sw3*lam
        sw4=sw4-swe4-step*sw4*lam
        sw5=sw5-swe5-step*sw5*lam
        sw6=sw6-swe6-step*sw6*lam
        sw7=sw7-swe7-step*sw7*lam
        sw8=sw8-swe8-step*sw8*lam

        r1=r1-re1-step*r1*lam
        r2=r2-re2-step*r2*lam
        r3=r3-re3-step*r3*lam
        r4=r4-re4-step*r4*lam

        g1=g1-ge1-step*g1*lam
        g2=g2-ge2-step*g2*lam
        g3=g3-ge3-step*g3*lam
        g4=g4-ge4-step*g4*lam

        b1=b1-be1-step*b1*lam
        b2=b2-be2-step*b2*lam
        b3=b3-be3-step*b3*lam
        b4=b4-be4-step*b4*lam

        i=i+1
        if i==5000:
            break;
            pass
        pass
    print("train finish")
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
            f1=node(zb[4],zb[0],[w1,w2])
            f2=node(zb[4],zb[1],[w3,w4])
            f3=node(zb[4],zb[2],[w5,w6])
            f4=node(zb[4],zb[3],[w7,w8])
            f5=node(zb[4],zb[5],[w9,w10])
            f6=node(zb[4],zb[6],[w11,w12])
            f7=node(zb[4],zb[7],[w13,w14])
            f8=node(zb[4],zb[8],[w15,w16])


            s1=node(f1,f2,[sw1,sw2])
            s2=node(f3,f4,[sw3,sw4])
            s3=node(f5,f6,[sw5,sw6])
            s4=node(f7,f8,[sw7,sw8])


            r=nodef(s1,s2,s3,s4,[r1,r2,r3,r4])
            g=nodef(s1,s2,s3,s4,[g1,g2,g3,g4])
            b=nodef(s1,s2,s3,s4,[b1,b2,b3,b4])
            kmc[j,i]=np.matrix([r,g,b])
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
    im = Image.open('屏幕截图(363).jpeg')
    pix = im.load()
    width = im.size[0]
    height = im.size[1]
    poto=np.zeros((height,width,3))
    print(width)
    print(height)

    for x in range(width):
        for y in range(height):

            r, g, b,m = pix[x, y]
            poto[y,x,0]=r
            poto[y,x,1]=g
            poto[y,x,2]=b
    img2 = Image.fromarray(np.uint8(poto))

    img2.save("yuan.png","png")



    #zb=[[(70, 95, 52), (80, 145, 212), (206, 242, 236), (128, 161, 137), (147, 196, 217)]]
    #zb=kmeans(zb,poto,width,height)
    #print(zb)
    imz=img2.copy()
    imgbw=BW(imz,width,height)

    imgbw.save("bw.png","png")


    #img=recolor(zb, imz ,width,height)
    #img.show()
    #img.save("5mean.png","png")
    imgtest=eNN(imgbw, img2,width,height)
    imgtest.show()
    imgtest.save("final.png","png")
    print("Done.")
