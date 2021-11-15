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

def ReLu(x):
    return max(0,x)

def dRelu(x):
    if x>0:
        return 1
        pass
    if x<=0:
        print("y")
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

def nodes(x0,x1,x2,x3,x4,x5,p):
    net=p[0]+p[1]*x1+p[2]*x2+p[3]*x3+p[4]*x4+p[5]*x5
    return ReLu(net)

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

    w1=[random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2)]
    w2=[random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2)]
    w3=[random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2)]
    w4=[random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2)]
    w5=[random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2),random.random()*math.sqrt(1/2)]

    wr=[random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7)]
    wg=[random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7)]
    wb=[random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7),random.random()*math.sqrt(6/7)]

    #for zb in zuobianzon:
    while i<10000:#train
        zbi=random.random()*len(zuobianzon)
        zbi=int(zbi)
        zb=zuobianzon.pop(zbi)
        color=tag[tuple(zb)]

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
        lam=0.00003
        print(color[0]-r)
        er=-2*(color[0]-r)/3
        eg=-2*(color[1]-g)/3
        eb=-2*(color[2]-b)/3

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
            tag[le]=half(km[b,a])
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
        color=double(tag[tuple(zb)])

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
        print(color[0]-r)
        er=-(color[0]-r)/3
        eg=-(color[1]-g)/3
        eb=-(color[2]-b)/3

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




def basicAgent(bw, km_a,w,h,zb):
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
    arrzzz=np.array(zuobianzon)

    tree = KDTree(arrzzz, leaf_size=2)# in this version, in order to improve the speed, we import the kdtree to help us solve the knn problem

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
            rie=list(ri)
            dist, ind = tree.query(np.array([rie]) , k=6)
            si=[arrzzz[ind[0,0]],arrzzz[ind[0,1]],arrzzz[ind[0,2]],arrzzz[ind[0,3]],arrzzz[ind[0,4]],arrzzz[ind[0,5]]]
            ri=[tag[tuple(si[0])],tag[tuple(si[1])],tag[tuple(si[2])],tag[tuple(si[3])],tag[tuple(si[4])],tag[tuple(si[5])]]
            cl=[]
            cl0=coooo(zb[0],ri)
            cl.append(("cl0",cl0/6))
            cl1=coooo(zb[1],ri)
            cl.append(("cl1",cl1/6))
            cl2=coooo(zb[2],ri)
            cl.append(("cl2",cl2/6))
            cl3=coooo(zb[3],ri)
            cl.append(("cl3",cl3/6))
            cl4=coooo(zb[4],ri)
            cl.append(("cl4",cl4/6))
            cl.sort(key=takep,reverse=True)
            print(cl)
            na,cee=cl[0]
            na2,cer=cl[0]
            if cer==cee:
                kmc[j,i]=ri[0]
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


def improveAgent(bw, km_a,w,h):#linear regression
    bw_array = np.array(bw)
    par=np.matlib.ones((3,9))
    x=0
    while x<3:
        y=0
        while y<9:
            par[x,y]=0.1
            y=y+1
        pass
        x=x+1
    po=np.matlib.ones((9,1))
    km = np.array(km_a)
    kmc=km.copy()
    print(len(kmc))
    fla=0
    step=0.000035
    a=int(w/4)
    print(w)
    while a<w/2-1 :
        b=1
        while b<h-1:
            le=()
            m=0
            while m<3 :
                n=0
                while n<3:
                    po[m*3+n,0]=bw_array[b+m-1,a+n-1]
                    n=n+1
                m=m+1
            color= np.asmatrix(km[b,a])
            mocolor=par*po
            gra=np.matlib.ones((3,9))
            x=0
            while x<3:
                y=0
                c=mocolor[x]-color[0,x]

                while y<9:
                    gra[x,y]=step*c*po[y,0]/9
                    y=y+1
                pass
                x=x+1

            par=par-gra
            b=b+1
        a=a+1
    print(par)
    i=int(w/2)
    while i<w-1:
        j=1
        while j<h-1:
            ri=np.matlib.ones((9,1))
            m=0
            while m<3 :
                n=0
                while n<3:
                    ri[m*3+n,0]=bw_array[j+m-1,i+n-1]
                    n=n+1
                m=m+1
            mocolor=par*ri
            kmc[j,i]=mocolor.T
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
    im = Image.open('34317117421847bbc4670b6db9ce551c.jpeg')
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

    #zb=[(30, 48, 27), (15, 111, 207), (235, 224, 221), (149, 167, 165), (79, 107, 86)]
    #zb=kmeans(zb,poto,width,height)
    #print(zb)
    imz=im.copy()
    imgbw=BW(imz,width,height)
    #imgbw.show()
    imgbw.save("bw.png","png")
    #img=recolor(zb, imz ,width,height)
    #img.show()
    #img.save("5mean.png","png")
    imgtest=eNN(imgbw, img2,width,height)
    imgtest.show()
    imgtest.save("final.png","png")
    #imgtest=basicAgent(imgbw, img,width,height,zb)
    #imgtest.show()
    #imgtest.save("final.png","png")
    print("Done.")
