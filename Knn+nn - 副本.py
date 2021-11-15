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
class Node:
    step=0.09
    lfir=0.000000000001
    def __init__(self, weight):
        self.weight = weight
        self.input=[]
        self.bias=0
        self.output=0
        self.forr=[]

    def ot(self,input):
        self.input=input
        out=np.dot(self.weight,input.T)+self.bias
        self.output=ReLu(out[0,0])
        return ReLu(out[0,0])


    def Backpropagation(self,back):
        net=0
        wei=0
        for value in back:
            net=net+value
        for value in self.weight:
            wei=wei+abs(value)
        forr=net*dRelu(self.output)*self.weight
        self.bias=self.bias-Node.step*net*dRelu(self.output)
        self.weight=self.weight-Node.step*Node.lfir*self.weight-Node.step*net*dRelu(self.output)*wei
        return forr

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

def compp(a,b):
    if a[0]==b[0] and a[1]==b[1] and a[2]==b[2]:
        return True
    return False


def suiji(x,y):
    return[(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y))]

def suiji(x,y):
    return[(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y))]

def suiji2(x,y):
    return[(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y)),(random.random()*2-1)*math.sqrt(6/(x+y))]

#def eNN(bw, km_a,nor,w,h,co):
    bw_array = np.array(bw)/255
    no=np.array(nor)
    km = np.array(km_a)
    kmc=km.copy()
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
            if compp(co[0],kmc[b,a]):
                tag[le]=[1,0,0,0,0]
                pass
            elif compp(co[1],kmc[b,a]):
                tag[le]=[0,1,0,0,0]
                pass
            elif compp(co[2],kmc[b,a]):
                tag[le]=[0,0,1,0,0]
                pass
            elif compp(co[3],kmc[b,a]):
                tag[le]=[0,0,0,1,0]
                pass
            elif compp(co[4],kmc[b,a]):
                tag[le]=[0,0,0,0,1]

            b=b+1
        a=a+1
    i=0

    w1=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

    wr=[0.15,0.15,0.15,0.15,0.15,0.15]
    fn1=Node(np.matrix(suiji(10,5)))
    fn2=Node(np.matrix(suiji(10,5)))
    fn3=Node(np.matrix(suiji(10,5)))
    fn4=Node(np.matrix(suiji(10,5)))
    fn5=Node(np.matrix(suiji(10,5)))

    jn1=Node(np.matrix(suiji2(6,5)))
    jn2=Node(np.matrix(suiji2(6,5)))
    jn3=Node(np.matrix(suiji2(6,5)))
    jn4=Node(np.matrix(suiji2(6,5)))
    jn5=Node(np.matrix(suiji2(6,5)))

    tn1=Node(np.matrix(suiji2(6,5)))
    tn2=Node(np.matrix(suiji2(6,5)))
    tn3=Node(np.matrix(suiji2(6,5)))
    tn4=Node(np.matrix(suiji2(6,5)))
    tn5=Node(np.matrix(suiji2(6,5)))

    sn1=Node(np.matrix(suiji2(6,5)))
    sn2=Node(np.matrix(suiji2(6,5)))
    sn3=Node(np.matrix(suiji2(6,5)))
    sn4=Node(np.matrix(suiji2(6,5)))
    sn5=Node(np.matrix(suiji2(6,5)))


    wn1=Node(np.matrix(suiji2(6,5)))
    wn2=Node(np.matrix(suiji2(6,5)))
    wn3=Node(np.matrix(suiji2(6,5)))
    wn4=Node(np.matrix(suiji2(6,5)))
    wn5=Node(np.matrix(suiji2(6,5)))


    cn1=Node(np.matrix(suiji2(6,1)))
    cn2=Node(np.matrix(suiji2(6,1)))
    cn3=Node(np.matrix(suiji2(6,1)))
    cn4=Node(np.matrix(suiji2(6,1)))
    cn5=Node(np.matrix(suiji2(6,1)))
    #for zb in zuobianzon:
    while i<10000:#train
        zbi=random.random()*len(zuobianzon)
        zbi=int(zbi)
        zb=zuobianzon.pop(zbi)
        colort=tag[tuple(zb)]
        #,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]
        f1=fn1.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))
        f2=fn2.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))
        f3=fn3.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))
        f4=fn4.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))
        f5=fn5.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))


        j1=jn1.ot(np.matrix([1,f1,f2,f3,f4,f5]))
        j2=jn2.ot(np.matrix([1,f1,f2,f3,f4,f5]))
        j3=jn3.ot(np.matrix([1,f1,f2,f3,f4,f5]))
        j4=jn4.ot(np.matrix([1,f1,f2,f3,f4,f5]))
        j5=jn5.ot(np.matrix([1,f1,f2,f3,f4,f5]))

        t1=tn1.ot(np.matrix([1,j1,j2,j3,j4,j5]))
        t2=tn2.ot(np.matrix([1,j1,j2,j3,j4,j5]))
        t3=tn3.ot(np.matrix([1,j1,j2,j3,j4,j5]))
        t4=tn4.ot(np.matrix([1,j1,j2,j3,j4,j5]))
        t5=tn5.ot(np.matrix([1,j1,j2,j3,j4,j5]))

        s1=sn1.ot(np.matrix([1,t1,t2,t3,t4,t5]))
        s2=sn2.ot(np.matrix([1,t1,t2,t3,t4,t5]))
        s3=sn3.ot(np.matrix([1,t1,t2,t3,t4,t5]))
        s4=sn4.ot(np.matrix([1,t1,t2,t3,t4,t5]))
        s5=sn5.ot(np.matrix([1,t1,t2,t3,t4,t5]))

        w1=wn1.ot(np.matrix([1,s1,s2,s3,s4,s5]))
        w2=wn2.ot(np.matrix([1,s1,s2,s3,s4,s5]))
        w3=wn3.ot(np.matrix([1,s1,s2,s3,s4,s5]))
        w4=wn4.ot(np.matrix([1,s1,s2,s3,s4,s5]))
        w5=wn5.ot(np.matrix([1,s1,s2,s3,s4,s5]))

        c1=cn1.ot(np.matrix([1,w1,w2,w3,w4,w5]))
        c2=cn2.ot(np.matrix([1,w1,w2,w3,w4,w5]))
        c3=cn3.ot(np.matrix([1,w1,w2,w3,w4,w5]))
        c4=cn4.ot(np.matrix([1,w1,w2,w3,w4,w5]))
        c5=cn5.ot(np.matrix([1,w1,w2,w3,w4,w5]))


        e1=-(colort[0]-c1)/5
        print((colort[0]-c1)*(colort[0]-c1))
        e2=-(colort[1]-c2)/5
        e3=-(colort[2]-c3)/5
        e4=-(colort[3]-c4)/5
        e5=-(colort[4]-c5)/5

        ee1=cn1.Backpropagation([e1])
        ee2=cn2.Backpropagation([e2])
        ee3=cn3.Backpropagation([e3])
        ee4=cn4.Backpropagation([e4])
        ee5=cn5.Backpropagation([e5])

        we1=wn1.Backpropagation([ee1[0,0],ee2[0,0],ee3[0,0],ee4[0,0],ee5[0,0]])
        we2=wn2.Backpropagation([ee1[0,1],ee2[0,1],ee3[0,1],ee4[0,1],ee5[0,1]])
        we3=wn3.Backpropagation([ee1[0,2],ee2[0,2],ee3[0,2],ee4[0,2],ee5[0,2]])
        we4=wn4.Backpropagation([ee1[0,3],ee2[0,3],ee3[0,3],ee4[0,3],ee5[0,3]])
        we5=wn5.Backpropagation([ee1[0,4],ee2[0,4],ee3[0,4],ee4[0,4],ee5[0,4]])

        se1=sn1.Backpropagation([we1[0,0],we2[0,0],we3[0,0],we4[0,0],we5[0,0]])
        se2=sn2.Backpropagation([we1[0,1],we2[0,1],we3[0,1],we4[0,1],we5[0,1]])
        se3=sn3.Backpropagation([we1[0,2],we2[0,2],we3[0,2],we4[0,2],we5[0,2]])
        se4=sn4.Backpropagation([we1[0,3],we2[0,3],we3[0,3],we4[0,3],we5[0,3]])
        se5=sn5.Backpropagation([we1[0,4],we2[0,4],we3[0,4],we4[0,4],we5[0,4]])

        te1=tn1.Backpropagation([se1[0,0],se2[0,0],se3[0,0],se4[0,0],se5[0,0]])
        te2=tn2.Backpropagation([se1[0,1],se2[0,1],se3[0,1],se4[0,1],se5[0,1]])
        te3=tn3.Backpropagation([se1[0,2],se2[0,2],se3[0,2],se4[0,2],se5[0,2]])
        te4=tn4.Backpropagation([se1[0,3],se2[0,3],se3[0,3],se4[0,3],se5[0,3]])
        te5=tn5.Backpropagation([se1[0,4],se2[0,4],se3[0,4],se4[0,4],se5[0,4]])

        je1=jn1.Backpropagation([te1[0,0],te2[0,0],te3[0,0],te4[0,0],te5[0,0]])
        je2=jn2.Backpropagation([te1[0,1],te2[0,1],te3[0,1],te4[0,1],te5[0,1]])
        je3=jn3.Backpropagation([te1[0,2],te2[0,2],te3[0,2],te4[0,2],te5[0,2]])
        je4=jn4.Backpropagation([te1[0,3],te2[0,3],te3[0,3],te4[0,3],te5[0,3]])
        je5=jn5.Backpropagation([te1[0,4],te2[0,4],te3[0,4],te4[0,4],te5[0,4]])


        fn1.Backpropagation([je1[0,0],je2[0,0],je3[0,0],je4[0,0],je5[0,0]])
        fn2.Backpropagation([je1[0,1],je2[0,1],je3[0,1],je4[0,1],je5[0,1]])
        fn3.Backpropagation([je1[0,2],je2[0,2],je3[0,2],je4[0,2],je5[0,2]])
        fn4.Backpropagation([je1[0,3],je2[0,3],je3[0,3],je4[0,3],je5[0,3]])
        fn5.Backpropagation([je1[0,4],je2[0,4],je3[0,4],je4[0,4],je5[0,4]])

        i=i+1
        #if i==10000:
        #    break;
        #    pass
        pass

    print("train finish")
    print(fn1.weight)

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
            f1=fn1.ot(np.matrix([1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8]]))
            f2=fn2.ot(np.matrix([1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8]]))
            f3=fn3.ot(np.matrix([1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8]]))
            f4=fn4.ot(np.matrix([1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8]]))
            f5=fn5.ot(np.matrix([1,zb[0],zb[1],zb[2],zb[3],zb[4],zb[5],zb[6],zb[7],zb[8]]))

            j1=jn1.ot(np.matrix([1,f1,f2,f3,f4,f5]))
            j2=jn2.ot(np.matrix([1,f1,f2,f3,f4,f5]))
            j3=jn3.ot(np.matrix([1,f1,f2,f3,f4,f5]))
            j4=jn4.ot(np.matrix([1,f1,f2,f3,f4,f5]))
            j5=jn5.ot(np.matrix([1,f1,f2,f3,f4,f5]))

            t1=tn1.ot(np.matrix([1,j1,j2,j3,j4,j5]))
            t2=tn2.ot(np.matrix([1,j1,j2,j3,j4,j5]))
            t3=tn3.ot(np.matrix([1,j1,j2,j3,j4,j5]))
            t4=tn4.ot(np.matrix([1,j1,j2,j3,j4,j5]))
            t5=tn5.ot(np.matrix([1,j1,j2,j3,j4,j5]))

            s1=sn1.ot(np.matrix([1,t1,t2,t3,t4,t5]))
            s2=sn2.ot(np.matrix([1,t1,t2,t3,t4,t5]))
            s3=sn3.ot(np.matrix([1,t1,t2,t3,t4,t5]))
            s4=sn4.ot(np.matrix([1,t1,t2,t3,t4,t5]))
            s5=sn5.ot(np.matrix([1,t1,t2,t3,t4,t5]))

            w1=wn1.ot(np.matrix([1,s1,s2,s3,s4,s5]))
            w2=wn2.ot(np.matrix([1,s1,s2,s3,s4,s5]))
            w3=wn3.ot(np.matrix([1,s1,s2,s3,s4,s5]))
            w4=wn4.ot(np.matrix([1,s1,s2,s3,s4,s5]))
            w5=wn5.ot(np.matrix([1,s1,s2,s3,s4,s5]))

            c1=cn1.ot(np.matrix([1,w1,w2,w3,w4,w5]))
            c2=cn2.ot(np.matrix([1,w1,w2,w3,w4,w5]))
            c3=cn3.ot(np.matrix([1,w1,w2,w3,w4,w5]))
            c4=cn4.ot(np.matrix([1,w1,w2,w3,w4,w5]))
            c5=cn5.ot(np.matrix([1,w1,w2,w3,w4,w5]))
            zas=[]
            zas.append(c1)
            zas.append(c2)
            zas.append(c3)
            zas.append(c4)
            zas.append(c5)
            print(zas)
            zas.sort(reverse=True)

            if zas[0]==c1:
                no[j,i]=np.matrix(co[0])
                pass
            elif zas[0]==c2:
                no[j,i]=np.matrix(co[1])
                pass
            elif zas[0]==c3:
                pass
                no[j,i]=np.matrix(co[2])
            elif zas[0]==c4:
                no[j,i]=np.matrix(co[3])
                pass
            elif zas[0]==c5:
                no[j,i]=np.matrix(co[4])
                pass
            j=j+1
        i=i+1

        #end = skimage.color.lab2rgb(km)*255
        #end = end.astype(np.uint8)
        ima = Image.fromarray(np.uint8(no))
    return ima


def eNN(bw, km_a,w,h,co):
    bw_array = np.array(bw)/255
    km = np.array(km_a)
    kmc=km.copy()
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
            tag[le]=kmc[b,a]/255
            b=b+1
        a=a+1
    i=0

    w1=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

    wr=[0.15,0.15,0.15,0.15,0.15,0.15]
    fn1=Node(np.matrix(suiji(19,5)))
    fn2=Node(np.matrix(suiji(19,5)))
    fn3=Node(np.matrix(suiji(19,5)))
    fn4=Node(np.matrix(suiji(19,5)))
    fn5=Node(np.matrix(suiji(19,5)))

    jn1=Node(np.matrix(wr))
    jn2=Node(np.matrix(wr))
    jn3=Node(np.matrix(wr))
    jn4=Node(np.matrix(wr))
    jn5=Node(np.matrix(wr))

    cn1=Node(np.matrix(wr))
    cn2=Node(np.matrix(wr))
    cn3=Node(np.matrix(wr))
    #for zb in zuobianzon:
    while i<2500:#train
        zbi=random.random()*len(zuobianzon)
        zbi=int(zbi)
        zb=zuobianzon.pop(zbi)
        colort=tag[tuple(zb)]
        #,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]
        f1=fn1.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))
        f2=fn2.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))
        f3=fn3.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))
        f4=fn4.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))
        f5=fn5.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))


        j1=jn1.ot(np.matrix([1,f1,f2,f3,f4,f5]))
        j2=jn2.ot(np.matrix([1,f1,f2,f3,f4,f5]))
        j3=jn3.ot(np.matrix([1,f1,f2,f3,f4,f5]))
        j4=jn4.ot(np.matrix([1,f1,f2,f3,f4,f5]))
        j5=jn5.ot(np.matrix([1,f1,f2,f3,f4,f5]))

        c1=cn1.ot(np.matrix([1,j1,j2,j3,j4,j5]))
        c2=cn2.ot(np.matrix([1,j1,j2,j3,j4,j5]))
        c3=cn3.ot(np.matrix([1,j1,j2,j3,j4,j5]))


        print(colort)
        e1=-(colort[0]-c1)/3
        print((colort[0]-c1)*(colort[0]-c1))
        e2=-(colort[1]-c2)/3
        e3=-(colort[2]-c3)/3


        ee1=cn1.Backpropagation([e1])
        ee2=cn2.Backpropagation([e2])
        ee3=cn3.Backpropagation([e3])


        je1=jn1.Backpropagation([ee1[0,0],ee2[0,0],ee3[0,0]])
        je2=jn2.Backpropagation([ee1[0,1],ee2[0,1],ee3[0,1]])
        je3=jn3.Backpropagation([ee1[0,2],ee2[0,2],ee3[0,2]])
        je4=jn4.Backpropagation([ee1[0,3],ee2[0,3],ee3[0,3]])
        je5=jn5.Backpropagation([ee1[0,4],ee2[0,4],ee3[0,4]])


        fn1.Backpropagation([je1[0,0],je2[0,0],je3[0,0],je4[0,0],je5[0,0]])
        fn2.Backpropagation([je1[0,1],je2[0,1],je3[0,1],je4[0,1],je5[0,1]])
        fn3.Backpropagation([je1[0,2],je2[0,2],je3[0,2],je4[0,2],je5[0,2]])
        fn4.Backpropagation([je1[0,3],je2[0,3],je3[0,3],je4[0,3],je5[0,3]])
        fn5.Backpropagation([je1[0,4],je2[0,4],je3[0,4],je4[0,4],je5[0,4]])


        i=i+1
        #if i==10000:
        #    break;
        #    pass
        pass

    print("train finish")
    print(fn1.weight)

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
            f1=fn1.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))
            f2=fn2.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))
            f3=fn3.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))
            f4=fn4.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))
            f5=fn5.ot(np.matrix([1,zb[0]*zb[0],zb[1]*zb[1],zb[2]*zb[2],zb[3]*zb[3],zb[4]*zb[4],zb[5]*zb[5],zb[6]*zb[6],zb[7]*zb[7],zb[8]*zb[8]]))


            j1=jn1.ot(np.matrix([1,f1,f2,f3,f4,f5]))
            j2=jn2.ot(np.matrix([1,f1,f2,f3,f4,f5]))
            j3=jn3.ot(np.matrix([1,f1,f2,f3,f4,f5]))
            j4=jn4.ot(np.matrix([1,f1,f2,f3,f4,f5]))
            j5=jn5.ot(np.matrix([1,f1,f2,f3,f4,f5]))

            r=cn1.ot(np.matrix([1,j1,j2,j3,j4,j5]))*255
            g=cn2.ot(np.matrix([1,j1,j2,j3,j4,j5]))*255
            b=cn3.ot(np.matrix([1,j1,j2,j3,j4,j5]))*255
            #print(r)
            #print((r,g,b))
            #hui=km[j,i]
            #print([hui[0],g,b])
            km[j,i]=np.matrix([r,g,b])
            #coo=co[0]
            #cd1=math.sqrt((coo[0]-r)*(coo[0]-r)+(coo[1]-g)*(coo[1]-g)+(coo[2]-b)*(coo[2]-b))
            #coo=co[1]
            #cd2=math.sqrt((coo[0]-r)*(coo[0]-r)+(coo[1]-g)*(coo[1]-g)+(coo[2]-b)*(coo[2]-b))
            #coo=co[2]
            #cd3=math.sqrt((coo[0]-r)*(coo[0]-r)+(coo[1]-g)*(coo[1]-g)+(coo[2]-b)*(coo[2]-b))
            #coo=co[3]
            #cd4=math.sqrt((coo[0]-r)*(coo[0]-r)+(coo[1]-g)*(coo[1]-g)+(coo[2]-b)*(coo[2]-b))
            #coo=co[4]
            #cd5=math.sqrt((coo[0]-r)*(coo[0]-r)+(coo[1]-g)*(coo[1]-g)+(coo[2]-b)*(coo[2]-b))
            #zas=[]
            #zas.append(cd1)
            #zas.append(cd2)
            #zas.append(cd3)
            #zas.append(cd4)
            #zas.append(cd5)
            #zas.sort()

            #if zas[0]==cd1:
        #        km[j,i]=np.matrix(co[0])
        #        pass
        #    elif zas[0]==cd2:
        #        km[j,i]=np.matrix(co[1])
        #        pass
        #    elif zas[0]==cd3:
        #        km[j,i]=np.matrix(co[2])
        #        pass
        #    elif zas[0]==cd4:
        #        km[j,i]=np.matrix(co[3])
        #        pass
#    elif zas[0]==cd5:
#                km[j,i]=np.matrix(co[4])
#                pass
            j=j+1
        i=i+1

        #end = skimage.color.lab2rgb(km)*255
        #end = end.astype(np.uint8)
        ima = Image.fromarray(np.uint8(km))
    return ima





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
    im = Image.open('70190c0b3c0c116c96b57f1bdfbdf338.jpeg')
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
    #(115, 128, 54), (15, 124, 102), (249, 238, 204), (252, 152, 2), (245, 191, 73)
    zb=[(70, 95, 52), (80, 145, 212), (206, 242, 236), (128, 161, 137), (147, 196, 217)]
    #zb=kmeans(zb,poto,width,height)
    print(zb)
    imz=img2.copy()
    imgbw=BW(imz,width,height)
    imgbw.save("bw.png","png")
    #img=recolor(zb, imz ,width,height)
    #img.save("5mean.png","png")
    imgtest=eNN(imgbw,img2,width,height,zb)
    imgtest.show()
    imgtest.save("final.png","png")
    print("Done.")
