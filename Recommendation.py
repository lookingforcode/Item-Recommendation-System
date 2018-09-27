'''
Author : Pavlos Tiritirs , Sotiris Tapaskos
Description: SVD/CFCB Unify Model for Recommandations
Date created: 9/27/2018
Python version: 3
'''


import heapq
import operator
import numpy as np
import time
import sparsesvd
from scipy.sparse import csc_matrix
import scipy.spatial.distance as sd

def maxmins(R,u,v):
    n, m = R.shape
    pieces = round(min(n,m)/150)
    max1=0
    max2=0
    min1=1000000
    min2=1000000
    for p in range(pieces):
        p1 = round((p / pieces) * n)
        p2 = round(((p + 1) / pieces) * n)
        C1=1-sd.cdist(u[p1:p2], u, 'cityblock')
        C1 /= C1.sum(axis=1)[:, None]
        C1= C1 * R
        C1+=R[p1:p2]*(-1000000)
        maxx1= C1.max()
        C1 += R[p1:p2]*(10000000)
        minn1=C1.min()
        if maxx1>max1:
            max1=maxx1
        if minn1<min1:
            min1=minn1
        p1 = round((p / pieces) * m)
        p2 = round(((p + 1) / pieces) * m)
        C2 = 1-sd.cdist(v[p1:p2], v, 'cityblock')
        C2 /= C2.sum(axis=1)[:, None]
        C2=R*C2.T
        C2+=R[:,p1:p2]*(-1000000)
        maxx2=C2.max()
        C2 += R[:,p1:p2] * (10000000)
        minn2=C2.min()
        if maxx2>max2:
            max2=maxx2
        if minn2<min2:
            min2=minn2

    max1=max(max1,R.max())
    max2=max(max2,R.max())
    min1=min(min1,R.data.min())
    min2=min(min2,R.data.min())

    return max1,max2,min1,min2

def UnifyModel(R,u,v,top,a):
    n,m=R.shape
    max1,max2,min1,min2 = maxmins(R, u , v)
    map1={}
    map2={}
    top20=np.zeros((n,20))
    ind20=np.zeros((n,20))
    pieces= round(m/70)
    for p in range(pieces):
        p1=round((p/pieces)*n)
        p2=round(((p+1)/pieces)*n)
        C1 = 1 - sd.cdist(u[p1:p2], u, 'cityblock')
        C1 /= C1.sum(axis=1)[:, None]
        C1 = C1 * R
        C1=((C1 - min1) / (max1 - min1)) * a
        q1=round((p/pieces)*m)
        q2=round(((p+1)/pieces)*m)
        C2=1-sd.cdist(v[q1:q2],v,'cityblock')
        C2 /= C2.sum(axis=1)[:, None]
        C2=R*C2.T
        C2=((C2-min2)/(max2-min2))*(1-a)
        if p==0:
            map1[0]=C1[:,q2:m]
            map2[0]=C2[p2:n]
            intersect=C2[0:p2]+C1[:,0:q2]+ (R[0:p2, 0:q2] * (-1000000)).toarray()
            for i in range(0, p2):
                list_top20 = list(zip(*heapq.nlargest(20, enumerate(intersect[i]), key=operator.itemgetter(1))))
                ind20[i] = list_top20[0]
                top20[i] = list_top20[1]
        else:
            newcol=np.zeros((p1,q2-q1))
            newrow=np.zeros((p2-p1,q1))
            k=0
            k2=0
            for i in range(p):
                newcol[k:k+len(map1[i])]=map1[i][:,0:(q2-q1)]+C2[k:k+len(map1[i])]+(R[k:k+len(map1[i]),q1:q2]*(-1000000)).toarray()
                newrow[:,k2:k2+len(map2[i][0])]=map2[i][0:(p2-p1)]+C1[:,k2:k2+len(map2[i][0])]+(R[p1:p2,k2:k2+len(map2[i][0])] * (-1000000)).toarray()
                k+=len(map1[i])
                k2+=len(map2[i][0])
            newcol_ind=[x for x in range(q1,q2)]
            intersect=C2[p1:p2]+C1[:,q1:q2]+ (R[p1:p2, q1:q2] * (-1000000)).toarray()

            for i in range(p):
                map1[i]=map1[i][:,(q2-q1):]
                map2[i]=map2[i][(p2-p1):]
            map1[p]=C1[:,q2:m]
            map2[p]=C2[p2:n]
            for i in range(0,p1):
                data=top20[i].tolist()+newcol[i].tolist()
                ind=ind20[i].tolist()+newcol_ind
                list_top20 = list(zip(*heapq.nlargest(20, enumerate(data), key=operator.itemgetter(1))))
                ind20[i] = [ind[x] for x in list_top20[0]]
                top20[i] = list_top20[1]
            ind2=[x for x in range(0,q2)]
            for i in range(p1,p2):
                data=newrow[i-p1].tolist()+intersect[i-p1].tolist()
                list_top20 = list(zip(*heapq.nlargest(20, enumerate(data), key=operator.itemgetter(1))))
                ind20[i] = [ind2[x] for x in list_top20[0]]
                top20[i] = list_top20[1]
    accur1 = 0
    accur5 = 0
    accur10 = 0
    accur20 = 0
    for i in range(n):
        ac1 = 0
        ac5 = 0
        ac10 = 0
        if ind20[i,0] in top[i]:
            accur1 += 1
            ac1 = 1
        ac5 = (len([x for x in ind20[i,1:5] if x in top[i]]) + ac1)
        accur5 += ac5
        ac10 = (len([x for x in ind20[i,5:10] if x in top[i]]) + ac5)
        accur10 += ac10
        accur20 += (len([x for x in ind20[i,10:20] if x in top[i]]) + ac10)

    return accur1,accur5,accur10,accur20



file='ml-1m/ratings.dat'
sep='::'
a=0.8
factors=140

start = time.time()

#Reading
f = open(file, 'r')
row = []
col = []
data = []
map = {}#mapping each user with movie , rating, timestamp
s1 = set()#set of movies
for line in f:
    elements = line.split(sep)
    e0 = int(elements[0])
    e1 = int(elements[1])
    e2 = float(elements[2])
    e3 = int(elements[3])
    s1.add(e1)
    if e0 in map:
        map[e0].append([e1, e2, e3])
    else:
        map[e0] = [[e1, e2, e3]]

f.close()

#k number of movies
k = 0
map2 = {}#mapping each movie in a column
for x in s1:
    map2[x] = k
    k += 1

rp = 0
top = []

for x in map:
    #sorted by timestamp
    map[x] = sorted(map[x], key=lambda y: y[2])
    #we take only 80% of the ratings and use the rest for testing accuracy
    size = int(len(map[x]) * 0.8)
    for input_list in map[x][0:size]:
        row.append(rp)
        col.append(map2[input_list[0]])
        data.append(input_list[1])
    top.append([map2[p[0]] for p in map[x][size:len(map[x])]])
    rp = rp + 1

N = rp
M = k
#creating sparse matrix
R = csc_matrix((data, (row, col)), shape=(N, M), dtype=float)
row = 0
col = 0
data = 0
map = 0
map2 = 0
s1 = 0
R.eliminate_zeros()

#SVD
u, s, v = sparsesvd.sparsesvd(R, factors)

# DIMENSIONALITY REDUCTION
s2 = s ** 2
threshold = 0.8 * sum(s2)
s_sum = 0
cut_point = 0
for i in range(len(s)):
    s_sum += s2[i]
    if s_sum >= threshold:
        cut_point = i

s2 = 0
s = 0
u = u.T
u = u[:, 0:cut_point]
v = v[0:cut_point, :]

#finding accuracy
accuracy1, accuracy5, accuracy10, accuracy20 = UnifyModel(R, u, v.T, top, a)
print("a = ", a, " ,factors = ", factors, " :")
print("average top1 accuracy= ", accuracy1 / N)
print("average top5 accuracy= ", accuracy5 / (5 * N))
print("average top10 accuracy= ", accuracy10 / (10 * N))
print("average top20 accuracy= ", accuracy20 / (20 * N))
duration = time.time() - start
print('SVD-COLLABORATIVE-CONTENT_BASE  FILTERING  time = ', duration)