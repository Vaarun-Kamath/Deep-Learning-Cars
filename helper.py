import numpy as np
import cv2 as cv

_kernel={}
def kernel(n:int, /)-> set[tuple[int,int]]:
    assert n&1, '\'n\' should be odd'
    if _kernel.get(n) is None:
        _kernel[n] = {
            (i,j)
            for i in range(-(n//2),n//2+1)
            for j in range(-(n//2),n//2+1)
            if i or j
        }
    return _kernel[n]


def get_new_img(dim:tuple[int,int],path:str,/):
    assert dim[0]>dim[1], '(width, height)'
    img = cv.resize(cv.imread(path), dim)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fltr = cv.bilateralFilter(gray, 10, 15, 15)
    track_points = cv.adaptiveThreshold(fltr, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 5)
    return track_points


def get_strt_pt(g_scale_img)->tuple[int,int]:
    for i,row in enumerate(g_scale_img):
        for j,pixel in enumerate(row):
            if pixel: return i,j


def nbrs(x:tuple[int,int],dim:tuple[int,int],img,/,*,k=3)->tuple[tuple[int,int]]:
    """returns neighbours of 'x' in 'img' of dimension 'dim'(width, height)
        if the neighbours are within the dimensions of img and in on state'"""
    i, j = x
    return {(i+y,j+x)for y,x in kernel(k) if dim[1]>(i+y)>=0 and dim[0]>(j+x)>=0 and img[i+y,j+x]}


def get_checkpoints(g_img, dim, /):
    assert dim[0]>dim[1], '(width, height)'
    outer, inner = get_corners(g_img, dim)
    return{
        (i,o)
        for i in inner
        for o in outer
        if (i[0]-o[0])**2 + (i[1]-o[1])**2 <= 140**2
    } 
    #* DEBUG
    #* outer, inner, corners = get_corners(g_img, dim)
    #* chk_points = {
    #*     (i,o)
    #*     for i in inner
    #*     for o in outer
    #*     if (i[0]-o[0])**2 + (i[1]-o[1])**2 <= 140**2
    #* }
    #* return outer, inner, corners, chk_points


def get_corners(g_img, dim, /):
    corners = cv.goodFeaturesToTrack(g_img,10000,0.2,50)
    outer, inner = set(), set()
    corners = {tuple(x[0]) for x in corners}
    a, b = get_sets(g_img, dim) # outer line, inner line
    for x,y in corners:
        neighbours = nbrs((int(y),int(x)), dim, g_img,k=5)
        if a&neighbours: outer.add((int(y),int(x)))
        elif b&neighbours: inner.add((int(y),int(x)))
        else: raise Exception('point couldn\'t find boundary --> Consider increase ing the \'k\' value')
    return outer, inner
    #* DEBUG
    #* return outer, inner, corners

def get_sets(g_img,dim,/):
    "returns the outer and inner set of points in the track"
    assert dim[0]>dim[1], '(width, height)'
    outer, inner = set(), set()
    img = np.copy(g_img)
    Q = [get_strt_pt(img)]
    while len(Q):
        for i,j in nbrs(Q.pop(0),dim,img):
            pt = (i,j)
            img[i,j]=0
            outer.add(pt)
            Q.append(pt)
    Q = [get_strt_pt(img)]
    while len(Q):
        for i,j in nbrs(Q.pop(0),dim,img):
            pt = (i,j)
            img[i,j]=0
            inner.add(pt)
            Q.append(pt)
    return outer, inner