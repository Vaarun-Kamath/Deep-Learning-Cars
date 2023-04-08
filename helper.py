import numpy as np
import cv2 as cv
from math import pi as PI,tan

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
    } # (inner(y,x) ,outer(y,x))
    # outer, inner, corners = get_corners(g_img, dim)                                                                                       #* DEBUG
    # chk_points = {                                                                                                                        #* DEBUG
    #     (i,o)                                                                                                                             #* DEBUG
    #     for i in inner                                                                                                                    #* DEBUG
    #     for o in outer                                                                                                                    #* DEBUG
    #     if (i[0]-o[0])**2 + (i[1]-o[1])**2 <= 140**2                                                                                      #* DEBUG
    # }                                                                                                                                     #* DEBUG
    # return outer, inner, corners, chk_points                                                                                              #* DEBUG


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
    # return outer, inner, corners                                                                                                          #* DEBUG

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


def get_current_chkpt(car, chkpts)->set|None:
    X1 = car.x - car.width/2
    Y1 = car.y - car.height/2
    for (SY,SX),(EY,EX) in chkpts:
        X_len = EX - SX or 1
        Y_len = EY - SY or 1
        Y_diff = Y1 - SY
        X_diff = X1 - SX
        denominator = car.width * Y_len
        assert denominator!=0, f'{car.width=},{Y_len=}'
        s = ( car.width * Y_diff) / denominator
        t = ( X_len * Y_diff - Y_len * X_diff ) / denominator
        if 0 <= s <= 1 and 0 <= t <= 1:
            return (SY,SX),(EY,EX)
        denominator = -X_len * car.height
        assert denominator!=0, f'{car.height=},{X_len=}'
        s = (- car.height * X_diff ) / denominator
        t = ( X_len * Y_diff - Y_len * X_diff ) / denominator
        if 0 <= s <= 1 and 0 <= t <= 1:
            return (SY,SX),(EY,EX)
    return None


def pts_on_line(cx,cy,theta, /):
    assert PI>=theta>=-PI, f'theta {theta} should be in radians, [-PI,PI]'
    x_step = (1,-1)[abs(theta)>PI/2]
    y_step = (1,-1)[theta<0]
    m = tan(theta)
    dy, dx = 0, 0
    while True:
        dy += y_step
        dx += x_step
        if m==0: y,x = 0,60_000
        else: y, x = int(m * dx), int(dy / m)
        if abs(y)>abs(x):
            dx -= x_step
            yield dy,x
        else:
            dy -= y_step
            yield y, dx


def get_dist(cx,cy, theta, g_img):
    for y,x in pts_on_line( cx, cy, theta ):
        if 0>int(y+cy) or 720<=int(y+cy) or  0>int(x+cx) or  1280<=int(x+cx) or g_img[int(y+cy), int(x+cx)]:
            return (y**2 + x**2)**0.5
            return y,x                                                                                                                      #* DEBUG