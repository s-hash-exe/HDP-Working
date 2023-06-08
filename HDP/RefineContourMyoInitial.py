"""
Created by  : Prof Arbind Kumar Gupta
Description : It uses hierarchical dynamic programming for optimising the contour already detected
              using DL.
Dated       : 28 Aug 2022
Status      : Working for both endo/pericardium
Bugs        : No known bugs
Changes     : Fixed problem in displaying contour for non squre images
Version     : 1.02
"""
import os
import sys
from pathlib import Path

import cv2
import math
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import pydicom as pd

from DynamicProgramming_03Nov import prepareGraph as DynProg
from dice import calDICE, readGroundTruth

sys.path.append(r'.')  # Enter the path where SupportFunctions.py is available
import SupportFunctions as SF

# plt.set_cmap('gray')  # to set default colour for matplotlib to gray
IMSIZE = (192, 192)
SAMPLE_PTS = 120
# SAMPLE_PTS = 20
ONECARDIO = False  # when called from within OneCario. Used to save results, if run as a stand alone program
SF.DISPLAY = True
plt.ioff()

# to share the image and loc of the image for plotting of contour on the image
img, loc = None, None
fp, data = None, {}


# -----------------------------------------------------------------------------------------------------------------------
def share() :
    return img, loc


# -----------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------
def ptOfIntersection(L1, L2) :  # L1[0] is a point on line L1 and {1[1] is slope of its normal in degree
    # L2 is defined by two points - L2[0] and L2[1]
    a1, b1 = math.cos(L1[1] * math.pi / 180), math.sin(L1[1] * math.pi / 180)
    c1 = -(a1 * L1[0][0] + b1 * L1[0][1])
    if c1 > 0 : a1, b1, c1 = -a1, -b1, -c1  # c1 being positive means the angle of normal to the line is flipped by 180 deg
    # For correct value of theta, the distance to the center should be positive, ie c1 should be negative
    l = math.sqrt((L2[0][0] - L2[1][0]) ** 2 + (L2[0][1] - L2[1][1]) ** 2)
    a2, b2 = (L2[1][1] - L2[0][1]) / l, (L2[0][0] - L2[1][0]) / l
    c2 = -(L2[0][0] * a2 + L2[0][1] * b2)
    if c2 > 0 : a2, b2, c2 = -a2, -b2, -c2

    den = a1 * b2 - a2 * b1
    if abs(den) < 0.01 :
        print('denominator is zero: %f. Returning average of two points in L2' % (den))
        return ((L2[0][0] + L2[1][0]) // 2, (L2[0][1] + L2[1][1]) // 2)
        # It is a case when the two lines L1 and L2 are parallel to each other. So an approx value is returned
    x, y = (b1 * c2 - b2 * c1) / den, (a2 * c1 - a1 * c2) / den
    return (x, y)


# -----------------------------------------------------------------------------------------------------------------------
def distToCentre(th, loc, ctr, idx) :
    dist = lambda p1, p2 : math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    th += 90  # slope of normal to first line L1 (in ptOfIntersection)
    if th >= 360 : th = th - 360
    x, y = ptOfIntersection([loc, th], [ctr[idx - 1], ctr[idx]])
    d = dist(loc, (x, y))
    return d


# -----------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------
# Added on : 27-03-23 (Partha Sir)

def oldCtrPtAngles(ctr, loc):
    chNegativeTheta = lambda th: th + 360 if th < 0 else th
    dirCnt, idx = 0, 0

    # get the angles of all points on the contour wrt centre at loc
    angles = [chNegativeTheta(math.atan2(pt[1] - loc[1], pt[0] - loc[0]) * 180 / math.pi) for pt in ctr]
    for i in range(len(angles)):  # find the direction of contour
        if angles[i] > angles[i - 1]:
            dirCnt += 1
        else:
            dirCnt -= 1
    if dirCnt < 0:
        angles.reverse(), ctr.reverse()
    angles = np.array(angles, float)
    ctr = [ctr[(angles.argmin() + i) % len(ctr)] for i in range(len(ctr))]
    angles = [angles[(angles.argmin() + i) % len(angles)] for i in range(len(angles))]
    return angles, ctr


def ctrPtAngles(ctr, loc) :
    chNegativeTheta = lambda th: th + 360 if th < 0 else th
    sign = lambda a: (a > 0) - (a < 0)
    dirCnt, idx = 0, 0

    # get the angles of all points on the contour wrt centre at loc
    angles = [chNegativeTheta(math.atan2(pt[1] - loc[1], pt[0] - loc[0]) * 180 / math.pi) for pt in ctr]
    flag = False
    for i in range(len(angles)):  # find the direction of contour
        if angles[i] > angles[i - 1]:
            dirCnt += 1
        else:
            dirCnt -= 1
    # if dirCnt < 0 or (dirCnt==0 and flag):
    if dirCnt < 0:
        angles.reverse(), ctr.reverse()
    angles = np.array(angles, float)

    for i in range(1, len(angles)):  # find the direction of contour
        if angles[i] < angles[i - 1]:
            angles[i] += 360
    return angles, ctr


def get_StartEndIdx(centre, fstPt, lastPt, numSamples) :
    chNegativeTheta = lambda th : th + 360 if th < 0 else th
    firstIdx =  chNegativeTheta(math.atan2(fstPt[1] - centre[1], fstPt[0] - centre[0]) * 180 / math.pi)
    lastIdx = chNegativeTheta(math.atan2(lastPt[1] - centre[1], lastPt[0] - centre[0]) * 180 / math.pi)

    inc = 360 / numSamples
    firstIdx, lastIdx = round(firstIdx / inc), round(lastIdx / inc) % numSamples
    return firstIdx, lastIdx

def getCentre(ctr) :
    # from cv2 import moments
    locs = [0, 0]
    M = cv2.moments(ctr)  # contour moments : set of scalar values that describe the distribution of pixels in an image or a contour
    locs = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return locs

def transformCtr(ctr,loc,sample=None):
    from copy import deepcopy
    samples = 120

    # chNegativeTheta = lambda th : th + 360 if th < 0 else th
    # h, inc = th, th//2

    h, inc = samples, 360 / samples
    angles, transCtr = [], [0.0] * h
    interCtr = []
    lCtr = deepcopy(ctr)  # make a local copy so that original is not changed
    lCtr = list(lCtr)
    c = lCtr

    try:
        angles, ctr = oldCtrPtAngles(c, loc)  # values start from 0+ and continue in clockwise direction
        prevTh = int(angles[0] / inc + 0.9)

        for idx in range(1, len(ctr)):
            nxTh = int(angles[idx] / inc)
            for t in range(prevTh, nxTh + 1):
                transCtr[t] = distToCentre(t * inc, loc, ctr, idx)
            prevTh = nxTh + 1

        # now handle the wrap around case of last point (angle < 360) and first point (angle > 0)
        for t in range(prevTh, int(angles[0] / inc) + h + 1):
            transCtr[t % h] = distToCentre(t * inc, loc, ctr, 0)
        interCtr.append(np.array(transCtr))
    except Exception as e:
        raise ValueError  # This should be caught by calling function to remove all contours & show a warning

    return interCtr[0]


def newtransformCtr(ctr, loc, samples) :  # ctrs is a tuple of endo and epi contours (list of points)
    from copy import deepcopy
    # chNegativeTheta = lambda th : th + 360 if th < 0 else th
    # h, inc = th, th//2

    h, inc = samples, 360 / samples
    angles, transCtr = [], [0.0] * h
    interCtr = []
    lCtr = deepcopy(ctr)  # make a local copy so that original is not changed
    lCtr = list(lCtr)
    c = lCtr

    try:
        angles, ctr = ctrPtAngles(c, loc)  # values start from 0+ and continue in clockwise direction
        prevTh = int(angles[0] / inc + 0.9)
        init_th = prevTh
        # print("Ang {}".format(angles[0] / inc))

        for idx in range(1, len(ctr)):
            nxTh = int(angles[idx] / inc)
            for t in range(prevTh, nxTh + 1):
                transCtr[t % samples] = distToCentre(t * inc, loc, ctr, idx)

            prevTh = nxTh + 1
        interCtr.append(np.array(transCtr))
    except Exception as e:
        raise ValueError  # This should be caught by calling function to remove all contours & show a warning
    new_ctr = []
    for i in range(0, samples):
        if (transCtr[(init_th + i) % samples] == 0):
            break
        new_ctr.append(transCtr[(init_th + i) % samples])
    new_ctr = [np.array(new_ctr)]
    return new_ctr,[angles[0],angles[len(angles)-1]]


# Added on : 27-03-23 (Partha Sir)
# -----------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------
def alternateInnerPath(path, cost) :
    THRESHOLD_RATIO = 2 / 3
    h, w = cost.shape

    maxIdx, maxLoc = np.argmax(path), np.max(
        path) + 1  # this is the point that lies on best path and should be included
    cost[cost < THRESHOLD_RATIO * cost[maxIdx, path[maxIdx]]] = 0
    innerPath = np.zeros(h, int)
    for r in range(h) :
        pt = path[r]  # to include point on best path
        while pt > 0 and cost[r, pt - 1] <= cost[r, pt] : pt -= 1
        while pt > 0 and cost[r, pt - 1] > cost[r, pt] : pt -= 1  # Find next peak in cost[r] that is just above path[r]
        if pt == 0 : continue
        innerPath[r] = pt

    seq = largestSequence(innerPath, path)
    newPath = smoothSeq(path, seq, innerPath)
    plt.imshow(np.transpose(cost)), plt.plot(path, 'b-')
    plt.plot(newPath, 'r--'), plt.show()

    return newPath


# -----------------------------------------------------------------------------------------------------------------------
def save(pList, fname) :
    path = r'D:\DSI\PythonProj\Myocardium\TemplateMatch\\'
    file = path + fname
    fp = open(file, 'w')
    for p in pList : fp.write(str(p) + ' ')
    fp.close()


def smoothSeq(path, seq, iPath) :
    newPath = np.zeros(len(path) + 5, int)
    newPath[0 :360] = path
    newPath[360 :365] = newPath[0 :5]
    newPath[seq[0] :seq[0] + seq[1]] = iPath[seq[0] :seq[0] + seq[1]]

    plt.plot(path, 'r--')
    plt.plot(newPath, 'b-'), plt.show()
    for r in range(seq[0], seq[0] + seq[1]) :
        mn = (np.sum(newPath[r - 4 :r + 5]) - newPath[r]) / 8
        if abs(newPath[r] - mn) > 1 :
            newPath[r] = mn

    return newPath[0 :len(path)]


# -----------------------------------------------------------------------------------------------------------------------
def largestSequence(ip, path) :
    lis, lps = [0, 0], [0, 0]
    i = 0
    while i < len(ip) :
        st = i
        if ip[i] == 0 :
            while i < len(ip) and ip[i] == 0 : i += 1
            ps = [st, i - st]
            if ps[0] - (
                    lps[0] + lps[1]) < 3 :  # if the two are adjacent with less than 3 pixel separation then merge them
                ps = [lps[0], ps[0] - lps[0] + ps[1]]
            if ps[1] >= lps[1] : lps = ps
        else :
            while i < len(ip) and ip[i] > 0 and ip[i] < path[i] : i += 1
            isq = [st, i - st]
            if isq[0] - (
                    lis[0] + lis[1]) < 4 :  # if the two are adjacent with less than 4 pixel separation then merge them
                isq = [lis[0], isq[0] - lis[0] + isq[1]]
            if isq[1] >= lis[1] : lis = isq
    return (lis)


# -----------------------------------------------------------------------------------------------------------------------
def getContour(ds) :
    olay, ctr, loc, radius = [0, 0], [0, 0], [0, 0], [0, 0]
    olay[0], olay[1] = ds.overlay_array(0x6000), ds.overlay_array(0x6002)

    for i in range(2) :
        if '3.6' in sys.version :
            _, ctrPList, h = cv2.findContours(olay[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else :
            ctrPList, h = cv2.findContours(olay[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(ctrPList) != 2 :
            print('Problem in detecting contour')
            return [None] * 3

        ctr[i] = ctrPList[0].reshape(-1, ctrPList[0].shape[2])
        m = cv2.moments(ctr[i])
        loc[i] = int(m['m10'] // m['m00']), int(m['m01'] // m['m00'])
        radius[i] = int(math.sqrt(cv2.contourArea(ctr[i]) / 3.1416))
        ctr[i] = list(ctr[i])  # converts the contour points from a numpy array to a Python list
        ctr[i].append(ctr[i][0])  # to make the contour as a closed curve
        ctr[i] = np.array(ctr[i], int)  # converts the contour points back to a numpy array of integers

    loc = (loc[0][0] + loc[1][0]) // 2, (loc[0][1] + loc[1][1]) // 2
    return ctr, loc, radius[0]


def oldTransformCtrBack(ctr,loc,samples=None,bound=None):
    h, fact = len(ctr), math.pi / 180
    inc = 360 / len(ctr)
    tCtr = []

    for t in range(h) :
        th = t * inc
        x, y = loc[0] + ctr[t] * math.cos(th * fact), loc[1] + ctr[t] * math.sin(th * fact)
        tCtr.append((x, y))

    tCtr.append(tCtr[0])  # to make it a closed curve

    tCtr = [(pt[0], pt[1]) for pt in tCtr]  # to take care of tructation error
    return np.array(tCtr)

# -----------------------------------------------------------------------------------------------------------------------
def transformCtrBack(ctr, loc, samples, bound) :
    h, fact = len(ctr), math.pi / 180
    first_angle = bound[0]
    last_angle = bound[1]
    inc = 360 / samples
    tCtr = []

    for t in range(len(ctr)):
        th = first_angle + (t * inc)
        x = loc[0] + ctr[t] * math.cos(th * fact)
        y = loc[1] + ctr[t] * math.sin(th * fact)
        tCtr.append((x, y))

    # tCtr.append(tCtr[0])  # to make it a closed curve
    if ((first_angle % 360) == (last_angle % 360)):
        tCtr.append(tCtr[0])
    tCtr = [(pt[0], pt[1]) for pt in tCtr]  # to take care of tructation error
    return np.array(tCtr)


# -----------------------------------------------------------------------------------------------------------------------
def plotCtrOnImage(fname, img, edgeIm, nCtrs, oCtrs, loc, imName, DICE=None, radius=None) :
    os = int(img.shape[1] * 0.125, img.shape[0] * 0.125) if radius is None else \
        (2 * radius + 10, int(2 * radius * img.shape[0] / img.shape[1]) + 10)
    # select an area of image within a range of 40 from the centre (location)
    cCtrs, oCtrs = nCtrs.copy(), oCtrs.copy()
    subImg = img[loc[1] - os[1] :loc[1] + os[1], loc[0] - os[0] :loc[0] + os[0]]
    fact = img.shape[1] / subImg.shape[1], img.shape[0] / subImg.shape[0]
    subImg = cv2.resize(subImg, (img.shape[1], img.shape[0]))  # to avoid pixalation affect

    nCtrs[0] = nCtrs[0] - [loc[0] - os[0], loc[1] - os[1]]
    # nCtrs[1] = nCtrs[1] - [loc[0] - os[0], loc[1] - os[1]]
    nCtrs[0]  = nCtrs[0] * fact
    # nCtrs[1] = nCtrs[1] * fact
    oCtrs[0]  = oCtrs[0] - [loc[0] - os[0], loc[1] - os[1]]
    # oCtrs[1] = oCtrs[1] - [loc[0] - os[0], loc[1] - os[1]]
    oCtrs[0] = oCtrs[0] * fact
    # oCtrs[1] = oCtrs[1] * fact
    # Since (loc[0]-os, loc[1]-os) is the new origin of the image being displayed
    SF.showImage(subImg, 'Endocardium - original (green), Refined (red)', 121), SF.plotCurves([oCtrs[0]], color='g-')
    plt.plot([os[0] * fact[0]], [os[1] * fact[1]], 'ro'), SF.plotCurves([nCtrs[0]], color='r-')

    # SF.showImage(subImg, 'Epicardium - original (green), Refined (red)', 122), SF.plotCurves([oCtrs[1]], color='g-')
    # plt.plot([os[0] * fact[0]], [os[1] * fact[1]], 'ro'), SF.plotCurves([nCtrs[1]], color='r-')

    fParts = str(fname).split('\\')
    if not ONECARDIO: imName = fParts[-4] + '-' + fParts[-3] + '-' + fParts[-1]
    else: imName = 'Endocardium'
    plt.suptitle('Image is: {} with DICE = {}'.format(imName, DICE))

    fname = str(fname).split('.')[0]
    if not ONECARDIO: plt.savefig(fname + '.png')  # , plt.close()
    SF.showImage(None, show=True, block=True, maximized=True)


# -----------------------------------------------------------------------------------------------------------------------
def adjustContour_points(cen, contours, value, pixel_spcing) :
    try :
        contours = np.resize(contours, (len(contours), 2))
        xpoints = []
        ypoints = []
        if contours.max() == 0 :
            return [0] * 6, [0] * 6
        if value == 'Epicardium' :
            for pt in contours :
                angle = (math.atan2(pt[1] - cen[1], pt[0] - cen[0]))
                xpoints.append(pt[0] + pixel_spcing * math.cos(angle) / 2)
                ypoints.append(pt[1] + pixel_spcing * math.sin(angle) / 2)
        if value == 'Endocardium' :
            for pt in contours :
                angle = (math.atan2(pt[1] - cen[1], pt[0] - cen[0]))
                xpoints.append(pt[0] - pixel_spcing * math.cos(angle) / 2)
                ypoints.append(pt[1] - pixel_spcing * math.sin(angle) / 2)
    except :
        return None
    xpoints = [int(x * 100) / 100 for x in xpoints]
    ypoints = [int(y * 100) / 100 for y in ypoints]
    return xpoints, ypoints


# -----------------------------------------------------------------------------------------------------------------------
def addOverlay_points(ds, ctrs, loc) :
    try :
        # add private block for storing epi- and endo- contour points
        block1 = ds.private_block(0x000b, "X-Y coordinates points", create=True)
        pixel_spcing = float(ds.PixelSpacing[0])
        # Endocardium points. Pixel spacing is not added as it is already at the border
        xpoints, ypoints = adjustContour_points(loc, ctrs[0], 'Endocardium', 0)
        x_p = " ".join(str(e) for e in xpoints)
        y_p = " ".join(str(e) for e in ypoints)
        rval = block1.add_new(0x02, "LT", x_p)
        block1.add_new(0x03, "LT", y_p)
        # Epicardium points
        xpoints, ypoints = adjustContour_points(loc, ctrs[1], 'Epicardium', 0)
        x_p = " ".join(str(e) for e in xpoints)
        y_p = " ".join(str(e) for e in ypoints)
        block1.add_new(0x04, "LT", x_p)
        block1.add_new(0x05, "LT", y_p)
    except :
        return None

    return None


# -----------------------------------------------------------------------------------------------------------------------
def getImageFileName(fPath) :
    # It makes a list of all images for one patient and returns that. In the next call it will work for
    # next  patient and then next hospital, till all patients for all hospitals are covered
    for hospital in os.listdir(fPath) :
        hPath = fPath / hospital
        for pat in os.listdir(hPath) :
            patPath = hPath / pat
            os.chdir(patPath)
            imgList = []
            for img in os.listdir(patPath / 'HHT_LGE_SAX_1') :
                imgList.append((patPath, img))
            yield imgList


# -----------------------------------------------------------------------------------------------------------------------
def saveImage(folder, imName, ds, ctrs, loc) :
    addOverlay_points(ds, ctrs, loc)
    fName = folder / imName
    os.makedirs(folder, exist_ok=True)
    ds.save_as(str(fName))


# -----------------------------------------------------------------------------------------------------------------------
def showOrigWithMask(fname, img, ctr) :
    myoMask = img.copy()
    # ctr = np.array(ctr[0]).astype(int), np.array(ctr[1]).astype(int)
    # myoMask = cv2.drawContours(myoMask, ctr, -1, color=255, thickness=cv2.FILLED)
    fParts = str(fname).split('\\')
    imName = fParts[-4] + '-' + fParts[-3] + '-' + fParts[-1]
    SF.showImage(img, imName + ' - Original', 121)
    SF.showImage(myoMask, 'with Myo(refined)', 122)
    SF.plotCurves([ctr[0]], color='r-')
    # SF.plotCurves([ctr[1]], color='r-')
    fname = str(fname).split('.')[0]
    plt.savefig(fname + '.png')  # , plt.close()
    SF.showImage(None, show=True, block=True, maximized=True)
    return


# -----------------------------------------------------------------------------------------------------------------------
def readImage(folder, SOPId, GTImage) :
    global data
    for imName in os.listdir(folder) :
        ds = pd.dcmread(folder / imName)  # read the dicom image
        ds.decompress()
        if ds.SOPInstanceUID == SOPId : break

    if ds.SOPInstanceUID == SOPId :
        print('Image file name is: ', imName)
        data['Pat ID'] = ds.PatientID
        data['St UID'] = ds.StudyInstanceUID
        data['Slice Loc'] = ds.SliceLocation
        data['LGE Image'] = str(imName)
        img = ds.pixel_array.copy()  # extract the pixel data from the dicom image file
        wc, ww = ds.WindowCenter, ds.WindowWidth
        img[img < wc - ww // 2] = wc - ww // 2
        img[img > wc + ww // 2] = wc + ww // 2
        img = ((img - (wc - ww // 2)) / ww * 255).astype(np.uint8)
    else :
        print('No matching LGE image for ground truth image: ', GTImage)
        return None, None, None

    return ds, img, imName

def getRadius(ctr) :
    loc,radius=[0,0],0
    ctr = np.array(ctr,int)
    ctr = ctr.reshape(-1,2)
    m = cv2.moments(ctr)
    loc = int(m['m10'] // m['m00']), int(m['m01'] // m['m00'])

    radius = int(math.sqrt(cv2.contourArea(ctr) / 3.1416))
    ctr = list(ctr)  # converts the contour points from a numpy array to a Python list
    # ctr.append(ctr[0])  # to make the contour as a closed curve
    ctr = np.array(ctr, int)
    return ctr, radius

# -----------------------------------------------------------------------------------------------------------------------
def getPartialCtr(ctr, fstIdx, lstIdx):
    if fstIdx < lstIdx:         pCtr = ctr[fstIdx:lstIdx]
    else:                       pCtr = np.concatenate((ctr[fstIdx:], ctr[:lstIdx]))
    return pCtr


def updateBPCost(costArr, mTheta, mDist, fDist, lDist):
    for i in [mTheta-10,mTheta,mTheta+10]:
        for j in range(len(costArr[0])):
            if (i==mTheta-10 and j==fDist) or (i==mTheta and j==mDist) or (i==mTheta+10 and j==lDist):
                costArr[i][j]=np.max(costArr)
            elif j==fDist or j==mDist or j==lDist:
                costArr[i][j]=0




def mainForDP(image_data, contour, ctrCentre,userPoint, mouseTheta, patient, GT, type='HHT_LGE') :
    global fp, data
    DPObj = DynProg()
    ctrsRef = [None, None]
    if not ONECARDIO:
        path = patient if not ONECARDIO else patient.rootFolder
        folders = [folder for folder in os.listdir(path) if type in folder and "Dummy" not in folder]
    else:
        folders = ['dummy']
        GT = ['dummy']

    for folder in folders :
        for SOPId in GT :
            if ONECARDIO:
                ds, img, LGEImName = image_data[0], image_data[1], image_data[2]
                print("Returned Array: {}".format(img))
            else:
                ds, img, LGEImName = readImage(path / folder, SOPId, GT[SOPId][0])
            if ds is None : continue
            if not ONECARDIO:
                ctrs, loc, rad = getContour(ds)  # rad is radius of endocardium
                if ctrs is None:
                    print('No contour detected for image: ', path / folder / LGEImName, 'Skipping)')
                    continue  # No contours in the image
                ctr = ctrs[0]
                ctr = ctrs[0][:len(ctrs[0]) // 2]
                loc = getCentre(ctr)

            else:
                ctr = contour
                # ctr = ctrs[0][:len(ctrs[0]) // 2]
                loc = ctrCentre
                loc = getCentre(np.asarray(contour))
                ctr, rad = getRadius(ctr)

            fstIdx, lstIdx = get_StartEndIdx(loc, ctr[0], ctr[-1], SAMPLE_PTS)
            first_angle, last_angle = fstIdx * 360 // SAMPLE_PTS, lstIdx * 360 / SAMPLE_PTS
            tCtr = transformCtr(ctr, loc, 120)
            # Cartesian coordinate system to Polar coordinate system
            #It uses the old function that works on a closed contour

            strip, BPmean = SF.getStrip(img, loc, rad - 10, rad + 30, samples=SAMPLE_PTS)
            # tCtr, strip = getPartialCtr(tCtr, fstIdx, lstIdx), getPartialCtr(strip, fstIdx, lstIdx)
            # Transform the image into polar coordinates with #SAMPLE_PTS along the contour and for radius
            # from rad-10, rad+25. Its dimension is (SAMPLE_PTS, 20)

            # -----------------------------------Modification------------------------------------------------------------
            for i in range(1) :  # for both endo and epicardium contours
                # DPObj.setupGraph(strip, BPmean, MCmean=45, tCtr=tCtr, rad=rad)
                if ONECARDIO:
                    mouseDist = math.sqrt((ctrCentre[0] - userPoint[0]) ** 2 + (ctrCentre[1] - userPoint[1]) ** 2)
                    mouseDist = mouseDist - rad + 10
                    # Correct for user input point
                    # updateBPCost(DPObj.BPCost,mTheta=mouseTheta,mDist=mouseDist,fDist=tCtr[fstIdx],lDist=tCtr[lstIdx])

                    tCtr, strip = getPartialCtr(tCtr, fstIdx, lstIdx), getPartialCtr(strip, fstIdx, lstIdx)

                    for i in [tCtr[0],tCtr[-1]]:
                        strip[0,i]=1
                        for j in range(1,len(strip)):
                            strip[j,i]=0

                    DPObj.setupGraph(strip, BPmean, MCmean=45, tCtr=tCtr, rad=rad, adjustPars=[mouseTheta,mouseDist,tCtr[fstIdx],tCtr[lstIdx]])
                    ctrCorrected = DPObj.runDynPorg()
                else:
                    tCtr, strip = getPartialCtr(tCtr, fstIdx, lstIdx), getPartialCtr(strip, fstIdx, lstIdx)
                    DPObj.setupGraph(strip, BPmean, MCmean=45, tCtr=tCtr, rad=rad)
                    ctrCorrected = DPObj.runDynPorg()
                ctrsRef[i] = transformCtrBack(ctrCorrected,loc,120,[first_angle,last_angle])  # Polar coordinate system to the original Cartesian coordinate system
                if ONECARDIO:
                    global data
                    imName = "Endocardium"
                    plotCtrOnImage(imName, img, DPObj.edgeIm, ctrsRef, [ctr], loc, LGEImName, DICE=85, radius=DPObj.rad)
                    return ctrsRef[0]

            save_folder = path / folder if ONECARDIO else path / 'RefinedContours'
            saveImage(save_folder, LGEImName, ds, ctrsRef, loc)

            if not ONECARDIO :
                global data
                dice = DL_dice = 0
                data['GT Image'] = GT[SOPId][0]
                data['DC w/o Refinement'] = DL_dice = calDICE(path, LGEImName, GT[SOPId][1], refine=False)
                data['DC with Refinement'] = dice = calDICE(path, LGEImName, GT[SOPId][1])
                save_folder = path / 'Result'
                os.makedirs(save_folder, exist_ok=True)
                # temp_ctr= [ctr]
                # showOrigWithMask(save_folder / ('Mask' + LGEImName), img, ctrsRef)
                print("Name {}".format(LGEImName))
                imName = save_folder / ('Ctr_' + LGEImName)
                plotCtrOnImage(imName, img, DPObj.edgeIm, ctrsRef, [ctr], loc, LGEImName, DICE=dice, radius=DPObj.rad)

                line = ''
                for fn in data.keys() :
                    line += str(data[fn]) + ', '
                print(line)
                fp.write(line[:-2] + '\n')  # write data for the image into the csv file pointer fp


# -----------------------------------------------------------------------------------------------------------------------

def createCSV(file):
    global fp
    if fp is None:            fp = open(file, 'w')
    fields = ['Hospital', 'Pat ID', 'St UID', 'Pat Folder', 'LGE Slice Loc', 'GT image', 'LGE Image',
              'DC w/o Refinement', 'DC with Refinement']
    line = ''
    for i in range(len(fields)):
        line += fields[i] + ', '
    fp.write(line[:-2] + '\n')



def mainFunc(contour=None, ctrCentre=None, userPoint=None, mouseTheta=None, image_data=None, standalone=None):
    global ONECARDIO
    ONECARDIO = not standalone
    root = Path(r'.')
    if ONECARDIO:
        # path_folder = dirs[0] = "E:\\Work\\HDP-Working\\HDP\\KMC\\440991\\HHT_LGE_SAX_1"
        # S0PId = dirs[1] =  "1.3.46.670589.11.33392.5.20.1.1.2028.2021010614543136408.1"
        # image_file = dirs[2] = "image_03.dcm"
        refinedContour = mainForDP(image_data, contour, ctrCentre, userPoint, mouseTheta, None, None)
        return refinedContour

    createCSV(root / 'DICE_Scores.csv')
    for hosp in ['KMC'] :  # os.listdir(root):
        if (root / hosp).is_dir() is False : continue
        data['Hospital'] = hosp
        for pat in os.listdir(root / hosp) :
            data['Pat Folder'] = str(root / hosp / pat)
            GT = readGroundTruth(root / hosp / pat)
            mainForDP(None, None, None, None, None, root / hosp / pat, GT)

    fp.close()
# -----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__' :
    mainFunc(standalone=True)