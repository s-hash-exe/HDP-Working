import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
# plt.set_cmap('gray')    # to set default colour for matplotlib to gray
from scipy.signal import savgol_filter

BLOCK = 'Wait'  # turn block = Wait for closing the window with button press, block=True to wait or block=False to come out
DISPLAY = True # To turn off or on display of intermediate image by setting global variable DISPLAY

#-------------------------------------------------------------------
def plotCurves(ctrs, index=-1, show=False, block=BLOCK, color='r-'):
    if DISPLAY:     # The variable pydevd is true if program is running in debug mode
        c = [ctrs[index]] if index != -1 else ctrs
        for ctr in c:
            if ctr is None:
                return
            if ctr.ndim == 2:
                plt.plot(ctr[:,0], ctr[:, 1], color)
            else:
                plt.plot(ctr)
        if show:
            if block == 'Wait':
                plt.text(20, 20, 'Click on the image or press a key to continue', fontsize=14, color='red')
                plt.waitforbuttonpress(), plt.close()

            elif block == True:    plt.show(block=True)
            else:                  plt.show(block=False)
        return

#-------------------------------------------------------------------
SF_FigureId = 0
def showImage(im, title=None, numWin = None, show=False, block=BLOCK, createFig=False, maximized=False,timeout=1):
    #if 'pydevd' in sys.modules and DISPLAY:     # The variable pydevd is true if program is running in debug mode
    if DISPLAY:     # The variable pydevd is true if program is running in debug mode
        if createFig:
            SF_FigureId = plt.figure(createFig)
            if title is not None: plt.suptitle(title)
        if numWin != None: plt.subplot(numWin)
        if im is not None: plt.imshow(im)
        if title is not None: plt.title(title)
        if show:
            # if maximized:                   plt.get_current_fig_manager().window.showMaximized()
            if block=='Wait':               plt.waitforbuttonpress(), plt.close()
            elif block == True:             plt.show(block=True)
            elif block == 'Timeout':
                plt.show(block=False)
                plt.pause(timeout)
                plt.close()
            return

#-------------------------------------------------------------------
def resizeIm(img, size):
    if size <= 0: return(img)
    imsize = img.shape
    if len(imsize) != len(size): raise  # the two dimensions must be same
    # exception is caught in calling function "readImage"
    if abs(imsize[0] / size[0] - imsize[1] / size[1]) > 0.01: raise
    # the scale factor is not maintained hence raising exception
    img = cv2.resize(img,size,interpolation=cv2.INTER_LINEAR)
    return img
#-------------------------------------------------------------------
def readImage(file, size=0):

    try:
        if file[-4:] == '.dcm':
            dimg = pydicom.dcmread(file)   # read the dicom image
            imgf = dimg.pixel_array     # extract the pixel data from the dicom image file
            max = np.max(imgf)
            img = np.uint8(imgf * (254 / max))  # to create ndarray of bytes needed for displaying image
        else:

            img = cv2.imread(file, 0)
            #if img == None: print('NULL returned by imread()')
            img = cv2.resize(img, (256, 256))
        if img.shape != size: img = resizeIm(img, size)
    except:
        print('Problem in reading file %s.', file)

    return (img)
#--------------------------------------------------------------------------------

SPREAD = 4  # For calculating the mean value of blood pool at the centre of myocardium
def getStrip(img, loc, rmin=3, rmax=25, samples=360, overlap=0):
    strip, inc = np.zeros((samples+2*overlap, rmax), int), 360/samples

    if rmin < 2: rmin = 2
    for theta in range(samples):
        th = theta * inc * 2 * np.pi / 360
        for r in range(rmin, rmax):
            x = loc[0] + r * np.cos(th)
            y = loc[1] + r * np.sin(th)
            strip[overlap+theta, r] = interpolate(img, x, y)

    if overlap > 0:
        strip[:overlap] = strip[samples:samples+overlap]
        strip[samples+overlap:] = strip[overlap:2*overlap]

    for r in range(rmin):
        strip[:,r] = strip[:,rmin]
    mean = np.mean(strip[overlap:samples+overlap, rmin:rmin+SPREAD])
    return (strip, int(mean))

#---------------------------------------------------------------------------
def interpolate(img, x, y):
    x,y=int(x),int(y)
    lx, ly = int(x), int(y) # lower (floor) of x and y which are float number
    ux, uy = lx+1, ly+1     # upper (ceiling) of x and y
    # print("Indices: {} {} {} {} ".format(lx,ly,ux,uy))
    lowerVal = img[ly, lx] * (ux-x) + img[ly, ux] * (x-lx)
    upperVal = img[uy, lx] * (ux-x) + img[uy, ux] * (x-lx)
    val = int(lowerVal * (uy-y) + upperVal * (y-ly) + 0.5)
    return val
# -----------------------------------------------------------------------
def showBoundaries(img, endoPts, loc, rmin, numBrd=1, periPts=[], title=[]):
    plt.figure(1)
    if title == []: title = 'Image'
    else:
        name_parts = title[:-4].split('\\')
        title = name_parts[-2]+'_'+name_parts[-1]
    showImage(img, title=title)
    showCtrOnImg(endoPts, loc)
    if numBrd > 1 and periPts != []:   showCtrOnImg(periPts, loc)
    showImage(im=None, show=True)

# -------------------------------------------------------------------------
def getBdr(bpts, loc):
    w = len(bpts)
    pts = []
    numPts = len(bpts)
    stPos = 0
    while bpts[stPos] == -1: stPos += 1
    # this is done in case boundary is detected for only a part of myocardium, not covering 360 deg
    for t in range(w):
        if t < stPos: continue
        th = t * 2*np.pi / w
        x = int(loc[0] + (bpts[t]) * np.cos(th) + 0.5)
        y = int(loc[1] + (bpts[t]) * np.sin(th) + 0.5)
        pts.append([x, y])
    pts.append(pts[0])  # to make it a closed contour
    pts = np.array(pts, np.uint8)
    return (pts)
# -------------------------------------------------------------------------
def showCtrOnImg(bpts, loc):
    pts = getBdr(bpts, loc)
    plt.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1)

#--------------------------------------------------------------------
MASK_SIZE = 3
mask = np.array([1] * MASK_SIZE + [-1] * MASK_SIZE) / MASK_SIZE
# mask polarity  (order of 1 and -1) will depend on whether the central region is white or dark

def getEdgeMap(strip, edgeDir=0): # applying edge detector
    # edgeDir: 1 - positive for radially out; -1 - positive for radially out; 0 for absolute edge values
    levels, nodes = strip.shape[:2]
    corim = np.zeros((levels, nodes), int)
    corim[:,0] = 1  # The first col on each row is set to 1 so that if there are no edges on a row then also there
    # will be one non-zero pixel

    for lvl in range(levels):
        for nd in range(MASK_SIZE, nodes-MASK_SIZE):
            corim[lvl, nd] = int(np.sum(strip[lvl, nd-MASK_SIZE:nd+MASK_SIZE] * mask))

    if edgeDir == -1:  corim *= edgeDir
    if edgeDir == 0:   corim = np.abs(corim)
    return (corim)
#------------------------------------------------------------
OVERLAP = 10
def smoothBoundary(pts):
    h = pts.shape[0]
    extEdges = np.zeros((h+2*OVERLAP), int)
    extEdges[OVERLAP:h+OVERLAP] = pts
    extEdges[0:OVERLAP] = pts[h-OVERLAP:]
    extEdges[h+OVERLAP:] = pts[0:OVERLAP]

    smooth_intpts = np.array(savgol_filter(extEdges,101,3), int)
    return(smooth_intpts[OVERLAP:h+OVERLAP])

# --------------------------------------------------------------
