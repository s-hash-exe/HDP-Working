import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from pydicom import DataElement
from pydicom.pixel_data_handlers.numpy_handler import pack_bits
# plt.set_cmap('gray')
import math

# ----------------------------------------------------------------------------------------------------------------------
def getCenter(cntr):
    try:
        if len(cntr) == 1: pass
    except:     # if cntr is None, then it will lead to exception. Return none as contour is not there
        return None
    m = cv2.moments(cntr)
    cx = int(m['m10'] // m['m00'])
    cy = int(m['m01'] // m['m00'])
    return ((cx, cy))
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def dcm_overlay_prep(ds, shape, address):
    ds.add(DataElement(address + 0x0010, 'US', shape[0]))
    ds.add(DataElement(address + 0x0011, 'US', shape[1]))
    ds.add(DataElement(address + 0x0015, 'IS', "1"))
    ds.add(DataElement(address + 0x0022, 'LO', "Annotation made by user"))
    ds.add(DataElement(address + 0x0040, 'CS', "G"))
    ds.add(DataElement(address + 0x0050, 'SS', [1, 1]))
    ds.add(DataElement(address + 0x0051, 'US', 1))
    ds.add(DataElement(address + 0x0100, 'US', 1))
    ds.add(DataElement(address + 0x0102, 'US', 0))
    return ds
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------


def get_mask(file,ds,img,shape):
    try:
        pixel_spacing = float(ds.PixelSpacing[0])
        # will read endocardium contour points
        #these points will be in float. directly coming from user/doctor
        if ([0x000b, 0x1002] in ds):
            endo_x, endo_y = ds[0x000b1002].value, ds[0x000b1003].value
            try: endo_x, endo_y = endo_x.decode("utf-8"), endo_y.decode("utf-8")
            except: pass
            temp_x, temp_y = endo_x.split(" "), endo_y.split(" ")
            if "" in temp_x:
                temp_x.remove("")
            if "" in temp_y:
                temp_y.remove("")

            temp_x = list(map(float, temp_x))
            temp_y = list(map(float, temp_y))

            l1, l2 = np.array(temp_x, float), np.array(temp_y, float)
            endo_xy = []
            for i in range(len(l1)):
                endo_xy.append([round(l1[i]), round(l2[i])])


        #will read epicardium
        if ([0x000b, 0x1004] in ds):
            epi_x, epi_y = ds[0x000b1004].value, ds[0x000b1005].value
            try: epi_x, epi_y = epi_x.decode("utf-8"), epi_y.decode("utf-8")
            except: pass
            temp_x, temp_y = epi_x.split(" "), epi_y.split(" ")
            if "" in temp_x:
                temp_x.remove("")

            if "" in temp_y:
                temp_y.remove("")
            temp_x = list(map(float, temp_x))
            temp_y = list(map(float, temp_y))

            l1, l2 = np.array(temp_x, float), np.array(temp_y, float)
            epi_xy = []
            for i in range(len(l1)):
                epi_xy.append([round(l1[i]), round(l2[i])])

        if max(endo_xy) == 0 or max(epi_xy)== 0:
            return None

        cen = getCenter(np.asarray(endo_xy))
        endo_xy1 =[]
        for pt in endo_xy:
            angle = (math.atan2(pt[1] - cen[1], pt[0] - cen[0]))
            endo_xy1.append((round(pt[0] + (pixel_spacing) * math.cos(angle)/2), round(pt[1] + (pixel_spacing) * math.sin(angle)/2)))

        cen = getCenter(np.asarray(epi_xy))
        epi_xy1 = []
        for pt in epi_xy:
            angle = (math.atan2(pt[1] - cen[1], pt[0] - cen[0]))
            epi_xy1.append((round(pt[0] - (pixel_spacing) * math.cos(angle)/2), round(pt[1] - (pixel_spacing) * math.sin(angle)/2)))


        ctr_endo = np.array(endo_xy1).reshape((len(endo_xy1),1,2)).astype(np.int32)
        ctr_epi  = np.array(epi_xy1).reshape((len(epi_xy1),1,2)).astype(np.int32)

        # To obtain the mask of myocardium which will be stored in the pickle file, for training
        mask = np.zeros(shape, dtype=np.int32)
        mask =cv2.drawContours(np.zeros_like(img).astype(np.uint8), [ctr_endo,ctr_epi], -1, color=1, thickness=cv2.FILLED)
        mask = cv2.resize(mask,dsize=(192, 192), interpolation=cv2.INTER_LINEAR)
        # plt.imshow(mask)
        # plt.show()

        # To obtain the adjusted mask of endocardium and store it as an overlay in the image
        inner_mask = np.zeros(shape, dtype=np.int32)
        inner_mask =cv2.drawContours(np.zeros_like(img).astype(np.uint8), [ctr_endo], -1, color=1, thickness=1)
        # plt.imshow(inner_mask)
        # plt.show()
        ds = dcm_overlay_prep(ds,shape, 0x60000000)
        packed_bytes = pack_bits(inner_mask.flatten())
        if len(packed_bytes) % 2:
            packed_bytes += b'\x00'
        ds.add(DataElement(0x60003000, 'OW', packed_bytes))

        # To obtain the adjusted mask of epicardium and store it as an overlay in the image
        outer_mask = np.zeros(shape, dtype=np.int32)
        outer_mask =cv2.drawContours(np.zeros_like(img).astype(np.uint8), [ctr_epi], -1, color=1, thickness=1)
        # plt.imshow(outer_mask)
        # plt.show()

        ds = dcm_overlay_prep(ds,shape, 0x60020000)
        packed_bytes = pack_bits(outer_mask.flatten())
        if len(packed_bytes) % 2:
            packed_bytes += b'\x00'
        ds.add(DataElement(0x60023000, 'OW', packed_bytes))
        ds.save_as(file)
        return mask
    except Exception as e:
        print(file,e)


# ----------------------------------------------------------------------------------------------------------------------
def get_mask_from_dcm_overlay(file_path):

    try:
        ds = pydicom.dcmread(file_path)
    except:
        return None
    try:
        arr1 = ds.overlay_array(0x6000)
        arr2 = ds.overlay_array(0x6002)
    except:
        print("Could not read one or both overlays")
        return None

    import sys
    if '3.6' in sys.version:
        _, contours1, hierarchy = cv2.findContours(arr1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours1, hierarchy = cv2.findContours(arr1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours1) > 1:
        contours1 = contours1[:1]
    if '3.6' in sys.version:
        _, contours2, hierarchy = cv2.findContours(arr2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours2, hierarchy = cv2.findContours(arr2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours2) > 1:
        contours2 = contours2[:1]
    mask_myo = cv2.drawContours(np.zeros_like(arr1), contours1 + contours2, -1,
            color=1, thickness=cv2.FILLED)
    mask_myo = cv2.resize(mask_myo, dsize=(192, 192), interpolation=cv2.INTER_LINEAR)

    return mask_myo



# root = r'F:\Annotation -Training\OneCardio_Studies_Testing\static'
# f = 'image_09.dcm'
# ds = pydicom.dcmread(root + '/' + f)
# img = ds.pixel_array
# shape = img.shape
# pixel_spcing = float(ds.PixelSpacing[0])
# get_mask(file,ds,img,shape)


























# 'static\\3388100\\HHT_LGE_SAX_1\\image_05.dcm'
