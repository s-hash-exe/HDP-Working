import pydicom as pd
import numpy as np
import os
from tqdm import tqdm
from prep import get_mask, get_mask_from_dcm_overlay

def readGroundTruth(folder):
    GT = {}  # dictionary to store masks of ground truth for each image ins id
    folder = folder / 'ground truth'
    for imName in os.listdir(folder):
        ds, img, sh, SOPId = get_attr(folder/imName)  # read the dicom image
        if ([0x000b, 0x1002] not in ds) and ([0x000b, 0x1004] not in ds) :
            GT[SOPId] = [imName, get_mask_from_dcm_overlay(folder/imName)]
        else:
            GT[SOPId] = [imName, get_mask(folder / imName, ds, img, sh)]  # Ground Truth mask
    return GT

def find_files(extension='', root='.'):
    file_paths = []
    for dirpath, dirnames, files in tqdm(os.walk(root)):
        for name in files:
            # if name.endswith(extension):
            file_paths.append(os.path.join(dirpath, name))
    return file_paths

# Pred and true and binary numpy arrays
def dice(pred, true, k = 1):
    smooth =0.001
    intersection = np.sum(pred[true==k]) * 2.0
    dice = (intersection + smooth) / ((np.sum(pred) + np.sum(true) )+ smooth)
    return dice

def get_attr(file):
    """
    To get attributes from the image required to get mask
    """
    ds = pd.dcmread(file)
    ds.decompress()
    img = ds.pixel_array
    shape = img.shape
    return ds, img, shape, ds.SOPInstanceUID

def calDICE(main_path, lge_file, gt_myo_mask, refine=True):
    try:
        pred_lge_path = main_path / 'RefinedContours' # Predicted o/p path
        if refine is False: pred_lge_path = main_path / 'HHT_LGE_SAX_1'  # Predicted o/p path
        ds_mod, img_mod, shape_mod, SOPId_mod = get_attr(pred_lge_path / lge_file)
        if ([0x000b, 0x1002] not in ds_mod) or ([0x000b, 0x1004] not in ds_mod):
            pred_myo_mask = get_mask_from_dcm_overlay(pred_lge_path / lge_file)
        else:
            pred_myo_mask = get_mask(pred_lge_path / lge_file, ds_mod, img_mod, shape_mod) # Predicted o/p mask

        dice_val = dice(gt_myo_mask, pred_myo_mask) * 100
        sl_loc = ds_mod.SliceLocation
        print('Dice score: %d for image %s at slice location %d'%(dice_val, lge_file, sl_loc))
    except:
        print('Can not calculate dice value for ', lge_file)
        return 0

    return dice_val




