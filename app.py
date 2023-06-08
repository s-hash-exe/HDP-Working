import sys, json, os
import pydicom as pd
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from pathlib import Path
import numpy as np

sys.path.insert(0, r'E:\Work\HDP-Working')
sys.path.insert(0, r'E:\Work\HDP-Working\HDP')
sys.path.insert(0, r'E:\Work\DeltaNudgeTool-py')

from HDP import RefineContourMyoInitial
from HDP import RefineContourMyo
import contourMgr

app = Flask("DNT")
CORS(app)

# -----------------------------------------------------------------------------------------------------------------------
def readImage(folder, SOPId, GTImage) :
    global data
    for imName in os.listdir(folder) :
        ds = pd.dcmread(folder / imName)  # read the dicom image
        ds.decompress()
        if ds.SOPInstanceUID == SOPId : break

    if ds.SOPInstanceUID == SOPId :
        print('Image file name is: ', imName)
        # data['Pat ID'] = ds.PatientID
        # data['St UID'] = ds.StudyInstanceUID
        # data['Slice Loc'] = ds.SliceLocation
        # data['LGE Image'] = str(imName)
        img = ds.pixel_array.copy()  # extract the pixel data from the dicom image file
        wc, ww = ds.WindowCenter, ds.WindowWidth
        img[img < wc - ww // 2] = wc - ww // 2
        img[img > wc + ww // 2] = wc + ww // 2
        img = ((img - (wc - ww // 2)) / ww * 255).astype(np.uint8)
    else :
        print('No matching LGE image for ground truth image: ', GTImage)
        return None, None, None

    return ds, img, imName

# Information from frontend
# Contour, UserPoint, Image
# We need image from frontend itself as it is required in the FLASK API which is running locally

# This function receives the contour from frontend
@app.route("/sendContour", methods=['POST'])
@cross_origin()
def getContour():
    if request.method == 'POST':
        data = request.get_json()
        data = json.loads(data)  # data[0] - mousePoint ; data[1] - contour information

        mousePoint = data[0]
        # img = data[2]
        contour = []             # contour - list of contour points
        for point in data[1]:
            contour.append([point['x'], point['y']])
        contour = [[169, 137],[178, 137],[183, 142],[183, 145],[185, 147],[185, 150],[183 ,152],[183 ,156],[177, 162],[173, 162],[172, 163],[165, 163],[159, 157],[159, 154],[157, 152],[157, 151],[159, 149],[159, 145],[165, 139],[167, 139],[169, 137]]

        # Here we pass the full contour along with user point to contourMgr.
        # Output -> Open contour's first and last index in the spead of 15 deg on each side

        firstPointIdx, lastPointIdx, ctrCentre, mouseDir = contourMgr.mainFunc(contour, mousePoint)

        print("First point index: {}".format(firstPointIdx))
        print("Last point index: {}".format(lastPointIdx))
        print("Mouse Direction: {}".format(mouseDir))

        # Extracting the contour between firstPointIdx and lastPointIdx

        openContour = []
        for i in range(lastPointIdx,lastPointIdx+len(contour)):
            openContour.append(contour[i%len(contour)])
            if(i%len(contour)==firstPointIdx):
                break

        tCtr = contour[:len(contour)//2]
        print("Contour: {}".format(tCtr))

        # Here was pass open contour to RefineContourMyo
        # Output -> Refined Contour

        dirs=["E:\\Work\\HDP-Working\\HDP\\KMC\\440991\\HHT_LGE_SAX_1","1.3.46.670589.11.33392.5.20.1.1.2028.2021010614543136408.1","image_03.dcm"]

        ds, img, LGEImName = readImage(Path(dirs[0]), dirs[1], dirs[2])
        print("Read Image: {}".format(img))
        refinedContour = RefineContourMyoInitial.mainFunc(tCtr, ctrCentre, mousePoint, mouseDir, [ds,img,LGEImName], False)
        print("Refined Contour: {}".format(refinedContour))


        # print("Received first and last point from contourMgr : {}, {}".format(firstPoint, lastPoint))

        # file1 = open('myfile.txt', 'w')
        # for x in contour:
        #     file1.write("{} {}\n".format(x[0],x[1]))
        # file1.write("BREAK\n")
        # file1.write("{} {}\n".format(firstPoint[0], firstPoint[1]))
        # file1.write("BREAK\n")
        # file1.write("{} {}\n".format(lastPoint[0], lastPoint[1]))
        # file1.write("BREAK\n")
        # file1.write("{} {}\n".format(mousePoint[0], mousePoint[1]))
        # file1.write("BREAK\n")
        # file1.write("{} {}\n".format(ctrCentre[0], ctrCentre[1]))
        # file1.write("BREAK\n")
        return "Contour points received"
    return "Dummy"


# This function sends the updated contour on performing HDP
@app.route("/getNewContour", methods=['GET'])
def sendUpdatedContour():
    ret = RefineContourMyoInitial.mainFunc()
    if request.method == 'GET':
        print('Sending new contour')
        response = {
            'username':'shashwat',
            'work':'Delta Nudge Tool'
        }
        response = jsonify(response)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    return "Hello"






