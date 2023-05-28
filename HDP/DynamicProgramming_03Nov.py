"""
Description : It uses dynamic programming hierarchically from micro level to macro level, called
              round. At each round, it uses a different optimization function to find the best path. Thus, it
              is a hierarchical dynamic programming based optimization using multi-level optimization functions.
              In the first round, loss is computed for detecting straight line segments (min mean sq dist of pts from line).
              It gives weightage to dc/dp (first derivative of contour wrt points along the contour to be constant.
              In the second round, the loss is computed based on gradient difference of the two line segments at the junction.
              In third round, it looks for continuity of curvature, that is, the first and second derivative at the joining
              points of two segments on the contour being same.

Test tool   : Use ppo.plotPaths to plot the path taken by dynamic programming at different levels of
              hierarchy. It shows how the path is chosen as DP progresses from one level to another
              level. ppo can also work with just the graph data generated during HDP run (without
              running HDP) if it has been saved earlier. To save it one time, call s.saveData().

Created by  : Prof Arbind Kumar Gupta
Dated       : 27 Aug 2022 @12.20
Status      : Working properly for most conours but parameter fine tuning to be done still.
Changes done: Changes to make save results for plotting paths as stand alone appln
              Switching between test and image data is by just changing the flag "test" at line 11
To Do       : May be to reduce smoothing
Issues      : None
Version     : 1.00
"""
fPath, test = r'D:\Arbind\DSI\PythonProj\DynProg', False  # set test to true to run DP on test data

import numpy as np
import sys
import pickle as pkl

""
# from PlotPaths import plotPaths as PP
""
sys.path.append(r'.')  # Enter the path where SupportFunctions.py is available
import SupportFunctions as SF
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from math import atan2, sin, cos, pi

# plt.set_cmap('gray')  # to set default colour for matplotlib to gray
from copy import deepcopy as dc

MAX_COST = 100000
NODE_NBRS = 2

#====================  Optimization parameters at round 1, 2, 3 and 4 respectively  ====================================
DIST_PENALTY = 15  # penalty for a node not aligned with all the nodes in a batch at round 1. It is % cost per unit distance
GRAD_PENALTY = 15  # penalty added when grad from current node to next node is not same. % cost per radian diff)
JOINT_PENALTY = 15  # penalty for difference in the start grad at sNode@sLevel with end gradient at eNode@eLevel
CTR_COST = 20  # weightage given to distance of a point from ctr center
SHAPE_COST = 10

#-----------------------------------------------------------------------------------------------------------------------
class prepareGraph():
    def __init__(s, lvl_step_size=1, nd_step_size=1):
        s.DofGraph = {0: {0: {}}}
        s.edgeIm = None
        s.BPCost = np.zeros((1, 1))  # It stores the cost of a potential boundary point based on (a) the previous pixel
        # being blood pool and (b) the average blood pool values of preceding pixels (OVERLAP) being close to BPmean
        s.numLevels = 1  # Number of levels of nodes used for dynamic programming (will be set in initialize function
        s.numNodes = 0  # to be set based on cost array and ND_STEP_SIZE

        s.ND_STEP_SIZE = nd_step_size  # Step size to be used for nodes at a given level
        s.LVL_STEP_SIZE = lvl_step_size  # Graph for Dynamic programming will be created sampling nodes in steps of LVL_STEP_SIZE
        s.nd_nbrs = NODE_NBRS + s.ND_STEP_SIZE // 2
        s.ctrType = 0  # 0 means endocardium; 1 means epicardium
        s.img, s.rad, s.loc = None, 10, None
        s.batch_size = 5  # It is reset in setupGraph for testdata

#-----------------------------------------------------------------------------------------------------------------------
    def interpolatePath(s, path):  # It interpolates for intermediate points when LVL_STEP_SIZE is greater than 1.
        ipath = []
        for i, nd in enumerate(path):
            ipath.append(nd)
            for lvl in range(1, s.LVL_STEP_SIZE):
                interNd = ipath[-1] + (nd - ipath[-1]) * lvl / s.LVL_STEP_SIZE + 0.5
                ipath.append(interNd)

        return np.array(ipath, int)

#-----------------------------------------------------------------------------------------------------------------------
    st, end = 0, 0

    def closeRadialGap(s, path, adjCost, r, spd):
        # When two points on the contour are far apart, they will not have overlap (with spread also)
        # This ensures that there will be overlap and there will be a horizontal path also
        global st, end
        if r == 0:
            st, end = path[r] - spd, path[r] + spd + 1
        else:
            st, end = min(st + spd, path[r] - spd), max(path[r] + spd + 1, end - spd)
        adjCost[r, st:end] = s.BPCost[r, st:end]
        return

#-----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------Old Code----------------------------------------------------------------------
    # def adjustBPCost(s, ctr, spd=2):
    #     maxEdge = s.edgeIm.max()
    #     s.BPCost = s.edgeIm + maxEdge // 2  # so that small areas where the edge mag is zero, does not stop
    #     # dyn programming. Also the relative weights of edges should not differ by a large factor
    #
    #     adjCost = np.zeros_like(s.BPCost)
    #     path = [int(ctr[r] + 0.5) for r in range(len(ctr))]
    #     h, w = s.BPCost.shape
    #     for r in range(h):
    #         # adjCost[r, path[r] - spd:path[r] + spd+1] = s.BPCost[r, path[r] - spd:path[r] + spd+1]
    #         s.closeRadialGap(path, adjCost, r, spd)
    #
    #     mn = mx = -1
    #     cSum = adjCost.sum(axis=0)  # to find non zero elements in any of the rows
    #     for i in range(len(cSum)):
    #         if cSum[i] > 0 and mn < 0: mn = i
    #         if cSum[i] > 0: mx = i
    #
    #     nonZero = adjCost[:, mn:mx + 1]
    #     # nonZero[nonZero == 0] = maxEdge//4  # all the zeros in the col range mn:mx+1 are set to a min value to allow search thro them
    #     adjCost[:, mn:mx + 1] = nonZero
    #     s.BPCost = adjCost
    #
    #     return
# ----------------------------------------Old Code----------------------------------------------------------------------

#----------------------------------------Modified Code------------------------------------------------------------------
    def adjustBPCost(s, ctr, spd=2, desired_path=None) :
        maxEdge = s.edgeIm.max()
        s.BPCost = s.edgeIm + maxEdge // 2

        adjCost = np.zeros_like(s.BPCost)
        path = [int(ctr[r] + 0.5) for r in range(len(ctr))]
        h, w = s.BPCost.shape
        for r in range(h) :
            # if(r>=len(path)): break
            s.closeRadialGap(path, adjCost, r, spd)
            #-------------------------------------Modification----------------------------------------------------------
            if desired_path :
                desired_r = desired_path[r]
                adjCost[r, :] *= (1 - desired_path[r])
                adjCost[r, desired_r - spd :desired_r + spd + 1] = s.BPCost[r, desired_r - spd :desired_r + spd + 1]
            #--------------------------------------Modification---------------------------------------------------------

        mn = mx = -1
        cSum = adjCost.sum(axis=0)
        for i in range(len(cSum)) :
            if cSum[i] > 0 and mn < 0 : mn = i
            if cSum[i] > 0 : mx = i

        nonZero = adjCost[:, mn :mx + 1]
        adjCost[:, mn :mx + 1] = nonZero
        s.BPCost = adjCost

        return

# ----------------------------------------Modified Code-----------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

    def createGraph(s, nodes):
        endNodeCost = lambda lvl, en, lvls: s.BPCost[lvl, en] if lvl < lvls else 1

        for level in range(-1, s.numLevels):
            startNdDict = {}  # Maintains list of end nodes with path for each start node (key)
            eLevel = level + 1
            for st_nd in range(s.numNodes):
                sNode = nodes[level, st_nd] if level >= 0 else -1
                if level >= 0 and s.BPCost[level, sNode] == 0:      continue  # there is no path from this start node
                connectedToDict = {}  # Maintains [cost, path] for each end node (key)
                for end_nd in range(s.numNodes):
                    eNode = nodes[eLevel, end_nd] if eLevel < s.numLevels else -1
                    enCost = endNodeCost(eLevel, eNode, s.numLevels)
                    # a small cost so that it is not ignored at the next step. Nodes at level 0 are dummy with 0 cost
                    if enCost == 0:                 continue
                    d = end_nd - st_nd if sNode != -1 else 0
                    if abs(d) < s.nd_nbrs:
                        connectedToDict[eNode] = [enCost, [eNode]] if eNode != -1 else [0, []]
                        if sNode == -1:
                            connectedToDict[eNode] = [0, [0]]
                            # A dummy level at the LHS was added in setup graph and its cost is set to 0
                            # this helps in creating graph with correct slope for sLevel == 1
                        if eNode == -1: break  # eNode is a virtual node and there are no other nodes at eLevel

                if len(connectedToDict) > 0:        startNdDict[sNode] = connectedToDict
                if sNode == -1: break  # eNode is a virtual node and there are no other nodes at eLevel
            if len(startNdDict):                    s.DofGraph[eLevel] = startNdDict
        return

#-----------------------------------------------------------------------------------------------------------------------
    def setupGraph(s, strip, BPmean, MCmean=40, tCtr=None, rad=15):
        if not test:
            s.tCtr = tCtr
            s.rad, s.strip = rad, strip
            edgeIm = SF.getEdgeMap(strip)
            s.edgeIm = edgeIm.copy()[::s.LVL_STEP_SIZE]
            h, w = s.edgeIm.shape
            s.numLevels = h

            if tCtr is not None:
                s.adjustBPCost(s.tCtr)
            # it adjusts the cost so that search is limited to a narrow width around the contour given as input
            s.BPCost = s.BPCost[::s.LVL_STEP_SIZE]
            s.meanEdgeMag = np.mean(s.BPCost[s.BPCost > 0])

            s.numNodes = (w + s.ND_STEP_SIZE - 1) // s.ND_STEP_SIZE
            nodes = np.zeros((s.numLevels, s.numNodes), int)

            for lvl in range(s.numLevels):
                for nd, i in enumerate(range(0, w, s.ND_STEP_SIZE)):
                    j = min(i + s.ND_STEP_SIZE, w)
                    nodes[lvl, nd] = min(i + s.ND_STEP_SIZE // 2, w - 1)
        else:
            s.rad, s.meanEdgeMag, s.batch_size = 0, 7, 3
            s.img = None
            s.BPCost = np.array([[0, 0, 0, 7, 0, 6, 0],
                                 [0, 0, 0, 0, 9, 8, 0],
                                 [0, 0, 0, 0, 0, 7, 0],

                                 [0, 0, 0, 0, 7, 6, 0],
                                 [0, 0, 0, 6, 0, 5, 0],
                                 [0, 0, 0, 7, 0, 6, 0],

                                 [0, 0, 0, 0, 9, 8, 0],
                                 [0, 0, 0, 0, 0, 7, 0],
                                 [0, 0, 0, 0, 7, 6, 0],

                                 [0, 0, 0, 6, 0, 5, 0]], int)

        s.BPCost = np.vstack([s.BPCost[0], s.BPCost])  # copied the nodes at the first level to ensure that the gradient
        # calculations are done correctly. Otherwise, the prev node is -1@-1 and grad can't be calculated
        nodes = np.array([[c for c in range(len(s.BPCost[0]))] for j in range(len(s.BPCost))])  # it is used as node ID
        s.numLevels, s.numNodes = nodes.shape
        s.createGraph(nodes)

#-----------------------------------------------------------------------------------------------------------------------
    def shapePenalty(s, p):
        return 0
        std = np.std(p)
        sPenalty = std * SHAPE_COST / 100
        return sPenalty

#-----------------------------------------------------------------------------------------------------------------------
    def selectBestPath(s):
        maxCost, path = 0, []
        sNodeDict = s.DofGraph[1]
        for sNode in sNodeDict:
            for eNode in sNodeDict[sNode]:
                if abs(eNode - sNode) >= NODE_NBRS:              continue  # The start and end nodes should be same
                c, p = sNodeDict[sNode][eNode]
                c = c - s.shapePenalty(p)
                if c > maxCost:
                    maxCost, path = c, p
        print('Best Cost is: %3.1f  and path is: %s' % (maxCost, path))
        return path

#-----------------------------------------------------------------------------------------------------------------------
    def FourierSmoothing(s, ctr):
        ft = np.fft.fft(ctr)
        filteredFT = ft.copy()
        filteredFT[6:] = 0
        fCtr = np.fft.ifft(filteredFT)
        fCtr = np.abs(fCtr)
        SF.plotCurves([ctr], color='b-')
        SF.plotCurves([fCtr], color='r-', show=True, block=True)

        return fCtr

#-----------------------------------------------------------------------------------------------------------------------
    def updateCtrCost(s, bs):
        ctr = s.tCtr
        for lvl in range(1, max(s.DofGraph.keys())):
            sNodeDict = s.DofGraph[lvl]
            for sNode in sNodeDict:
                thRange = [i for i in range((lvl - 1) * bs, min(lvl * bs, s.numLevels-1))]
                rList = ctr[thRange]
                for eNode in sNodeDict[sNode]:
                    c, p = sNodeDict[sNode][eNode]
                    diff = [i * i for i in rList - p]
                    msd = np.mean(diff)
                    sNodeDict[sNode][eNode] = c - msd * CTR_COST / bs, p
        return

#-----------------------------------------------------------------------------------------------------------------------
    def saveData(s, graph, aGrad):  # to save data for running the plotting of paths for visual interpretation
        file = fPath + ('\\testGraph.pkl' if test else '\\imgGraph.pkl')
        with open(file, 'wb') as f:
            pkl.dump(graph, f)
            pkl.dump(aGrad, f)
            pkl.dump(s.BPCost, f)
            if not test:    pkl.dump(s.img, f)
            pkl.dump(s.loc, f)
            pkl.dump(s.batch_size, f)

        print('----------- Saved data and terminating the program -------------')
        exit(0)

    def runDynPorg(s):
        DP_Ins = DynProg()
        DP_Ins.sign = 0  # 1 if s.ctrType == 0 else -1 <<<<<<<<<<<<<<<<<<=============================
        """sign = -1 means it adds a penalty for higher r value -> makes the contour closer to blood pool
           sign= 1 means it adds to the path cost and makes the contour further from centroid of myocardium."""
        DP_Ins.rad = s.rad + 6
        DP_Ins.initialize(s.BPCost, nd_nbrs=s.nd_nbrs, BATCH_SIZE=s.batch_size, mean=s.meanEdgeMag)
        graph, aGrad = {}, {}  # to store graphs and grdients at different rounds (to plot and study any missing details)

        while (s.numLevels > 2):  # As there is a dummy node -1 at level -1
            """In the first round, loss is computed for detecting straight line segments (min mean sq dist of pts from line).
            It gives weightage to dc/dp (first derivative of contour wrt points along the contour to be constant.
            In the second round, the loss is computed based on gradient difference of the two line segments at the junction.
            In third round, it looks for continuity of curvature, that is, the first and second derivative at the joining
            points of two segments on the contour being same.   """
            s.DofGraph, grad = DP_Ins.processGraphinBatch(s.DofGraph)
            if DP_Ins.round == 1 and not test: s.updateCtrCost(s.batch_size)
            DP_Ins.numLevels = max(s.DofGraph.keys())
            s.numLevels = DP_Ins.numLevels
            DP_Ins.round += 1
            graph[DP_Ins.round], aGrad[DP_Ins.round] = dc(s.DofGraph), dc(grad)
        path = np.array(s.selectBestPath(), int)

        """ppo = PP(s.BPCost, graph, aGrad, s.batch_size, DP_Ins)     
        To plot paths in the graph for a visual representation of DP on the path taken.
        See header of PlotPaths for more details.
        ppo.img, ppo.loc = s.img, s.loc
        ppo.plotPaths()     #s.saveData(graph, aGrad) """

        if s.LVL_STEP_SIZE > 1:    path = s.interpolatePath(path)
        fPath_SG = savgol_filter(path, 15, 2)
        # SF.showImage(s.edgeIm.transpose(), numWin=211), SF.plotCurves([s.tCtr]), SF.plotCurves([fPath_SG])
        # SF.showImage(s.strip.transpose(), numWin=212), SF.plotCurves([s.tCtr], show=True, block='Wait')
        return fPath_SG

# ======================================================================================================================
class DynProg:
    def __init__(s):
        s.BPCost = None  # It stores the cost of a potential boundary point based on (a) the previous pixel being blood pool and (b) the
        # average blood pool values of preceding pixels (OVERLAP) being close to BPmean
        s.WEIGHT = 20  # weight for edge cost in percent when they are not at the same horizontal level
        s.numNodes = 0  # Number of nodes at each level considered for dynamic programming (set in initialization fn)
        s.NUM_ENODES = 0
        s.numLevels = 1  # Number of levels of nodes used for dynamic programming (will be set in initialize function
        s.BATCH_SIZE = 2  # Number of levels in one batch run for dynamic programming (set in initialization fn)
        s.maxRowCost = []  # stores the maximum cost for each row (level) as a 1D array of length numLvls
        s.DofGraph, s.grad = {}, {}
        s.rad, s.sign = 10, -1
        s.gradient, s.penalty = s.gradient1, s.penalty1

#-----------------------------------------------------------------------------------------------------------------------

    def initialize(s, BPCost, nd_nbrs=2, BATCH_SIZE=5, mean=6):
        s.BPCost = BPCost
        s.numLevels, s.numNodes = s.BPCost.shape
        s.BATCH_SIZE = BATCH_SIZE
        s.nd_nbrs = nd_nbrs  # Separation between two nodes at adjacent levels that is considered as a valid transition
        s.round = 1
        s.meanEdgeMag = mean

        return

#-----------------------------------------------------------------------------------------------------------------------
    def updateGraphGrad(s):
        # It updates the gradient by taking the mean of 3 points, instead of just one point, as done earlier (due to less num of pts)
        # It also computes the second order derivatives at the start of each line segment at round 3
        if s.round != 3:        return

        for lvl in range(0, max(s.DofGraph.keys())):
            for sNode in s.DofGraph[lvl]:
                for eNode in s.DofGraph[lvl][sNode]:
                    _, p = s.DofGraph[lvl][sNode][eNode]
                    if len(p) < 10:
                        s.grad[lvl][sNode][eNode] = s.grad[lvl][sNode][eNode] + [0, 0]
                        # 0, 0 is appended as gradOfGrad on the LHS and RHS are both 0 (can't be calculated)
                        continue
                    # Calculate gradient on the RHS of the line segment
                    st, mid, end = np.mean(p[-7:-4]), np.mean(p[-5:-2]), np.mean(p[-3:])
                    eGrad = int(atan2(end - st, 5) * 180 / pi)
                    s2, e2 = int(atan2(mid - st, 3) * 180 / pi), int(atan2(end - mid, 3) * 180 / pi)
                    eGradOfGrad = (e2 - s2) / 3

                    # Calculate gradient on the LHS of the line segment
                    st, mid, end = np.mean(p[0:3]), np.mean(p[2:5]), np.mean(p[4:7])
                    sGrad = int(atan2(end - st, 5) * 180 / pi)
                    s2, e2 = int(atan2(mid - st, 3) * 180 / pi), int(atan2(end - mid, 3) * 180 / pi)
                    sGradOfGrad = (e2 - s2) / 3
                    s.grad[lvl][sNode][eNode] = [sGrad, eGrad, sGradOfGrad, eGradOfGrad]
        return

#-----------------------------------------------------------------------------------------------------------------------
    def gradient1(s, sNode, sLevel, eNode=None, eLevel=None, type='RHS'):
        grad = s.grad[sLevel][sNode][eNode]
        grad = grad[0] if type == 'LHS' else grad
        return grad

    def gradient2(s, sNode, sLevel, eNode=None, eLevel=None, type='RHS'):
        grad = None
        if type == 'RHS':
            connTo = s.grad[eLevel - 1]  # All the  nodes at prev level that are connected to eNodes@eLevel
            grad = [connTo[nd][eNode][1] for nd in connTo if eNode in connTo[nd]]
            if len(grad) == 0: grad = None
        else:
            connTo = s.DofGraph[sLevel - 1]  # All the  nodes at previous level that are connected to sNodes@slevel
            c, grad = -10000, None
            for nd in connTo:  # find the node that has the highest cost to sNode
                if sNode not in connTo[nd]: continue
                if connTo[nd][sNode][0] > c: c, grad = connTo[nd][sNode][0], s.grad[sLevel - 1][nd][sNode][1]
        return grad

#-----------------------------------------------------------------------------------------------------------------------
    def secOrderGrad(s, sNode, sLevel,
                     path):  # find the end gradient of line segment from pNode@sLevel-1 to sNode@sLevel
        if s.round < 3:    return 0
        pNode = path[-1]  # The last node on the best path at the previous level sLevel-1
        gradofGrad = None
        connTo = s.DofGraph[sLevel - 1]  # All the  nodes at previous level that are connected to sNodes@slevel
        c, grad = -10000, None
        for nd in connTo:  # find the node that has the highest cost to sNode
            if sNode not in connTo[nd]: continue
            if connTo[nd][sNode][0] > c:
                c, grad, gradofGrad = connTo[nd][sNode][0], s.grad[sLevel - 1][nd][sNode][1], \
                    s.grad[sLevel - 1][nd][sNode][3]
        return grad

#-----------------------------------------------------------------------------------------------------------------------
    def penalty1(s, sGrad, eGrad, cNode, cLevel, eNode, eLevel, nNode, pLineSeg=None):
        if s.round != 1:    return 0
        sLevel, nLevel = max(eLevel - s.BATCH_SIZE, 0), cLevel + 1
        thRad = (eGrad + 90) * pi / 180  # slope of normal to the line, eGrad being slope of the line
        r = eLevel * cos(thRad) + eNode * sin(thRad)
        d = nLevel * cos(thRad) + nNode * sin(thRad) - r
        return abs(d) * DIST_PENALTY * s.meanEdgeMag / 100

    def penalty2(s, sGrad, eGrad, cNode, cLevel, eNode, eLevel, nNode, pLineSeg=None):
        if s.round != 2:    return 0
        pEGrad = s.gradient2(cNode, cLevel, type='LHS')  # gradient of the best path joining to cNode@cLevel
        cSGrad = s.grad[cLevel][cNode][nNode][0]  # the first gradient is start gradient

        gDiff = ((cSGrad - pEGrad) / 5) ** 2  # The cost increases as square of  the gradient difference
        gPenalty = gDiff * GRAD_PENALTY * s.meanEdgeMag / 100
        # negativeGradPenalty = -(cGrad)*SHAPE_COST if cGrad < 0 else 0     # to prevent a concave shape of contour
        # gPenalty = (gPenalty + negativeGradPenalty) * cost
        return gPenalty

    def penalty3(s, sGrad, eGrad, cNode, cLevel, eNode, eLevel, nNode, pNodeD=None):
        if s.round != 3:    return 0
        pEGrad = s.gradient2(cNode, cLevel, type='LHS')  # gradient of the best path joining to cNode@cLevel
        cSGrad = s.grad[cLevel][cNode][nNode][0]  # the start gradient at the current level
        gPenalty = abs(pEGrad - cSGrad) * JOINT_PENALTY * s.meanEdgeMag / 100
        if cNode in pNodeD:  # find the start and end 2nd order gradient betweem pNode@cLevel-1, cNode@cLevel, nNode@cLevel+1
            pNode = pNodeD[cNode]
            p2ndOrdGrad, c2ndOrdGrad = s.grad[cLevel - 1][pNode][cNode][3], s.grad[cLevel][cNode][nNode][2]
            gPenalty += (abs(p2ndOrdGrad - c2ndOrdGrad)) * JOINT_PENALTY * s.meanEdgeMag / 100

        return gPenalty

#-----------------------------------------------------------------------------------------------------------------------
    def dynamicPorgramming_NR(s, sNode, sLevel, eNode, eLevel, sGrad=0, eGrad=0, pcost=0, path=[], dataHook=None):
        # dataHook is a function provided by PlotPaths for getting the internmediate paths for showing progress
        pathExists = lambda d, k: False if k not in d.keys() else True
        bPNdD, pLineSeg, DofBestSNodes = {sNode: [0, []]}, {}, {}
        for lvl in range(sLevel, eLevel):
            nBCNdD, cLineSeg, DofBestSNodes[lvl] = {}, {}, []
            cNdD = s.DofGraph[lvl + 1].keys() if lvl + 1 < eLevel else [eNode]
            # gives the nodes at the current level lvl since graph contains path from nodes@lvl to nodes@lvl+1
            for eNd in cNdD:  # for each node eNd at the next level, find the best start node bSNd
                maxCost, maxPath = -10000, None
                for bSNd in bPNdD:  # For each of the best node at previous level connected to at the current level
                    c, p = bPNdD[bSNd]  # it stores the cost and best path upto node bSNd@lvl-1
                    try:
                        if not pathExists(s.DofGraph[lvl][bSNd], eNd):                        continue
                    except:
                        continue
                    path = p + s.DofGraph[lvl][bSNd][eNd][1]
                    cost = c + s.DofGraph[lvl][bSNd][eNd][0] - \
                           s.penalty(sGrad, eGrad, bSNd, lvl, eNode, eLevel, eNd, pLineSeg)
                    cost = int(cost * 10) / 10  # to convert it into a float with precision 1
                    if cost > maxCost:      maxCost, maxPath, cLineSeg[eNd] = cost, path, bSNd

                if maxCost != -10000:   nBCNdD[eNd], DofBestSNodes[lvl] = [maxCost, maxPath], DofBestSNodes[lvl] + [
                    [cLineSeg[eNd], eNd]]
            if len(nBCNdD) > 0:         bPNdD, pLineSeg = nBCNdD, cLineSeg

        if dataHook is not None:
            dataHook(bestSNodes=DofBestSNodes)
        return bPNdD[eNode] if eNode in bPNdD.keys() else None

#-----------------------------------------------------------------------------------------------------------------------
    def bestPath_WithGrad(s, sNode, sLevel, eNode, eLevel):
        """sGrad is the slope from best node@sLevel-2 to sNode@sLevel-1  and eGrad is the list of slopes from
        eNode@eLevel-1 to all connected nodes@eLevel. Because DofGraph contains connections from a node at previous
        level to a node at the current level.
        Now the best path from sNode to eNode is the one that matches most closely with the starting slope"""
        sGrad = s.gradient(sNode, sLevel, eNode, eLevel, type='LHS')  # best gradient to the sNode@sLevel
        eGrad = s.gradient(sNode, sLevel, eNode, eLevel)
        # List of gradients from the eNode@eLevel-1 to all the nodes at the eLevel
        if sGrad is None or eGrad is None: return None, None, None  # there are no gradient (connection) to sNone / to eNode

        maxCost, retVal, bEGrad, meg = -10000, None, 0, 0
        for eg in eGrad:
            rVal = s.dynamicPorgramming_NR(sNode, sLevel, eNode, eLevel, sGrad, eg, 0, [])
            if rVal is None:                continue
            if rVal[0] > maxCost:
                maxCost, retVal, bEGrad = rVal[0], rVal, eg

        return retVal, sGrad, bEGrad

#-----------------------------------------------------------------------------------------------------------------------
    def updateGradForNewGraph(s, sNode, sLevel, eNode, sgrad, egrad, append=False):
        try:
            if append:
                newsLvl = max(s.newGraph.keys()) + 1
            else:
                newsLvl = (sLevel + s.BATCH_SIZE - 1) // s.BATCH_SIZE

            sd = s.newGrad[newsLvl] if newsLvl in s.newGrad.keys() else {}
            d1 = {} if sNode not in sd.keys() else sd[sNode]
            d1[eNode] = [sgrad, egrad]

            sd[sNode] = d1
            s.newGrad[newsLvl] = sd
        except:
            pass
        return None

#-----------------------------------------------------------------------------------------------------------------------
    def setInitialGradients(s):
        if s.round > 1:
            newGrad = s.newGrad
        else:
            newGrad = dict()
            for sLevel in range(1, s.numLevels, s.BATCH_SIZE):
                eLevel = min(sLevel + s.BATCH_SIZE, s.numLevels)
                stNdDict, newStNdDict = s.DofGraph[sLevel], dict()
                # DofGraph contains cost from nodes at level sLevel-1 to a node at level sLevel

                for sNode in stNdDict.keys():
                    endNdDict, newEndNdDict = s.DofGraph[eLevel], {}
                    for eNode in endNdDict:
                        grad = int(atan2(eNode - sNode, eLevel - sLevel) * 180 / pi) if sNode != -1 else 0
                        newEndNdDict[eNode] = [grad]  # start and end gradient for sNode@sLevel to eNode@eLevel
                    newStNdDict[sNode] = newEndNdDict

                newGrad[sLevel] = newStNdDict

            newStNdDict = {-1: {}}
            for eNode in s.DofGraph[0][-1]:
                newStNdDict[-1][eNode] = 0
            newGrad[0] = newStNdDict

        s.grad, s.newGrad = newGrad, {}
        return

#-----------------------------------------------------------------------------------------------------------------------
    def printGraph(s, lvl):
        for sLevel in range(lvl, lvl + s.BATCH_SIZE):
            print('Level=', sLevel)
            stNdDict = s.DofGraph[sLevel]
            for sNode in stNdDict.keys():
                print('SNode=', sNode, end=' [')
                endNdDict = stNdDict[sNode]
                for eNode in endNdDict:
                    print(eNode, endNdDict[eNode], end=' ')
                print(']')

#-----------------------------------------------------------------------------------------------------------------------
    def updateNewGraphforEndNodes(s, end='LHS'):
        # Update the best path at sLevel = -1 and eLevel = s.numLevels
        # end='RHS' -> use the sNode as -1 for sLevel = -1 and eLevel = 0 and
        # end='LHS' -> use the eNode as -1 for sLevel = s.numLevels-1 and eLevel = s.numLevels
        newNdDict = {}
        if end == 'LHS' or end == 'BOTH':
            newNdDict[-1] = {}
            for sNode in s.DofGraph[0][-1]:
                newNdDict[-1][sNode] = [0, []]
                s.updateGradForNewGraph(-1, 0, sNode, 0, 0)  # gradient[0] used only when s.round==1
            s.newGraph[0] = newNdDict

        else:  # end = 'RHS' or 'BOTH'
            for sNode in s.DofGraph[s.numLevels]:
                newNdDict[sNode] = {-1: {0: []}}
                s.updateGradForNewGraph(sNode, s.numLevels + 1, -1, 0, 0,
                                        append=True)  # gradient[0] used only when s.round==1
            s.newGraph[len(s.newGraph)] = newNdDict

#-----------------------------------------------------------------------------------------------------------------------
    def processGraphinBatch(s, graph):
        s.DofGraph, s.newGraph = graph, {}
        s.setInitialGradients()
        if s.round == 3:       s.updateGraphGrad()
        s.updateNewGraphforEndNodes(end='LHS')  # to creates nodes at level=numLevels in the new graph
        s.penalty = s.penalty1 if s.round == 1 else s.penalty2 if s.round == 2 else s.penalty3
        s.gradient = s.gradient1 if s.round == 1 else s.gradient2

        for sLevel in range(1, s.numLevels, s.BATCH_SIZE):
            eLevel = min(sLevel + s.BATCH_SIZE, s.numLevels)
            stNdList, newStNdDict = s.DofGraph[sLevel].keys(), {}
            # DofGraph contains cost from nodes at level sLevel-1 to a node at level sLevel

            for sNode in stNdList:
                endNdList, newEndNdDict = s.DofGraph[eLevel].keys(), {}
                for eNode in endNdList:
                    rVal, sg, eg = s.bestPath_WithGrad(sNode, sLevel, eNode, eLevel)
                    if rVal == None:                continue
                    newEndNdDict[eNode] = rVal
                    s.updateGradForNewGraph(sNode, sLevel, eNode, sg, eg)

                if len(newEndNdDict) > 0:   newStNdDict[sNode] = newEndNdDict
            if len(newStNdDict) > 0: s.newGraph[
                (sLevel + s.BATCH_SIZE - 1) // s.BATCH_SIZE] = newStNdDict  # Only if there are connected eNodes

        s.updateNewGraphforEndNodes(end='RHS')  # to creates nodes at level=numLevels in the new graph
        return (s.newGraph, s.newGrad.copy())
#-----------------------------------------------------------------------------------------------------------------------
