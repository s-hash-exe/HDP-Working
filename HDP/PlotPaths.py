"""
Created by  : Prof Arbind Kumar Gupta
Dated       : 27 Aug 2022 @12.00
Description : Use ppo.plotPaths to plot the path taken by dynamic programming at different levels of
              hierarchy. It shows how the path is chosen as DP progresses from one level to another
              level.
              ppo can also work with just the graph data generated during HDP run (without running HDP)
              if it has been saved earlier. To save it one time, call s.saveData().

Status      : Working for all tested conditions
Changes done: Bug fixes for proper display of DP paths at a given round and level
              To read saved data for plotting of paths with running DP
              Switching between test and image data is by just changing the flag "test" at line 11
To Do       :
Known Bugs  :
Changes     : Plotting of contours for both endo & epi in one image with image and dice value
Version     : 1.00
"""

import numpy as np
import matplotlib.pyplot as plt
import time as t
import pickle as pkl
import math

# plt.set_cmap('gray')
plt.rcParams['figure.figsize'] = [8, 6]

# -----------------------------------------------------------------------------------------
def main(img=None, loc=None):
    fPath, test = r'D:\Arbind\DSI\PythonProj\Myocardium\OneCardio\Api\Dicom_Server\Hierarchical DP', True
    file = fPath + ('\\testGraph.pkl' if test else '\\imgGraph.pkl')

    with open(file, 'rb') as f:
        graph  = pkl.load(f)
        aGrad  = pkl.load(f)
        BPCost = pkl.load(f)
        if not test:        img    = pkl.load(f)
        loc    = pkl.load(f)
        bs     = pkl.load(f)

    ppo = plotPaths(BPCost, graph, aGrad, bs)  # to plot paths in the graph
    if img is not None:     ppo.img = img
    if loc is not None:     ppo.loc = loc
    ppo.plotAllPathsRnd1()
    ppo.plotPaths()
    exit(0)


# -----------------------------------------------------------------------------------------

class plotPaths():
    def __init__(s, edgeIm, graph, grad, bs, DP_Ins=None):
        from copy import deepcopy as dc
        s.start, s.DP_Ins = True, DP_Ins
        s.graph, s.grad, s.edgeIm = dc(graph), dc(grad), dc(edgeIm)
        s.edgeIm = s.edgeIm[1:, ]
        s.size, s.batchSize = edgeIm.shape, bs
        s.img, s.loc = None, None
        s.sLevel2, s.eLevel2 = 1, 2

        s.eNdId2, s.sNdId2 = 0, 0     # eNdId2 selects end node at round2 & sNdId2 selects start node at round2
        s.details = False
        s.colorMap = ['.r', '.g', '.c', '.b', '.m']
        s.rounds = list(graph.keys())
        s.round1, s.round2 = s.rounds[0], s.rounds[1]
        s.updateState()

        #s.fig0 = plt.figure(0)  # to plot all line segments of round 1 output on the edge image
        s.fig = plt.figure(1)
        #  Register callback functions
        cid = s.fig.canvas.mpl_connect('button_press_event', s.onClick)
        cid = s.fig.canvas.mpl_connect('button_release_event', s.onRel)
        cid = s.fig.canvas.mpl_connect('key_press_event', s.onPress)
        #s.fig2 = plt.figure(2)  # to plot lines as dynamaicProgramming_NR() progresses from sLevel to eLvel

        s.yMin = s.yMax = -1
        cSum = s.edgeIm.sum(axis=0)  # to find non zero elements in any of the rows
        for i in range(len(cSum)):
            if cSum[i] > 0 and s.yMin < 0: s.yMin = i
            if cSum[i] > 0: s.yMax = i
        s.yMin, s.yMax = s.yMin-5, s.yMax+5     # the range of values for the y axis while plotting

    # -----------------------------------------------------------------------------------------
    def updateState(s):
        s.graph1, s.graph2 = s.graph[s.round1], s.graph[s.round2]
        s.grad1, s.grad2 = s.grad[s.round1], s.grad[s.round2]
        s.numLevels2, s.numLevels1 = max(s.graph2.keys()), max(s.graph1.keys())
        s.sNodesAtSLvl2, s.eNodesAtELvl2 = list(s.graph2[s.sLevel2].keys()), list(s.graph2[s.eLevel2].keys())
        if s.sNdId2 >= len(s.sNodesAtSLvl2):    s.sNdId2 = len(s.sNodesAtSLvl2)-1
        if s.eNdId2 >= len(s.eNodesAtELvl2):    s.eNdId2 = len(s.eNodesAtELvl2)-1
        s.sIdx, s.eIdx = (s.sLevel2-1) * s.batchSize ** (s.round2-1), (s.eLevel2-1) * s.batchSize ** (s.round2-1)

        #print('==>> Round1 = %d, round2 = %d, sNdId2 = %d, eNdId2 = %d'%(s.round1, s.round2, s.sNdId2, s.eNdId2), end=', ')
        #print('sLevel2 = %d, eLevel2 = %d and image display segment =%d, %d'%(s.sLevel2, s.eLevel2, s.sIdx, s.eIdx))

    # -----------------------------------------------------------------------------------------
    def handleKey(s, k):
        if   k == 'p' or k == 'P':  s.runDPForPlotting(1)
        elif k == 'a':              s.sLevel2, s.eLevel2 = 1, s.numLevels2
        elif k == 'd':              s.details = not s.details   # to print path details
        elif k == 'f' or k == 'F':  s.sIdx, s.eIdx = 1, s.size[1]   # reset the display area to the full image size

        elif k == '+':
            if s.round2 >= s.rounds[-1]: return False
            s.round1, s.round2 = s.round1+1, s.round2+1
            s.sLevel2, s.eLevel2, s.sNdId2, s.eNdId2 = 1, 2, 0, 0
        elif k == '-':
            if s.round1 <= s.rounds[0]: return False
            s.round1, s.round2 = s.round1-1, s.round2-1
            s.sLevel2, s.eLevel2, s.sNdId2, s.eNdId2 = 1, 2, 0, 0

        elif k == 'R':
            if s.eNdId2 < len(s.eNodesAtELvl2)-1:       s.eNdId2 += 1
            print('handleKey: ', s.sNdId2)
        elif k == 'r':
            if s.eNdId2 > 0:             s.eNdId2 -= 1
            print('handleKey: ', s.eNdId2)

        elif k == 'L':
            if s.sNdId2 < len(s.sNodesAtSLvl2)-1:       s.sNdId2 += 1
        elif k == 'l':
            if s.sNdId2 > 0:             s.sNdId2 -= 1


        elif k == '>':
            if s.eLevel2-s.sLevel2 == s.numLevels2: pass
            elif s.eLevel2 < s.numLevels2:   s.sLevel2, s.eLevel2 = s.sLevel2 + 1, s.eLevel2 + 1
        elif k == '<' and s.sLevel2 > 1:
            s.sLevel2, s.eLevel2 = s.sLevel2 - 1, s.eLevel2 - 1

        else:
            return False
        return True

    # -----------------------------------------------------------------------------------------
    def onPress(s, event):
        k = event.key
        if s.handleKey(k) is False: return
        s.updateState()
        s.plotPaths()

        return

    def onRel(s, event):
        x, y = event.xdata, event.ydata
        if x is None:  return
        print('End image index is ',s.eIdx)

    def onClick(s, event):
        ix, iy = event.xdata, event.ydata
        if ix is None:  return
        print('Start image index is ',s.sIdx)

    # -----------------------------------------------------------------------------------------
    def selDisplayArea(s, xval):
        # sIdx & eIdx select part of image area that has been selected for display. Line points outside of this area are clipped
        sidx = (s.sIdx-1) - xval[0] if xval[0] < s.sIdx else 0
        if xval[0] > s.eIdx:    eidx = 0
        else:
            eidx = len(xval) if xval[-1] <= s.eIdx else xval.searchsorted(s.eIdx)
        return sidx, eidx

    # -----------------------------------------------------------------------------------------
    def subPlot(s, round, figNum=1, lvl=None):
        try:
            if figNum == 1:
                if s.img is None:
                    plotNum = 122 if round == 2 else 121
                else:
                    plotNum = 132 if round == 2 else 131 if round == 1 else 133
            elif figNum == 2:
                subPlotNum = (lvl - s.sLevel1) + 1
                plotNum = 100 + 10* s.batchSize + subPlotNum
            else: plotNum = '111'
        except: print('In subplot: fig number and round is', figNum, round)
        return plotNum

    def plot(s, lvl, path, grad, round, style='-', figNum=1, lWidth=1):
        if len(path) == 0: return None
        if figNum >= 1:
            plt.subplot(s.subPlot(round, figNum, lvl))    # to select the subplot area where to plot
        rnd  = s.round2 if round == 2 else s.round1
        startLvl = s.batchSize ** (rnd-1) * (lvl-1)
        xval = np.array([startLvl + x for x in range(len(path[0][1]))])
        sidx, eidx = s.selDisplayArea(xval)
        if eidx <= sidx:    return None  # no part of line segment lies in display area

        for k in range(len(path)):
            if round == 1 or (s.eNdId2 == -1 or k <= s.eNdId2):  # plot all paths for round 1 and select path for round 2
                p = path[k][1]
                clr = s.colorMap[min(k, len(s.colorMap) - 1)] + style
                plt.plot(xval[sidx:eidx]-s.sIdx, p[sidx:eidx], clr, linewidth=lWidth)
                if s.details is True and grad is not None:
                    print('round=%d,  cost=%d, sGrad=%d, eGrad=%d  for path %s at level %d'%(rnd, path[k][0], \
                          grad[k][0], grad[k][1], str(p), lvl))
        return (sidx, eidx)

    # -----------------------------------------------------------------------------------------

    def Round1Paths(s, path2, lvl2):
        try:
            s.sLevel1 = (lvl2 - 1) * s.batchSize + 1
            s.eLevel1 = min(s.sLevel1+s.batchSize, s.numLevels1)
            for i, lvl in enumerate(range(s.sLevel1, s.eLevel1)):
                if path2 is not None:
                    eNodes = [path2[(i+1)*s.batchSize**(s.round1-1)-1] if lvl < s.numLevels1-1 else path2[-1]]
                else:
                    eNodes = list(s.graph1[lvl+1].keys())
                for eNode in eNodes:
                    paths = []
                    for sNode in s.graph1[lvl].keys():
                        try:
                            paths.append([s.graph1[lvl][sNode][eNode], s.grad1[lvl][sNode][eNode]])
                        except: pass
                    paths.sort(reverse=True)   # sort the [cost, path] and grad list in descending order based on cost
                    grad = [paths[k][1] for k in range(len(paths))]   # to separate path and gradients
                    paths = [paths[k][0] for k in range(len(paths))]
                    yield (lvl, paths, grad)
        except:
            print('In round1Path: ', path2)

    def Round2Paths(s, eNdId):
        for lvl in range(s.sLevel2, s.eLevel2):
            if s.eLevel2 - s.sLevel2 == 1:
                nList, path = s.graph2[lvl] if s.sNdId2 == -1 else [s.sNodesAtSLvl2[s.sNdId2]], []
            else:
                nList, path = s.graph2[lvl].keys(), s.graph2[lvl].values()
            for sNode in nList:
                path = []
                for k, eNd in enumerate(s.graph2[lvl][sNode]):
                    if k == eNdId:  eNdPath = s.graph2[lvl][sNode][eNd]
                    path.append([s.graph2[lvl][sNode][eNd], s.grad2[lvl][sNode][eNd]])
                path.sort(reverse=True)   # sort the [cost, path] and grad list in descending order based on cost
                grad = [path[k][1] for k in range(len(path))]   # to separate path and gradients
                path = [path[k][0] for k in range(len(path))]

                yield (lvl, np.array(path), grad, eNdPath)

    # -----------------------------------------------------------------------------------------
    def transformCtrBack(s, ctr, loc, lvl, round):
        h, fact = len(ctr), math.pi / 180
        inc, tCtr = 360 / 120, []
        stPos = (lvl - 1) * s.batchSize ** (round - 1) - 5  # 5 is the overlap at theta = 0 / 360 degree

        for t in range(h):
            th = (t+stPos) * inc
            x, y = loc[0] + ctr[t] * math.cos(th * fact), loc[1] + ctr[t] * math.sin(th * fact)
            tCtr.append((x, y))

        return np.array(tCtr)

    # --------------------------------------------------------------------------------------------------------------
    def plotAllPathsRnd1(s):
        plt.figure(0)   # select the figure to draw the complete paths for round 1
        s.eIdx = (s.numLevels1-1)*s.batchSize
        if s.img is not None:
            plt.imshow(s.edgeIm.transpose())
        for i, lvl in enumerate(range(1, s.numLevels1)):
            for sNode in s.graph1[lvl].keys():
                paths = []
                for eNode in s.graph1[lvl][sNode].keys():
                    paths.append(s.graph1[lvl][sNode][eNode])
                paths.sort(reverse=True)  # sort the [cost, path] and grad list in descending order based on cost
                s.plot(lvl, paths, None, 1, figNum=0)
        plt.ylim([s.yMin, s.yMax])
        plt.show(block=False)
        s.updateState(),   plt.figure(1)
        return

    # --------------------------------------------------------------------------------------------------------------
    def dataHook(s, bestSNodes):    # bestSNodes is a dictionary that stores the best starting node for each of the
        # nodes at a level betweeen sLevel1 and eLevel1. It is updated by dynamicPorgramming_NR fn in dynamicPrograaming.py
        s.bestSNodes = bestSNodes
        return

    def startDP(s, round):
        if s.DP_Ins is None: return
        #First initialize the DP_Ins object before running the dynamic programming
        dp, subPlotNum = s.DP_Ins, 1
        rnd = s.round1 if round == 1 else s.round2
        dp.DofGraph, dp.grad, dp.newGraph = s.graph[rnd], s.grad[rnd], {}
        dp.round, dp.numLevels, s.numLevels1 = rnd, max(dp.DofGraph.keys()), max(dp.DofGraph.keys())
        if dp.round == 3:       dp.updateGraphGrad()
        dp.updateNewGraphforEndNodes(end = 'LHS')  # to creates nodes at level=numLevels in the new graph
        dp.penalty = dp.penalty1 if dp.round == 1 else dp.penalty2 if dp.round == 2 else dp.penalty3
        dp.gradient = dp.gradient1 if dp.round == 1 else dp.gradient2
        return dp

    def runDPForPlotting(s, round):
        dp = s.startDP(round)
        if dp is None: return False
        sNode, eNode =s.sNodesAtSLvl2[s.sNdId2], s.eNodesAtELvl2[s.eNdId2]
        s.sLevel1 = (s.sLevel2 - 1) * s.batchSize + 1
        s.eLevel1 = min(s.sLevel1 + s.batchSize, s.numLevels1)
        rVal, sg, eg = dp.bestPath_WithGrad(sNode, s.sLevel1, eNode, s.eLevel1)  # get the best sg, eg value and then trace DP

        dp.dynamicPorgramming_NR(sNode, s.sLevel1, eNode, s.eLevel1, sg, eg, 0, [], s.dataHook)

        plt.figure(2), plt.figure(2).clf()
        print('In runDPForPlotting: ', rVal[1], s.bestSNodes)
        for lvl1 in range(s.sLevel1, s.eLevel1):
            pathToENodes = []
            for sNode, eNode in s.bestSNodes[lvl1]:
                pathToENodes.append(s.graph1[lvl1][sNode][eNode]) # the best path to eNode p[1][-1]@lvl1
                print('in runDP... ', lvl1, [sNode, eNode], pathToENodes, 'and display area is: ', s.sIdx, s.eIdx)

            sIdx, eIdx = s.plot(lvl1, pathToENodes, None, round, figNum=2)
            sIdx, eIdx = sIdx + (lvl1-s.sLevel1)*s.batchSize,  eIdx + (lvl1-s.sLevel1)*s.batchSize
            s.plot(lvl1, [[0, rVal[1][sIdx:eIdx]]], None, round, style='--', figNum=2, lWidth=2)
            plt.ylim([s.yMin, s.yMax])
            print('In runDP... ', lvl1, rVal[1][sIdx:eIdx])
        return True

    # --------------------------------------------------------------------------------------------------------------
    def plotCtrOnImage(s, p, lvl):
        if s.img is not None:
            os = 50     # select an area of image within a range of 40 from the centre (location)
            subImg = s.img[s.loc[1]-os:s.loc[1]+os, s.loc[0]-os:s.loc[0]+os]
            plt.subplot(133), plt.imshow(subImg)
            plt.plot([os], [os], 'ro')
            tCtr = s.transformCtrBack(p[1], s.loc, lvl, s.round2)
            plt.plot(tCtr[:,0]-s.loc[0]+os, tCtr[:,1]-s.loc[1]+os, 'r')
            # Since (s.loc[0]-os, s.loc[1]-os) is the new origin of the image being displayed

    def plotImages(s):
        plt.figure(1), s.fig.clf(1)
        if s.img is None:
            plt.subplot(121), plt.imshow(s.edgeIm[s.sIdx:s.eIdx+1, :].transpose())
            plt.subplot(122), plt.imshow(s.edgeIm[s.sIdx:s.eIdx+1, :].transpose())
        else:
            plt.subplot(131), plt.imshow(s.edgeIm[s.sIdx:s.eIdx + 1, :].transpose())
            plt.subplot(132), plt.imshow(s.edgeIm[s.sIdx:s.eIdx + 1, :].transpose())

    def setyAxisLimit(s):
        plt.subplot(s.subPlot(1, 1, None))
        plt.ylim([s.yMin, s.yMax])
        plt.subplot(s.subPlot(2, 1, None))
        plt.ylim([s.yMin, s.yMax])

    def raise_window(figNum=None):
        cfm = plt.get_current_fig_manager()
        cfm.window.activateWindow()

    def plotPaths(s):
        if s.eIdx - s.sIdx <= 1: return
        if s.start and 1:   # 0 to bypass code; 1 for simulation of key press to test the program
            s.handleKey('>'), s.handleKey('>')
            s.updateState()
            s.runDPForPlotting(1)
            s.start = False
        s.plotImages()

        for lvl2, path2, grad2, pathToENd in s.Round2Paths(s.eNdId2):
            s.plot(lvl2, path2, grad2, 2, style=':')   # Plot all the paths first as dotted line
            s.plot(lvl2, [pathToENd], None, 2)  # plot the selected path on top as a solid like
            s.plotCtrOnImage(pathToENd, lvl2)
            for lvl1, path1, grad1 in s.Round1Paths(pathToENd[1], lvl2):
                s.plot(lvl1, path1, grad1, 1)
        s.setyAxisLimit()

        title = 'sNd2 = '+str(s.sNodesAtSLvl2[s.sNdId2])+'(' + str(s.sNdId2) + '), eNd2 = '+str(s.eNodesAtELvl2[s.eNdId2])\
              +'('+str(s.eNdId2)+'), sLevel2 = '+str(s.sLevel2)+', eLevel2 = '+str(s.eLevel2) + ' at round '+str(s.round2)
        plt.suptitle(title),  plt.show(block=False)
        plt.figure(1), s.raise_window()


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

if __name__ == '__main__':
    from DynamicProgramming_V10 import DynProg as DPObj
    main()