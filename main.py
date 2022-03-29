# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from camera_L import cameradataL
from camera_R import cameradataR

cameraR = cv2.VideoCapture(1)
cameraL = cv2.VideoCapture(2)

pathR = './imgR/'
pathL = './imgL/'

fnameR = 'imgR_'
fnameL = 'imgL_'

ft = time.time()

def imgin():
    i = 100

    while True:
        ret, imgR = cameraR.read()
        ret, imgL = cameraL.read()



        cv2.imshow('imgR', imgR)
        cv2.imshow('imgL', imgL)


        if cv2.waitKey(1) & 0xFF == ord('s'):
            i += 1
            cv2.imwrite(pathR + fnameR + str(i) + '.jpg', imgR)
            cv2.imwrite(pathL + fnameL + str(i) + '.jpg', imgL)
            print('save:' + pathR + fnameR + str(i) + '.jpg' + pathL + fnameL + str(i) + '.jpg')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cameraL.release()
    cameraR.release()
    cv2.destroyWindow("imgR")
    cv2.destroyWindow("imgL")

def undistortion(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    if roi != (0, 0, 0, 0):
        dst = dst[y:y + h, x:x + w]

    return dst

def disparity_SGBM(left_image, right_image, down_scale=False):
    

    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 5
    blockSize = 3

    param = {'minDisparity': 64,
             'numDisparities': 128,
             'blockSize': blockSize,
             'P1': 64 * img_channels * blockSize ** 2,
             'P2': 127 * img_channels * blockSize ** 2,
             'disp12MaxDiff': -1,
             'preFilterCap': 63,
             'uniquenessRatio': 3,
             'speckleWindowSize': 200,
             'speckleRange': 2,
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }

    sgbm = cv2.StereoSGBM_create(**param)

    size = (left_image.shape[1], left_image.shape[0])

    if down_scale == False:
        disparity_left = sgbm.compute(left_image, right_image)
        disparity_right = sgbm.compute(right_image, left_image)
        disparity_left = cv2.normalize(disparity_left, disparity_left, alpha=255,
                                      beta=5, norm_type=cv2.NORM_MINMAX)
        disparity_left = np.uint8(disparity_left)
        disparity_right = cv2.normalize(disparity_right, disparity_right, alpha=255,
                                       beta=5, norm_type=cv2.NORM_MINMAX)
        disparity_right = np.uint8(disparity_right)


    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = size[0] / left_image_down.shape[1]
        disparity_left_half = sgbm.compute(left_image_down, right_image_down)
        disparity_right_half = sgbm.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left *= factor
        disparity_right *= factor

    return disparity_left, disparity_right

def stereoCalibrate_camera():
    frameR = cv2.imread('./imgR/imgR_1.jpg')
    frameRG = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00000001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00000001)

    npcmpR = np.load('cmpR.npz')
    npcmpL = np.load('cmpL.npz')


    objpointsL = npcmpL['objpointsL']
    objpointsR = npcmpR['objpointsR']
    imgpointsR = npcmpR['imgpointsR']
    imgpointsL = npcmpL['imgpointsL']
    mtxL = npcmpL['mtxL']
    mtxR = npcmpR['mtxR']
    distL = npcmpL['distL']
    distR = npcmpR['distR']

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    flags |= cv2.CALIB_ZERO_TANGENT_DIST

    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpointsL, imgpointsL, imgpointsR, mtxL, distL, mtxR,
                                                               distR, frameRG.shape[::-1], criteria_stereo, flags)
    # print(retS, MLS, dLS, MRS, dRS, R, T, E, F)

    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, frameRG.shape[::-1], R, T, flags=flags,
                                                      alpha=0, newImageSize=(0, 0))

    # print('RL, RR, PL, PR, Q, roiL, roiR', RL, RR, PL, PR, Q, roiL, roiR)

    Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                  frameRG.shape[::-1], cv2.CV_16SC2)

    Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                   frameRG.shape[::-1], cv2.CV_16SC2)
    return Left_Stereo_Map, Right_Stereo_Map


def rectified(capL, cpaR):

    #frameR = cv2.imread('./imgR/imgR_101.jpg', 0)
    #frameL = cv2.imread('./imgL/imgL_101.jpg', 0)
    #frameR = cpaR.read()
    #frameL = capL.read()

    Left_rectified = cv2.remap(capL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LINEAR)
    im_L = Image.fromarray(Left_rectified)  # numpy to image

    Right_rectified = cv2.remap(cpaR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LINEAR)
    im_R = Image.fromarray(Right_rectified)  # numpy to image
    Left_rectified, Right_rectified = disparity_SGBM(capL, cpaR)
    return Left_rectified, Right_rectified

def skinmask(roi):
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 轉換至YCrCb空間
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (3, 3), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu處理
    res = cv2.bitwise_and(roi, roi, mask=skin)
    kernel = np.ones((3, 3), np.uint8)  # 設置卷積核
    erosion = cv2.erode(res, kernel)  # 腐蝕操作
    res = cv2.dilate(erosion, kernel)

    return res



Left_Stereo_Map, Right_Stereo_Map = stereoCalibrate_camera()

if __name__ == '__main__':

    while True:

        ret, imgR = cameraR.read()
        ret, imgL = cameraL.read()

        resL = skinmask(imgL)
        resR = skinmask(imgR)


        cv2.imshow('imgR', imgR)
        cv2.imshow('imgL', imgL)
        cv2.imshow('res', resL)


        Left_rectified, Right_rectified= rectified(imgR, imgL)
        cv2.imshow('rectified', Left_rectified)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cameraL.release()
    cameraR.release()
    cv2.destroyWindow('imgR')
    cv2.destroyWindow('imgL')
    cv2.destroyWindow('res')
    cv2.destroyWindow('rectified')




