import cv2
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from camera_L import cameradataL
from camera_R import cameradataR
from SGB import disparity_SGBM, sgbm
from tensorflow.keras import models
import tensorflow as tf
#
# cameraR = cv2.VideoCapture(2)
# cameraL = cv2.VideoCapture(1)

cameraR = cv2.VideoCapture('./outputR.avi')
cameraL = cv2.VideoCapture('./outputL.avi')

pathR = './imgR/'
pathL = './imgL/'

fnameR = 'imgR_'
fnameL = 'imgL_'

stime = 0
model = models.load_model('cnn_model_new.h5')


# model_RGB = models.load_model('cnn_model_RGB.h5')
def imgin():
    i = 500

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
            cameradataL()
            cameradataR()
            break

    cameraL.release()
    cameraR.release()
    cv2.destroyWindow("imgR")
    cv2.destroyWindow("imgL")


def stereoCalibrate_camera():
    frameR = cv2.imread('./imgR/imgR_119.jpg')
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

    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpointsR, imgpointsR, imgpointsL, mtxR, distR, mtxL,
                                                               distL, frameRG.shape[::-1], criteria_stereo, flags)
    print(retS, MLS, dLS, MRS, dRS, R, T, E, F)

    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, frameRG.shape[::-1], R, T, flags=flags,
                                                      alpha=0, newImageSize=(0, 0))

    print('RL, RR, PL, PR, Q, roiL, roiR', RL, RR, PL, PR, Q, roiL, roiR)

    Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                  frameRG.shape[::-1], cv2.CV_16SC2)

    Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                   frameRG.shape[::-1], cv2.CV_16SC2)
    return Left_Stereo_Map, Right_Stereo_Map, Q


def rectified(capL, cpaR):
    # frameR = cv2.imread('./imgR/imgR_101.jpg', 0)
    # frameL = cv2.imread('./imgL/imgL_101.jpg', 0)
    # frameR = cpaR.read()
    # frameL = capL.read()

    # Left_rectified = cv2.remap(capL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LINEAR)
    # im_L = Image.fromarray(Left_rectified)
    #
    # Right_rectified = cv2.remap(cpaR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LINEAR)
    # im_R = Image.fromarray(Right_rectified)
    # cv2.imshow('iml', Left_rectified)
    # cv2.imshow('imr', Right_rectified)

    Left_rectified, Right_rectified = disparity_SGBM(capL, cpaR)

    return Left_rectified, Right_rectified


def skinmask(roi):
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 轉換至YCrCb空間
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (7, 7), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu處理
    res = cv2.bitwise_and(roi, roi, mask=skin)
    ker = np.ones((5, 5), np.uint8)
    res = cv2.dilate(res, ker, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    res = cv2.erode(res, kernel)

    return res


def find_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    con, hie = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bounding_box = [cv2.boundingRect(cnt) for cnt in con]

    return bounding_box, con


def findCon(img, rimg):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    con, hie = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bounding_box = [cv2.boundingRect(cnt) for cnt in con]
    # cv2.drawContours(rimg, con, -1, (255, 255, 255), 3)
    imgarr = np.array(gray, dtype=bool)
    rimg *= imgarr

    # for bbox in bounding_box:
    #     [x, y, w, h] = bbox
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    exist = (rimg != 0)
    mean = rimg.sum() / exist.sum()
    # print('deep:'+str(mean))
    # rimg = imgarr * mean

    return rimg.astype('uint8'), img, mean


def point(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    con_frame = np.copy(th)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(len(contours)):
        cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
        hull = cv2.convexHull(contours[i])
        cv2.polylines(image, [hull], True, (0, 255, 0), 2)
    return image


def findCon_RGB(img, rimg):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    con, hie = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for j in range(len(con)):
        n, h, w = np.shape(con[j])
        if n >= 300 and h >= 1 and w >= 2:
            M = cv2.moments(con[j])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.drawContours(rimg, con, -1, (255, 255, 255), 1)
            # cv2.circle(rimg, (cx, cy), 7, (255, 255, 255), -1)
            for i in range(n):
                if i % 60 == 0:
                    arr = con[j]
                    cv2.line(rimg, (cx, cy), arr[i][0], (255, 255, 255), 1)
                    continue
    return rimg


# def cnn(img, img_d, switch=True):
#     img = findCon_RGB(img, img)
#     img = cv2.resize(img, (64, 48), interpolation=cv2.INTER_AREA)
#     cv2.imshow('tt', img)
#
#     img_d = cv2.resize(img_d, (64, 48), interpolation=cv2.INTER_AREA)
#     if switch:
#         img = img.T
#         img_d = img_d.T
#
#         data = np.empty((48, 64, 4))
#         data = data.T
#         data[0] = img[0]
#         data[1] = img[1]
#         data[2] = img[2]
#         data[3] = img_d
#         data = data.T
#         i_data = np.empty((1, 48, 64, 4))
#         i_data[0] = data / 255.0
#         i_data = tf.keras.utils.normalize(i_data)
#         predictions = (model.predict(i_data) > 0.8)
#
#     else:
#         i_data = np.empty((1, 48, 64, 3))
#         i_data[0] = img / 255.0
#         # predictions = (model_RGB.predict(i_data) > 0.5)
#
#     return predictions


def cnn_roi(img, img_d, poin, switch=True):
    [x, y, w, h] = poin

    image = np.empty((480, 640, 3), dtype='uint8')
    image[:] = (0, 0, 0)

    image_d = np.empty((480, 640), dtype='uint8')
    image_d[:] = 0
    image[y:y + h, x:x + w] = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_d[y:y + h, x:x + w] = img_d
    image = findCon_RGB(image, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

    imgarr = np.array(gray, dtype=bool)
    exist = (image_d != 0)

    if image_d.sum() != 0:
        mean = image_d.sum() / exist.sum()
    else:
        mean = 1.0

    image_d = imgarr * mean
    image_d = image_d.astype('uint8')
    cv2.imshow('deep_mean', image_d)
    image = cv2.resize(image, (64, 48), interpolation=cv2.INTER_AREA)
    image_d = cv2.resize(image_d, (64, 48), interpolation=cv2.INTER_AREA)

    if switch:
        img = image.T
        img_d = image_d.T

        data = np.empty((48, 64, 4))
        data = data.T
        data[0] = img[0] + 20
        data[1] = img[1]
        data[2] = img[2]
        data[3] = img_d
        data = data.T
        i_data = np.empty((1, 48, 64, 4))
        i_data[0] = data / 255.0
        i_data = tf.keras.utils.normalize(i_data)
        predictions = (model.predict(i_data)>0.99)

    # else:
    #     i_data = np.empty((1, 48, 64, 3))
    #     i_data[0] = img / 255.0
    #     predictions = (model_RGB.predict(i_data) > 0.5)

    # y = np.argmax(predictions, axis=1)

    # print(y)
    # print('***'+ str(predictions[0, y[0]]))

    return predictions, mean


# imgin()
# Left_Stereo_Map, Right_Stereo_Map, Q = stereoCalibrate_camera()

class main:
    def __init__(self, capL, capR, model):
        self.cameraR = capR
        self.cameraL = capL
        self.model = model
        self.imgL = main.img_readL(self)
        self.diff = main.img_readL(self)
        self.imgR = main.img_readR(self)
        self.resL = main.skin_mask(self, self.imgL)
        self.resR = main.skin_mask(self, self.imgR)
        self.Left_rectified = main.sgbm_rectified(self)
        self.j = 96
        self.s = 0

    def img_readL(self):
        ret, img = self.cameraL.read()
        return img

    def img_readR(self):
        ret, img = self.cameraR.read()
        return img

    def skin_mask(self, roi):
        YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
        (y, cr, cb) = cv2.split(YCrCb)
        cr1 = cv2.GaussianBlur(cr, (7, 7), 0)
        _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res = cv2.bitwise_and(roi, roi, mask=skin)
        # ker = np.ones((3, 3), np.uint8)
        # res = cv2.dilate(res, ker, iterations=2)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        # res = cv2.erode(res, kernel)
        return res

    def sgbm_rectified(self):
        # frameR = cv2.imread('./imgR/imgR_101.jpg', 0)
        # frameL = cv2.imread('./imgL/imgL_101.jpg', 0)
        # frameR = cpaR.read()
        # frameL = capL.read()

        # Left_rectified = cv2.remap(capL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LINEAR)
        # im_L = Image.fromarray(Left_rectified)
        #
        # Right_rectified = cv2.remap(cpaR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LINEAR)
        # im_R = Image.fromarray(Right_rectified)
        # Left_rectified, Right_rectified = disparity_SGBM(self.resL, self.imgR)
        Left_rectified = sgbm(self.imgL, self.imgR)

        return Left_rectified.astype('uint8')


    def find_bbox(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        con, hie = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        bounding_box = [cv2.boundingRect(cnt) for cnt in con]

        return bounding_box, con

    def run(self):
        self.imgL = main.img_readL(self)
        self.imgR = main.img_readR(self)
        tm = cv2.TickMeter()
        tm.start()
        diff = backsub.apply(self.imgL)
        # diff = cv2.absdiff(cv2.cvtColor(self.imgL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.diff, cv2.COLOR_BGR2GRAY))
        # _, diff = cv2.threshold(diff, 5, 1, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        diff = cv2.dilate(diff, kernel, iterations=5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        diff = cv2.erode(diff, kernel, iterations=3)
        cv2.imshow('diff', diff)
        # self.imgL = self.imgL * cv2.merge([diff, diff, diff])

        self.resL = main.skin_mask(self, self.imgL)
        self.resR = main.skin_mask(self, self.imgR)
        self.Left_rectified = main.sgbm_rectified(self)
        # cv2.imshow('rectified', self.Left_rectified)

        bounding_box, con = main.find_bbox(self, self.resL*cv2.merge([diff, diff, diff]))

        if len(bounding_box):
            for i in range(len(bounding_box)):
                [x, y, w, h] = bounding_box[i]
                if w * h >2000:
                    roi = self.resL[y:y + h, x:x + w]
                    roi_d = self.Left_rectified[y:y + h, x:x + w]
                    # image = np.empty((480, 640, 3), dtype='uint8')
                    # image[:] = (0, 0, 0)
                    #
                    # image_d = np.empty((480, 640), dtype='uint8')
                    # image_d[:] = 0
                    # image[y:y + h, x:x + w] = roi
                    # image_d[y:y + h, x:x + w] = roi_d
                    # cv2.imwrite('./train_data/errer/' + str(self.s) + '.jpg', image)
                    # cv2.imwrite('./train_data/errer_d/' + str(self.s) + '.jpg', image_d)
                    self.s += 1
                    predictions, deep = cnn_roi(roi* cv2.merge([diff[y:y + h, x:x + w], diff[y:y + h, x:x + w], diff[y:y + h, x:x + w]]), roi_d, bounding_box[i])
                    print(str(i) + str(predictions[0]))


                    # cv2.rectangle(imgL, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    # if predictions[0][0] or predictions[0][1] or predictions[0][2]:
                    #     cv2.putText(self.imgL, 'pose_1', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 4)
                    #     cv2.putText(self.imgL, 'deep:' + str(int(deep)), (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    #                 (0, 0, 255), 2, 4)
                    #     cv2.rectangle(self.imgL, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # elif predictions[0][3] or predictions[0][4] or predictions[0][5]:
                    #     cv2.putText(self.imgL, 'pose_2', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 4)
                    #     cv2.putText(self.imgL, 'deep:' + str(int(deep)), (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    #                 (0, 0, 255), 2, 4)
                    #     cv2.rectangle(self.imgL, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # else:
                    #     pass

        exist_T = (cv2.cvtColor(self.diff, cv2.COLOR_BGR2GRAY) != 0)
        tm.stop()
        if (exist_T.sum() / diff.sum()) == 1:
            self.diff = main.img_readL(self)

        cv2.putText(self.imgL, 'FPS:' + str(int(tm.getFPS())), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 4)
        # cv2.imshow('imgR', self.imgR)
        cv2.imshow('imgL', self.imgL)
        cv2.imshow('res', self.resL)
        # Left_rectified = cv2.Canny(Left_rectified, 100, 200)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            self.j += 1
            cv2.imwrite('./new_train_data/train/' + str(self.j) + str(int(deep)) + '.jpg', self.resL)
            print('./train/' + str(self.j) + '.jpg' + './train_D/' + str(self.j) + '.jpg')

        if cv2.waitKey(1) & 0xFF == ord('p'):
            p = cnn_roi(self.resL, self.Left_rectified)
            print(p[0])
        if cv2.waitKey(1) & 0xFF == ord('d'):
            self.diff = main.img_readL(self)
            print('updata')

    def close(self):
        self.cameraL.release()
        self.cameraR.release()
        # cv2.destroyWindow('imgR')
        # cv2.destroyWindow('imgL')
        # cv2.destroyWindow('res')
        # cv2.destroyWindow('rectified')
        cv2.destroyAllWindows()


if __name__ == '__main__':
    j = 500
    fps = 0
    backsub = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)

    a = main(cameraL, cameraR, model)
    while True:
        a.run()

