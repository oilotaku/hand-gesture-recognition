import cv2
import numpy as np
import glob

def cameradataL():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00000001)

    objp = np.zeros((6 * 12, 3), np.float32)
    objp[:, :2] = np.mgrid[0:12, 0:6].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.


    images = glob.glob(r"./imgL/*.jpg")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (12, 6), None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (25, 13), (-1, -1), criteria)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(img, (12, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints , gray.shape[::-1], None, None)
    np.savez('cmpL', retL=ret, mtxL=mtx, distL=dist, rvecsL=rvecs, tvecsL=tvecs, objpointsL=objpoints, imgpointsL=imgpoints)

    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints