import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''
    r,c,ch = img1.shape
#    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
  #  img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1[0]),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2[0]),5,color,-1)
    return img1,img2

def nothing(x):
    pass



# Defining the Parameter for stereoSGBM
min_disparity =  -1
max_disparity = 159
num_disparity = max_disparity - min_disparity
SADWindowSize = 5
uniqueness = 5
speckle_windows_size = 5
speckle_range = 5
P1 = 8*3*SADWindowSize**2
P2 = 32*3*SADWindowSize**2


# KL = np.array([[7242.753,0,1079.538], #[7242.753 0 1079.538; 0 7242.753 1018.846; 0 0 1]
#               [ 0,7242.753,1018.846],
#               [ 0,0,1]])
#
# KR = np.array([[7242.753,0,1588.865],#[7242.753 0 1588.865; 0 7242.753 1018.846; 0 0 1]
#               [ 0,7242.753,1018.846],
#               [ 0,0,1]])
KL = np.array([[3997.684,0, 1176.728], #[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1]
              [ 0,3997.684,1011.728],
              [ 0,0,1]])

KR = np.array([[3997.684,0,1307.839],##[3997.684 0 1307.839; 0 3997.684 1011.728; 0 0 1]
              [ 0,3997.684,1011.728],
              [ 0,0,1]])
dist_coeff = None
b = 193.001 # Baseline
imgL = cv2.imread('E:\Stero Rectification\Motorcycle-perfect\im0.png')
imgR = cv2.imread('E:\Stero Rectification\Motorcycle-perfect\im1.png')

grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
image_size = grayL.shape

sift = cv2.xfeatures2d.SIFT_create()
kp1, desc1 = sift.detectAndCompute(grayL, None)
kp2, desc2 = sift.detectAndCompute(grayR, None)
bf = cv2.BFMatcher(crossCheck=False)
matches = bf.knnMatch(desc1, desc2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
good = sorted(good, key=lambda x: x.distance)
pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

E, mask = cv2.findEssentialMat(pts1, pts2, KL, method=cv2.FM_RANSAC, prob=0.99,
                               threshold=0.4, mask=None)
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]
print("intial = ",pts1[:5],pts2[:5])

points,R,t,mask = cv2.recoverPose(E,pts1,pts2,R = None,t = None,mask = None)
K_inv = np.linalg.inv(KL)
F = K_inv.T @ E @ K_inv

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
#img2,_ = drawlines(imgL,imgR,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
#img1,_ = drawlines(imgR,imgL,lines2,pts2,pts1)
print("F = ",F)
###################
R1,R2,P1,P2= cv2.stereoRectify(KL,None,KL,None,(image_size[1],image_size[0]),R,t,flags = cv2.CALIB_ZERO_DISPARITY)[:4]

mapx1,mapy1 = cv2.initUndistortRectifyMap(KL,None,R1,P1,(image_size[1],image_size[0]),cv2.CV_16SC2)
mapx2,mapy2 = cv2.initUndistortRectifyMap(KL,None,R2,P2,(image_size[1],image_size[0]),cv2.CV_16SC2)
print("shape = ",mapx1.shape,mapy1.shape)

rectified_imgL = cv2.remap(imgL,mapx1,mapy1,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
rectified_imgR = cv2.remap(imgR,mapx2,mapy2,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)

cv2.imwrite('Rectified0.jpg',rectified_imgL)
cv2.imwrite('Rectified1.jpg',rectified_imgR)
###################
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
#img4,_ = drawlines(rectified_imgL,rectified_imgR,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
#img3,_ = drawlines(rectified_imgR,rectified_imgL,lines2,pts2,pts1)

rectified_imgL = cv2.pyrDown(rectified_imgL)
rectified_imgR = cv2.pyrDown(rectified_imgR)

left_matcher = cv2.StereoSGBM_create(minDisparity=min_disparity,numDisparities=num_disparity,blockSize=SADWindowSize
                               ,P1= 8*3*SADWindowSize**2,P2=32*3*SADWindowSize**2,uniquenessRatio=uniqueness,disp12MaxDiff=2,
                                speckleWindowSize=speckle_windows_size,speckleRange=speckle_range)

disparity = left_matcher.compute(rectified_imgL,rectified_imgR)

cv2.imshow("disparity_left",disparity)
cv2.imwrite("left_disparity.png",disparity)
##################
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
right_disparity = right_matcher.compute(rectified_imgR,rectified_imgL)
cv2.imshow('rigth_disparity',right_disparity)
sigma = 1.5
lambda_ = 8000
wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
wls.setLambda(lambda_)
wls.setSigmaColor(sigma)
filtered_disparity = wls.filter(disparity,rectified_imgL,disparity_map_right = right_disparity)
cv2.filterSpeckles(filtered_disparity,0,400,max_disparity-5)
_,disparity = cv2.threshold(filtered_disparity,0,max_disparity*16,cv2.THRESH_TOZERO)
filtered_disparity = (filtered_disparity/16).astype(np.uint8)
print(disparity.min())
cv2.imshow('filter',filtered_disparity)
cv2.imwrite("wls_disparity.png",filtered_disparity)

mask = disparity > disparity.min()

Q = np.float32([[1,0,0,-KL[0,2]],
                [0,1,0,-KL[1,2]],
                [0,0,0,KL[0,0]],
                [0,0,-1/b,(KL[0,2]-KR[0,2])/b]])
points = cv2.reprojectImageTo3D(filtered_disparity,Q)
print(points.shape)
points = points.reshape(-1,3)

color = rectified_imgL.reshape(-1,3)
color = color/255
color = np.flip(color,axis = 1)
print(color.shape)
xyzrbg = np.concatenate((points,color),axis=1)
print(xyzrbg.shape)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyzrbg[:,:3])
pcd.colors = o3d.utility.Vector3dVector(xyzrbg[:,3:])
#pcd.paint_uniform_color([1,0.5,0])
o3d.io.write_point_cloud('data.ply',pcd)

o3d.visualization.draw_geometries([pcd])
cv2.waitKey(0)
cv2.destroyAllWindows()
