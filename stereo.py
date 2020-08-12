import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def decompose_essential_matrix(E,K,pts1,pts2):
    [U, D, V] = np.linalg.svd(E)
    diag_arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    new_E = U @ diag_arr @ V
    [U, D, V] = np.linalg.svd(new_E)
    Y = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = - U @ Y @ V
    R2 = - U @ Y.T @ V
    t = U[:, 2].reshape(3, 1)
    R_mat = np.array([R1, R1, R2, R2])
    T_mat = np.array([t, -t, t, -t])
    P1 = np.zeros((3, 4))
    P1[:, :3] = np.eye(3)
    P1 = K @ P1
    print(R1, "\n", R2)
    for i in range(4):
        P2 = np.concatenate((R_mat[i], T_mat[i]), axis=1)
        P2 = K @ P2
        world_pts = cv2.triangulatePoints(P1, P2, pts1, pts2)
        X, Y, Z = world_pts[:3, :] / world_pts[3, :]
        Z_ = R_mat[i][2, 0] * X + R_mat[i][2, 1] * Y + R_mat[i][2, 2] * Z + T_mat[i][2]
        print(len(np.where(Z < 0)[0]), len(np.where(Z_ < 0)[0]))
        if len(np.where(Z < 0)[0]) == 0:
            R = R_mat[i]
            t = T_mat[i]
            break
    return R,t

def drawlines(img1,img2,lines,pts1,pts2):
    r,c,ch = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1[0]),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2[0]),5,color,-1)
    return img1,img2

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


KL = np.array([[3997.684,0, 1176.728],
              [ 0,3997.684,1011.728],
              [ 0,0,1]])

KR = np.array([[3997.684,0,1307.839],
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

#R,t = decompose_essential_matrix(E,KL,pts1,pts2)
points,R,t,mask = cv2.recoverPose(E,pts1,pts2,R = None,t = None,mask = None)
K_inv = np.linalg.inv(KL)
F = K_inv.T @ E @ K_inv
print("R = {} \n t = {}".format(R,t))
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)

R1,R2,P1,P2= cv2.stereoRectify(KL,None,KL,None,(image_size[1],image_size[0]),R,t,flags = cv2.CALIB_ZERO_DISPARITY)[:4]
#print(R1 @ R2.T) # these gives the rotation between the two camera
mapx1,mapy1 = cv2.initUndistortRectifyMap(KL,None,R1,P1,(image_size[1],image_size[0]),cv2.CV_16SC2)
mapx2,mapy2 = cv2.initUndistortRectifyMap(KL,None,R2,P2,(image_size[1],image_size[0]),cv2.CV_16SC2)
print("shape = ",mapx1.shape,mapy1.shape)

rectified_imgL = cv2.remap(imgL,mapx1,mapy1,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
rectified_imgR = cv2.remap(imgR,mapx2,mapy2,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)

rectified_imgL = cv2.pyrDown(rectified_imgL)
rectified_imgR = cv2.pyrDown(rectified_imgR)

left_matcher = cv2.StereoSGBM_create(minDisparity=min_disparity,numDisparities=num_disparity,blockSize=SADWindowSize
                               ,P1= 8*3*SADWindowSize**2,P2=32*3*SADWindowSize**2,uniquenessRatio=uniqueness,disp12MaxDiff=2,
                                speckleWindowSize=speckle_windows_size,speckleRange=speckle_range)

left_disparity = left_matcher.compute(rectified_imgL,rectified_imgR)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
right_disparity = right_matcher.compute(rectified_imgR,rectified_imgL)

# wls filtering
sigma = 1.5
lambda_ = 8000
wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
wls.setLambda(lambda_)
wls.setSigmaColor(sigma)
filtered_disparity = wls.filter(left_disparity,rectified_imgL,disparity_map_right = right_disparity)
cv2.filterSpeckles(filtered_disparity,0,400,max_disparity-5)
_,filtered_disparity = cv2.threshold(filtered_disparity,0,max_disparity*16,cv2.THRESH_TOZERO)
filtered_disparity = (filtered_disparity/16).astype(np.uint8)

cv2.imshow('filter',filtered_disparity)
cv2.imwrite("wls_disparity.png",filtered_disparity)

depth_map = KL[0,0]*b / (filtered_disparity)
depth_map = depth_map.astype('uint16')
cv2.imshow('depth map',depth_map)

# Reprojection matrix
Q = np.float32([[1,0,0,-KL[0,2]],
                [0,1,0,-KL[1,2]],
                [0,0,0,KL[0,0]],
                [0,0,-1/b,(KL[0,2]-KR[0,2])/b]])


points = cv2.reprojectImageTo3D(filtered_disparity,Q)
points = points.reshape(-1,3)
color = rectified_imgL.reshape(-1,3)
color = np.flip(color,axis = 1)/255
xyzrbg = np.concatenate((points,color),axis=1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyzrbg[:,:3])
pcd.colors = o3d.utility.Vector3dVector(xyzrbg[:,3:])
o3d.io.write_point_cloud('data.ply',pcd)
o3d.visualization.draw_geometries([pcd])
cv2.waitKey(0)
cv2.destroyAllWindows()
