import cv2
import os
import sys
import pyrealsense2
from realsense_depth import *
from HandEyeCalibration_class import CameraCalibration
from realsense_depth import DepthCamera
from gen3_gripper_pose import GripperPose 
import matplotlib.pyplot as plt

image_folder = "Cal2/RGBImgs/"
TS_base2ee_folder = "Cal2/T_base2ee/"
cartesian_pose_folder = "Cal2/GripperPose/"

# Initialize Camera Intel Realsense
dc = DepthCamera()
gp = GripperPose()

def find_chessboard_corners(images, pattern_size, ShowCorners=False):
    """Finds the chessboard patterns and, if ShowImage is True, shows the images with the corners"""
    chessboard_corners = []
    IndexWithImg = []
    i = 0
    print("Finding corners...")
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)
        if ShowCorners:
            #plot image using maplotlib. The title should "Detected corner in image: " + i
            cv2.imshow("corners", image)
            cv2.waitKey(1)
        else:
            print("No chessboard found in image: ", i)
    return ret
# 欧拉角转换为旋转矩阵
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1,0,0],
                    [0,np.cos(theta[0]),-np.sin(theta[0])],
                    [0,np.sin(theta[0]),np.cos(theta[0])]])

    R_y = np.array([[np.cos(theta[1]),0,np.sin(theta[2])],
                    #[0,-1,0],
                    [0, 1, 0],
                    [-np.sin(theta[1]),0,np.cos(theta[1])]])

    R_z = np.array([[np.cos(theta[2]),-np.sin(theta[2]),0],
                    [np.sin(theta[2]),np.cos(theta[2]),0],
                    [0, 0,1]])

    #return np.dot(np.dot(R_z,R_y),R_x)
    # combined rotation matrix
    R = np.dot(R_z, R_y.dot(R_x))
    return R

def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R


def get_Ts_hand_in_base(end_effector_pose):
    Ts = [float(i) for i in end_effector_pose]
    # R_hand_in_base= eulerAnglesToRotationMatrix(np.array(Ts[3:]))
    R_hand_in_base = euler_angles_to_rotation_matrix(Ts[3], Ts[4], Ts[5])
    # R_hand_in_base= R.from_euler('xyz',np.array(Ts[3:]),degrees=True).as_matrix()
    T_hand_in_base = Ts[:3]

    # R T拼接
    Ts_hand_in_base = np.zeros((4, 4), np.float64)
    Ts_hand_in_base[:3, :3] = R_hand_in_base
    Ts_hand_in_base[:3, 3] = np.array(T_hand_in_base).flatten()
    Ts_hand_in_base[3, 3] = 1
    return Ts_hand_in_base

save_number = input("Enter the number of datas you want to save: ")

image_num = 1

for i in range(int(save_number)):
    saved_flag = True
    while saved_flag:
        ret, depth_image, color_image = dc.get_frame()
        detect_corner_img = [color_image.copy()]
        ret = find_chessboard_corners(images=detect_corner_img, pattern_size=(13,7), ShowCorners=True)
        # 存储图像
        if not ret:
            print("Falid finding chessboard corners!")
            # 标定相机
        else:
            print('Avaliable')
            key = cv2.waitKey(500)

            if ret:
                if key == ord('s'):  # 如果按下's'键
                    # 获取当前系统时间并格式化
                    img_name = f'color_image{image_num:03}.png' 
                    img_save_path = os.path.join(image_folder,img_name)
                    cv2.imwrite(img_save_path, color_image)  # 保存图像
                    print(f"图像已保存至：{img_save_path}")
                    gripper_pose = gp.return_gripper_pose()
                    print(f"{gripper_pose[3:]}\n")
                    base2ee = get_Ts_hand_in_base(gripper_pose)
                    TBase2EE_name = f'TBase2EE_{image_num:03}.npz'
                    pose_name = f'Pose_{image_num:03}.npz'
                    TS_base2ee_save_path = os.path.join(TS_base2ee_folder,TBase2EE_name)
                    cartesian_pose_save_path = os.path.join(cartesian_pose_folder,pose_name)
                    np.savez(TS_base2ee_save_path, arr_0=base2ee)
                    np.savez(cartesian_pose_save_path, arr_0=np.array(gripper_pose))
                    saved_flag = False
                    image_num += 1
                elif key == ord('q'):  # 如果按下'q'键
                    cv2.destroyAllWindows()
                    sys.exit()
                else:
                    pass
            cv2.waitKey(1)