import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from statical_camera_info import get_camera_intrinsics

class CameraCalibration:
    """
    Camera calibration class for Eye-to-Hand calibration. 
    This class takes as input a folder with images and a folder with the corresponding Base2endeffector transforms
    and outputs the intrinsic matrix in a .npz file. It also performs hand-eye calibration and saves those results in a .npz file.
    The images with the corner detection are saved in a folder called 'DetectedCorners'.

    Optional parameters:
    pattern_size: the number of corners in the chessboard pattern, default is (8,14)
    square_size: the size of the squares in the chessboard pattern, default is 15/1000
    ShowProjectError: if True, it will show the reprojection error for each image in a bar plot, default is False
    ShowCorners: if True, it will show the chessboard corners for each image, default is False
    """

    def __init__(self, image_folder, Transforms_folder, pattern_size=(7, 13), square_size=15/1000, ShowProjectError=False, ShowCorners=False):

        # Initiate parameters
        self.pattern_size = pattern_size
        self.square_size = square_size

        # Load images and joint positions
        self.image_files = sorted(glob.glob(f'{image_folder}/*.png'))
        self.transform_files = sorted(glob.glob(f'{Transforms_folder}/*.npz'))
        self.images = [cv2.imread(f) for f in self.image_files]
        self.images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in self.images]
        self.All_T_base2EE_list = [np.load(f)['arr_0'] for f in self.transform_files]

        # Find chessboard corners and index of images with chessboard corners
        self.chessboard_corners, self.IndexWithImg = self.find_chessboard_corners(self.images, self.pattern_size, ShowCorners=ShowCorners)
        intrinsic_matrix, depth_scale, coefficients = get_camera_intrinsics()
        self.intrinsic_matrix = intrinsic_matrix

        # Remove transforms where corners weren't detected
        self.T_base2EE_list = [self.All_T_base2EE_list[i] for i in self.IndexWithImg]

        # Save intrinsic matrix
        np.savez("IntrinsicMatrix.npz", self.intrinsic_matrix)

        # Calculate camera extrinsics
        self.RTarget2Cam, self.TTarget2Cam = self.compute_camera_poses(self.chessboard_corners,
                                                                       self.pattern_size, self.square_size,
                                                                       self.intrinsic_matrix)

        # Convert to homogeneous transformation matrix
        self.T_target2cam = [np.concatenate((R, T), axis=1) for R, T in zip(self.RTarget2Cam, self.TTarget2Cam)]
        for i in range(len(self.T_target2cam)):
            self.T_target2cam[i] = np.concatenate((self.T_target2cam[i], np.array([[0, 0, 0, 1]])), axis=0)

        # Calculate T_cam2target
        self.T_cam2target = [np.linalg.inv(T) for T in self.T_target2cam]
        self.R_cam2target = [T[:3, :3] for T in self.T_cam2target]
        self.R_vec_cam2target = [cv2.Rodrigues(R)[0] for R in self.R_cam2target]
        self.T_cam2target = [T[:3, 3] for T in self.T_cam2target]  # 4x4 transformation matrix

        # Calculate T_Base2EE
        print(f'{self.T_base2EE_list}\n')

        self.TEE2Base = [np.linalg.inv(T) for T in self.T_base2EE_list]
        self.REE2Base = [T[:3, :3] for T in self.TEE2Base]
        self.R_vecEE2Base = [cv2.Rodrigues(R)[0] for R in self.REE2Base]
        self.tEE2Base = [T[:3, 3] for T in self.TEE2Base]

        # Create folder to save final transforms
        if not os.path.exists("FinalTransforms"):
            os.mkdir("FinalTransforms")

        # Debugging: Check consistency in inputs for all methods
        print("\n*** Checking Input Consistency ***")
        for i in range(len(self.RTarget2Cam)):
            print(f"Image {i}:")
            print("RTarget2Cam:\n", self.RTarget2Cam[i])
            print("TTarget2Cam:\n", self.TTarget2Cam[i])
            print("REE2Base:\n", self.REE2Base[i])
            print("tEE2Base:\n", self.tEE2Base[i])
            print("-----")

        # Solve hand-eye calibration using Eye-to-Hand (Fixed Camera)
        for i in range(0, 5):
            print("Method:", i)
            
            # Perform the calibration
            try:
                R_cam2base, t_cam2base, R_gripper2target, t_gripper2target = cv2.calibrateRobotWorldHandEye(
                    self.RTarget2Cam,  # Rotation from target (chessboard) to camera
                    self.TTarget2Cam,  # Translation from target (chessboard) to camera
                    self.REE2Base,     # Rotation from end-effector to base
                    self.tEE2Base,     # Translation from end-effector to base
                    method=i
                )
            except cv2.error as e:
                print(f"Error during Method {i}: {e}")
                continue

            # Print the calibration results
            print("R_cam2base:\n", R_cam2base)
            print("t_cam2base:\n", t_cam2base)
            print("R_gripper2target:\n", R_gripper2target)
            print("t_gripper2target:\n", t_gripper2target)

            # Check if results are singular
            if np.allclose(R_cam2base, 0) or np.allclose(t_cam2base, 0):
                print(f"Method {i} produced a singular matrix, skipping this method.")
                continue

            # Create the 4x4 transformation matrix T_cam2base
            T_cam2base = np.concatenate((R_cam2base, t_cam2base), axis=1)
            T_cam2base = np.concatenate((T_cam2base, np.array([[0, 0, 0, 1]])), axis=0)

            # Save results in folder FinalTransforms
            np.savez(f"FinalTransforms/T_cam2base_Method_{i}.npz", T_cam2base)

            # Save the inverse transform too
            T_base2cam = np.linalg.inv(T_cam2base)
            np.savez(f"FinalTransforms/T_base2cam_Method_{i}.npz", T_base2cam)



    def find_chessboard_corners(self, images, pattern_size, ShowCorners=False):
        """Finds the chessboard patterns and, if ShowImage is True, shows the images with the corners"""
        chessboard_corners = []
        IndexWithImg = []
        i = 0
        print("Finding corners...")
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size)
            if ret:
                chessboard_corners.append(corners)

                cv2.drawChessboardCorners(image, pattern_size, corners, ret)
                if ShowCorners:
                    # Plot image using matplotlib. The title should "Detected corner in image: " + i
                    plt.imshow(image)
                    plt.title("Detected corner in image: " + str(i))
                    plt.show()

                # Save the image in a folder named "DetectedCorners"
                if not os.path.exists("DetectedCorners"):
                    os.makedirs("DetectedCorners")

                cv2.imwrite("DetectedCorners/DetectedCorners" + str(i) + ".png", image)

                IndexWithImg.append(i)
                i = i + 1
            else:
                print("No chessboard found in image: ", i)
                i = i + 1
        return chessboard_corners, IndexWithImg

    def compute_camera_poses(self, chessboard_corners, pattern_size, square_size, intrinsic_matrix, Testing=False):
        """Takes the chessboard corners and computes the camera poses"""
        # Create the object points. Object points are points in the real world that we want to find the pose of.
        object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
        object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        # Estimate the pose of the chessboard corners
        RTarget2Cam = []
        TTarget2Cam = []
        i = 1
        for corners in chessboard_corners:
            _, rvec, tvec = cv2.solvePnP(object_points, corners, intrinsic_matrix, None)
            # rvec is the rotation vector, tvec is the translation vector
            if Testing:
                print("Current iteration: ", i, " out of ", len(chessboard_corners[0]), " iterations.")
                print("rvec: ", rvec)
                print("--------------------")
            i += 1
            R, _ = cv2.Rodrigues(rvec)  # R is the rotation matrix from the target frame to the camera frame
            RTarget2Cam.append(R)
            TTarget2Cam.append(tvec)

        return RTarget2Cam, TTarget2Cam


if __name__ == "__main__":
    # Create an instance of the class
    image_folder = "Cal2/RGBImgs/"
    PoseFolder = "Cal2/T_base2ee/"
    calib = CameraCalibration(image_folder, PoseFolder, ShowProjectError=True)
