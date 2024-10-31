import numpy as np

# 读取 .npz 文件
data = np.load('/home/claire/Documents/Gen3-cv-HandEye/Cal2/GripperPose/Pose_001.npz')

# 打印文件中的所有数组的名称
print("文件中的数组名称：", data.files)

# 遍历并打印每个数组的内容
for array_name in data.files:
    print(f"{array_name}:")
    print(data["arr_0"])
