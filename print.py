import numpy as np
import glob

file_list = glob.glob('/home/claire/Documents/Gen3-cv-HandEye/FinalTransforms/*.npz')
for filename in file_list:
    print(filename)
    data = np.load(filename)
    # 遍历并打印每个数组的内容
    for array_name in data.files:
        print(f"{array_name}:")
        print(data["arr_0"])
        # 处理 data


