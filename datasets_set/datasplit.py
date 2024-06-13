import os
import random
import shutil

# 原数据集目录
root_dir = 'gta5'
# 划分比例
train_ratio = 0.8
valid_ratio = 0.2

# 设置随机种子
random.seed(42)

# 拆分后txt文件目录
save_dir = 'Segmentation'
save_path = os.path.join(root_dir, save_dir)
os.makedirs(save_path, exist_ok=True)


# 获取图片及mask文件列表
image_files = os.listdir(os.path.join(root_dir, 'images'))
mask_files = os.listdir(os.path.join(root_dir, 'masks'))
    
# 随机打乱文件列表
combined_files = list(zip(image_files, mask_files))
random.shuffle(combined_files)
image_files_shuffled, mask_files_shuffled = zip(*combined_files)

# 根据比例计算划分的边界索引
train_bound = int(train_ratio * len(image_files_shuffled))
valid_bound = int((train_ratio + valid_ratio) * len(image_files_shuffled))

