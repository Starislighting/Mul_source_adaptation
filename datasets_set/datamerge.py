import os
import shutil
import re
import time
# 原数据集目录
root_dir = 'datasetmerge'

os.makedirs('images', exist_ok=True)
os.makedirs('label_xml', exist_ok=True)
os.makedirs('label_txt', exist_ok=True)


# 进行文件移动以及重命名的操作
def move_act(file_dir):
    
    # 计数器
     
    person_lists = os.listdir(file_dir)
    for fp_name in person_lists:
        global image_count
        if bool(re.search('png$', fp_name)) or bool(re.search('jpg$', fp_name)):
            
            image_count = image_count + 1
            xml_name = str(image_count) + ".xml"
            img_name = str(image_count) + ".png"

            shutil.copy(os.path.join(file_dir, fp_name), os.path.join('images', img_name))
            shutil.copy(os.path.join(file_dir, fp_name[:-4]+".xml"), os.path.join('label_xml', xml_name))
            
            
def run_main():
	# 获取图片文件列表
	person_lists = os.listdir(root_dir)
	for i in person_lists:
	    file_dir = os.path.join(root_dir, i)
	    result = move_act(file_dir)
    
if __name__ == "__main__":


    global image_count
    image_count = 0
    run_main()


