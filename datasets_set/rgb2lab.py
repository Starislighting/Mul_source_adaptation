import numpy as np
import cv2
import os
import random

def test_cvtcolor(img_ori):
    img_lab = cv2.cvtColor(img_ori, cv2.COLOR_RGB2Lab)
    return img_lab

def img_lab_stand(read_path, target_path, save_path):
    # 源域图像读取及计算
    source = cv2.imread(read_path)
    lab_source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    lab_source_cal = img_std_mean_get(lab_source)

    # 目标域图像读取及计算
    target = cv2.imread(target_path)
    lab_target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    lab_target_cal = img_std_mean_get(lab_target)

    # 进行转换
    source_w = lab_source.shape[0]
    source_h = lab_source.shape[1]
    save_img = np.zeros((source_w, source_h, 3), np.uint8)
    for x in range(source_w):
        for y in range(source_h):
            save_img[x, y][0] = (lab_source[x, y][0] - lab_source_cal[0]) / lab_source_cal[3] * lab_target_cal[3] + lab_target_cal[0]
            save_img[x, y][1] = (lab_source[x, y][1] - lab_source_cal[1]) / lab_source_cal[4] * lab_target_cal[4] + lab_target_cal[1]
            save_img[x, y][2] = (lab_source[x, y][2] - lab_source_cal[2]) / lab_source_cal[5] * lab_target_cal[5] + lab_target_cal[2]

    # 返回rgb空间并保存
    save_img_rgb = cv2.cvtColor(save_img, cv2.COLOR_LAB2BGR)
    cv2.imwrite(save_path, save_img_rgb)


# 从单个图像获取其标准差和均值（适用于三通道）
def img_std_mean_get(img):
    w = img.shape[0]
    h = img.shape[1]
    light_add = []
    a_add = []
    b_add = []
    for x in range(w):
        for y in range(h):
            light_add.append(img[x, y][0])
            a_add.append(img[x, y][1])
            b_add.append(img[x, y][2])
    # 计算原始图像的均值及标准差
    mean_light_add = np.mean(light_add)
    mean_a_add = np.mean(a_add)
    mean_b_add = np.mean(b_add)
    std_light_add = np.std(light_add)
    std_a_add = np.std(a_add)
    std_b_add = np.std(b_add)
    return [mean_light_add, mean_a_add, mean_b_add, std_light_add, std_a_add, std_b_add]


if __name__ == '__main__':

    source_root = 'source_imgs'
    target_root = 'target_imgs'
    save_root = 'save_imgs'

    source_lists = os.listdir(source_root)
    target_lists = os.listdir(target_root)

    for source_fp in source_lists:
        index = random.randint(0, len(target_lists) - 1)

        read_path = os.path.join(source_root, source_fp)
        target_path = os.path.join(target_root, target_lists[index])
        save_path = os.path.join(save_root, source_fp)

        img_lab_stand(read_path=read_path, target_path=target_path, save_path=save_path)

    # img_new = np.zeros((w, h, 3))
    # lab = np.zeros((w, h, 3))
    # for i in range(w):
    #     for j in range(h):
    #         Lab = RGB2Lab(img[i, j])
    #         lab[i, j] = (Lab[0], Lab[1], Lab[2])
    #
    # img_return = np.zeros((w, h, 3))
    # for i in range(w):
    #     for j in range(h):
    #         pixel_return = Lab2RGB(lab[i, j])
    #         img_return[i, j] = (pixel_return[0], pixel_return[1], pixel_return[2])

    # print(lab)
    # cv2.imshow('lab', lab)
    # cv2.waitKey(0)

    # print(img_ret)
    # cv2.imshow('img_return', img_ret)
    # cv2.waitKey(0)

    # cv2.imwrite(r'./test_img/dm18vvxy+.jpg', lab)
