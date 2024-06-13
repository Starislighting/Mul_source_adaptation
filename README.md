# Multi-Source Domain Adaptation with Collaborative Learning for Semantic Segmentation

本代码库基于CVPR2021Multi-Source Domain Adaptation with Collaborative Learning for Semantic Segmentation论文，对其中所提及的语义分割任务中多源域协作学习的领域适应方法及相关训练策略进行了复现，基线模型参考文章选用了deeplabv3模型，但由于文章源代码并未开源，部分超参数设置及步骤实现只能依据文章中提及的描述和公式进行复现。最终对文章中提及的方法和训练策略进行了基本的复现，但在实际训练模型过程中，训练效果并未达到预期。

## 1 论文介绍

《Multi-Source Domain Adaptation with Collaborative Learning for Semantic Segmentation》论文主要介绍了多源域协作学习下实现领域适应的方法，并在语义分割的实际任务上对方法进行了一定的实现，从论文的实验设计及结果上来看，文章提及的方法能够有效提升领域适应的能力。文章中提及的方法分为基于LAB色彩格式的图像空间统一化、多源域间协作学习、未标记目标域协作学习三个步骤。

### 1.1 背景介绍

文章研究的大背景为领域泛化问题。领域泛化问题的出发点为在训练过程中模型所接触到的数据和真实情况可能存在较大的差异，进而可能导致训练得到的模型无法在真实环境下获得一个较好的效果，而领域泛化就旨在增强模型在全新未见过的数据域中的表现效果，这种能力提升被称为模型泛化能力的提升。以下会将训练时所使用的域称为源域，因为可能有多个训练域，因此就存在多个源域，而实际使用的域则为目标域。

领域适应问题与领域泛化问题较为相像，其目的基本一致，均是希望在目标域上收获较好的效果，但存在一定的区别，领域泛化更强调目标域不可见，但领域适应中目标域的部分往往是可以访问的，只是其相应的标记数据可能无法访问。

因此领域适应问题也可以看做领域泛化问题的一种退化。

在领域泛化问题的研究中，常用的有基于数据、基于特征和基于策略迁移的方法，这篇论文中所提出的多源域协作学习更多的可以看做为一种策略迁移的方法。

### 1.2 论文方法介绍

论文提出的方法主要包括基于LAB色彩格式的图像风格空间统一化、多源域间协作学习、未标记目标域协作学习三个步骤，其出发点为目标域部分可访问，最终得到多个具有较好泛化效果的模型，同时文章进行了大量的实验来验证其方法的有效性，并设置了消融实验来验证方法各步骤的有效性。基本的方法框架如下图所示。

![image-20240614021835316](https://github.com/Starislighting/Mul_source_adaptation/test_imgs/1.png)

#### 1.2.1 基于LAB色彩格式的图像风格空间统一化

论文中指出，域之间的域差异主要在于图像的外观，即颜色和纹理，这种差异将进一步增加多源域适应的难度，而有效的风格转换可以在一定程度上减少这种差异，进而降低多源域适应的难度，提升模型的泛化能力。论文从简单高效的角度出发，提出了一种图像转换方法，通过对齐像素值的分布，将源域中的图像样式转换为目标域的样式。同时由于LAB颜色空间的色域相对RGB颜色空间更易区分处理，在LAB色彩空间上进行的图像风格空间的统一化比直接在RGB颜色空间上操作更为简单，效果更好，因此论文将在LAB颜色空间上进行统一化处理。相关公式如下：

![image-20240614022856833](/home/kolo/.config/Typora/typora-user-images/image-20240614022856833.png)

其中将源域和目标域的颜色空间从RGB转换到LAB中，分别计算源域和目标域LAB颜色空间中各个通道下的均值和标准差，通过公式方式对源域图像进行处理，最终将处理后的源域图像从LAB色彩空间转换为RGB色彩空间

#### 1.2.2 多源域间协作学习

多源域协作学习是指充分利用各个领域之间的差异性，同时在多个源域间训练得到多个模型，训练过程中不同源域间会进行协同处理，以此来尽可能避免模型落入某个特定的训练源域中，使其能够在更多的目标域上收获一个较好的效果。

其公式如下：

![image-20240614023410174](/home/kolo/.config/Typora/typora-user-images/image-20240614023410174.png)

![image-20240614023430484](/home/kolo/.config/Typora/typora-user-images/image-20240614023430484.png)

![image-20240614023444251](/home/kolo/.config/Typora/typora-user-images/image-20240614023444251.png)

![image-20240614023520646](/home/kolo/.config/Typora/typora-user-images/image-20240614023520646.png)

大致思路为在一次训练过程中同时训练多个模型，对于其中任意一个模型，其目标函数在原有交叉熵结果的基础上，加上一个协作损失，协作损失为其他模型在本源域上预测结果同本模型在本源域上预测结果的KL散度的平均值。

#### 1.2.3 未标记目标域协作学习

论文中指出，在实践中，未标记的目标域数据的收集通常相对容易且便宜。此外，在目标域数据上训练的模型可以学习更好的特征，并在目标域上表现更好。因此提出了一种协作学习方法，充分利用目标域中未标记的图像，并进一步提高模型在目标域上的性能。其公式和伪代码如下：

![image-20240614023859171](/home/kolo/.config/Typora/typora-user-images/image-20240614023859171.png)

![image-20240614023915554](/home/kolo/.config/Typora/typora-user-images/image-20240614023915554.png)

![image-20240614023944816](/home/kolo/.config/Typora/typora-user-images/image-20240614023944816.png)

大致思路为，在一次训练中，由于同时会对多个模型进行训练，每一个训练结果均能够对随机一张未标记的目标域图像进行预测，将预测结果进行softmax处理后取均值，随后用argmax的思想生成其伪标签，进而就可以将伪标签同任意模型在未标记目标域图像上的训练结果计算交叉熵，这一交叉熵乘以一个与当前训练轮次相关的值，随后就可以加入至任意一个模型的目标函数中。

#### 1.2.4 整体的目标函数

![image-20240614024258373](/home/kolo/.config/Typora/typora-user-images/image-20240614024258373.png)

整体的目标函数如上图所示，可以看出共分为三个部分，一个是最基本的交叉熵，第二个是多源域协作学习的损失，最后一个是未标记目标域协作学习的损失，最后公式中前文未提及的两个参数为超参数，在训练前进行预定义。

### 1.3 相关数据集介绍

这部分将简单介绍一下文章中用到的几个数据集

#### 1.3.1 GTA5合成数据集

GTA5数据集包括24966张密集注释图像，这些图像是从分辨率为1914×1052的游戏引擎合成的。其基本事实标签与城市景观一致，与Cityscapes的类别基本一致。

#### 1.3.2 cityscapes数据集

cityscapes数据集由5000幅真实世界的城市交通场景图像组成，分辨率为2048×1024，像素注释密集。该数据集分为2975个用于训练，500个用于验证，1525个用于测试。城市景观注释了33个类别，其中19个用于培训和评估。

#### 1.3.3 SYNTHIA数据集

SYNTHIA是一个大型合成数据集，由虚拟城市渲染的照片逼真帧组成。在实验中，使用SYNTHIA-RANDCITYSCAPES集合进行适应。它包含9400张分辨率为1280×760的图像，这些图像被注释为16类。与GTA5类似，它的标记也是自动生成的，并与Cityscapes兼容。

### 2.1 复现细节提前说明

- 由于没有源码进行参考，同时数据集上需要额外进行处理和制备，因此整体复现流程及实验流程主要在pytorch框架下进行，并使用MSAdapter将模型迁移到mindspore的生态下
- 由于复现过程涉及多源域，参考原论文，并考虑数据集的大小和相关资源消耗，因此主要选用了人工合成的GTA5游戏数据集以及cityscapes城市街道数据集，其中cityscape数据集由于样式较多，故将其中一部分数据集分离出来作为目标域数据集。因此最终本次复现共涉及三个数据集的使用，包括：GTA5数据集、cityscapes源域数据集、cityscapes目标域数据集。前两个数据集组成多源域，最后一个数据集为目标域。
- 参考原论文，基线模型选用了deeplabv3（其相对v2更为成熟一些），并事先在resnet101的网络上进行了简单的复现
- 相关预训练权重：deeplabv3_resnet50: https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth；deeplabv3_resnet101: https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth，预训练权重是在coco上预训练得到的，并对Voc数据集进行了调整，其类别数为21。在实验过程中，发现相关数据集在Voc上表现良好，但是由于gta5和cityscapes数据集其类别数为33（且其标记上同Voc存在很大差异），因此训练效果并不良好，在多轮调整下略有改善

### 2.2 服务器环境说明

* Ubuntu22.04

* 计算环境cuda11.3

* 硬件环境GPU为4090单卡（在最小batchsize下显存基本占满，大致需要24G+）

* Python3.8

* 其他必要库

  ```
  numpy==1.22.0
  torch==1.10.0
  torchvision==0.11.1
  opencv-python==4.9.0.80
  Pillow
  ```

### 2.3 项目文件目录

### 2.4 复现流程1-数据集的制备

- 共使用了三个数据集，包括：GTA5数据集、cityscapes源域数据集、cityscapes目标域数据集

- GTA5数据集下载官网：https://download.visinf.tu-darmstadt.de/data/from_games/

- cityscapes数据集下载官网：https://www.cityscapes-dataset.com/dataset-overview/

- 其中GTA5数据集将2000张图片及mask作为训练集、500张作为验证集；cityscapes源域数据集同样将2000张图片及mask作为训练集、500张作为验证集；cityscapes目标域数据集为250张与cityscapes源域数据集差异较大的图片

- 其中GTA5数据集需要提前生成train和val的清单，代码如下。

  ```
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
  ```

- GTA5和cityscapes数据集均需要根据其文件目录情况进行特定的数据集加载处理，需要在Voc加载（常见的默认用于复现的数据集）的基础上进行修改，以下为部分数据集加载的代码改写。

- GTA5部分

  ```
  def __init__(self, gta5_root, transforms=None, txt_name: str = "train.txt"):
      super(GTA5Segmentation, self).__init__()
  
      # 路径的构造
      image_dir = os.path.join(gta5_root, 'images')
      mask_dir = os.path.join(gta5_root, 'masks')
      txt_path = os.path.join(gta5_root, "Segmentation", txt_name)
  
      assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
  
      with open(os.path.join(txt_path), "r") as f:
          file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
  
      self.images = [os.path.join(image_dir, x + ".png") for x in file_names]
      self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
      assert (len(self.images) == len(self.masks))
      self.transforms = transforms
  ```

- cityscapes部分

  ```
  def __init__(self, root, split='train', mode='fine', target_type='semantic', transforms=None):
      self.root = os.path.expanduser(root)
      self.mode = 'gtFine'
      self.target_type = target_type
      self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
  
      self.targets_dir = os.path.join(self.root, self.mode, split)
      self.transform = transforms
  
      self.split = split
      self.images = []
      self.targets = []
  
      repeat_num = 1
      if split in ['val']:
          repeat_num = 4
      for i in range(repeat_num):
          for city in os.listdir(self.images_dir):
              img_dir = os.path.join(self.images_dir, city)
              target_dir = os.path.join(self.targets_dir, city)
  
              for file_name in os.listdir(img_dir):
                  self.images.append(os.path.join(img_dir, file_name))
                  target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                               self._get_target_suffix(self.mode, self.target_type))
                  self.targets.append(os.path.join(target_dir, target_name))
  ```

### 2.5 复现流程2-基于LAB色彩格式的图像风格空间统一化

这部分是对LAB色彩格式下图像风格空间统一化处理的复现，其使用了cv库进行处理，在原文中提及cuda可以加速相应处理，但并未找到相关资料，以下为相关示例代码。

在处理过程中，对源域内的每一张图片，均随机选择一张目标域图片作为风格空间统一化的对象，将其均转换到lab色彩空间中，对图片的每个通道计算其均值和标准差，根据公式进行统一化处理，随后转回rgb空间并保存。

```
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
```

### 2.6 复现流程3-在多源域间的协作学习复现

多源域协作学习本质上为训练策略的调整，在复现该部分的代码时，从数据集加载到训练、参数回传、保存均需做出相应的修改，这里主要以两源域作为示例进行修改。在两源域学习中，单次训练时需同时对两个模型进行训练，并通过kl散度计算协作损失，具体参见公式。

训练的入口示例如下，其中部分代码进行了额外的集成，这里不做展示：

```
def main_double_model(args):
    
    # 基本的训练配置
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    gta5_data_path = "/home/kolo/PycharmProjects/datasets/gta5/"
    city_data_path = "/home/kolo/PycharmProjects/datasets/cityscapes/cityscapesScripts/"

    # 用来保存训练以及验证过程中信息
    model1_results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 数据集加载
    model1_train_loader, model1_val_loader = dataset_load("GTA5", gta5_data_path, batch_size)
    # 模型和优化器的加载
    model1 = create_model(aux=args.aux, num_classes=num_classes)
    model1.to(device)
    model1_run_sets = model_inits(model1, model1_train_loader)

    # 模型2相关
    model2_results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model2_train_loader, model2_val_loader = dataset_load("cityscapes", city_data_path, batch_size)
    model2 = create_model(aux=args.aux, num_classes=num_classes)
    model2.to(device)
    model2_run_sets = model_inits(model1, model1_train_loader)

    # 计时开始
    start_time = time.time()

    # 训练迭代
    for epoch in range(args.start_epoch, args.epochs):

        mean_loss, lr, mean_loss_2, lr_2 = train_one_epoch_double_run(device, epoch, model1, model1_run_sets, model1_train_loader, model2,
                                                   model2_run_sets, model2_train_loader, source_val_loader=model2_val_loader, print_freq=args.print_freq)

        print("test model1")
        confmat = evaluate(model1, model2_val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        model_result_save(model1_results_file, model1, model1_run_sets, epoch, mean_loss, val_info, lr, model_name="GTA5")

        print("test model2")
        confmat_2 = evaluate(model2, model2_val_loader, device=device, num_classes=num_classes)
        val_info_2 = str(confmat_2)
        print(val_info_2)
        model_result_save(model2_results_file, model2, model2_run_sets, epoch, mean_loss_2, val_info_2, lr_2, model_name="CITY")

    # 计时结束
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
```

单轮训练的核心代码如下所示：

```
def train_one_epoch_double_run(device, epoch, model1, model1_run_sets, model1_train_loader, model2,
                               model2_run_sets, model2_train_loader, source_val_loader,
                               print_freq=10,):
    # model1的训练
    global lr, lr_2
    model1.train()
    model1_metric_logger = utils.MetricLogger(delimiter="  ")
    model1_metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header1 = "Model1" + ' Epoch: [{}]'.format(epoch)

    # model2的训练
    model2.train()
    model2_metric_logger = utils.MetricLogger(delimiter="  ")
    model2_metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header2 = "Model2" + ' Epoch: [{}]'.format(epoch)

    # source域
    source_val_logger = utils.MetricLogger(delimiter="  ")
    header3 = "Source" + ' Epoch: [{}]'.format(epoch)

    train_cnt = 10
    # 协作训练
    for pack1, pack2, pack3 in zip(model1_metric_logger.log_every(model1_train_loader, print_freq, header1),
                                   model2_metric_logger.log_every(model2_train_loader, print_freq, header2),
                                   source_val_logger.log_every(source_val_loader, print_freq, header3)):
        source_image = pack3[0]
        source_image = source_image.to(device)
        train_cnt = train_cnt + 10

        # 模型1的训练
        image1 = pack1[0]
        target1 = pack1[1]
        image1, target1 = image1.to(device), target1.to(device)
        with torch.cuda.amp.autocast(enabled=model1_run_sets[2] is not None):
            output_i_in_i = model1(image1)
            output_i_in_k = model2(image1)
            output_i_in_s = model2(source_image)

            source_predict11 = model1(source_image)
            source_predict12 = model2(source_image)
            argmax_1 = soft_argmax(source_predict11['out'], source_predict12['out'])

            loss_source = criterion(output_i_in_s, argmax_1)
            loss_self = criterion(output_i_in_i, target1)
            loss_across = criterion_kl(output_i_in_i['out'], output_i_in_k['out'])
            loss_self = loss_self + 0.5 * loss_across + 0.1 * train_cnt / 1000 * loss_source

        model1_run_sets[0].zero_grad()
        if model1_run_sets[2] is not None:
            model1_run_sets[2].scale(loss_self).backward()
            model1_run_sets[2].step(model1_run_sets[0])
            model1_run_sets[2].update()
        else:
            loss_self.backward()
            model1_run_sets[0].step()

        model1_run_sets[1].step()

        lr = model1_run_sets[0].param_groups[0]["lr"]
        model1_metric_logger.update(loss=loss_self.item(), lr=lr)

        # 模型2的训练
        image2 = pack2[0]
        target2 = pack2[1]
        image2, target2 = image2.to(device), target2.to(device)
        with torch.cuda.amp.autocast(enabled=model2_run_sets[2] is not None):
            output_i_in_i_2 = model2(image2)
            output_i_in_k_2 = model1(image2)
            output_i_in_s_2 = model2(source_image)

            source_predict21 = model1(source_image)
            source_predict22 = model2(source_image)
            argmax_2 = soft_argmax(source_predict21['out'], source_predict22['out'])

            loss_source_2 = criterion(output_i_in_s_2, argmax_2)
            loss_self_2 = criterion(output_i_in_i_2, target2)
            loss_across_2 = criterion_kl(output_i_in_i_2['out'], output_i_in_k_2['out'])
            loss_self_2 = loss_self_2 + 0.5 * loss_across_2 + 0.1 * train_cnt / 1000 * loss_source_2

        model2_run_sets[0].zero_grad()
        if model2_run_sets[2] is not None:
            model2_run_sets[2].scale(loss_self_2).backward()
            model2_run_sets[2].step(model2_run_sets[0])
            model2_run_sets[2].update()
        else:
            loss_self_2.backward()
            model2_run_sets[0].step()

        model2_run_sets[1].step()

        lr_2 = model2_run_sets[0].param_groups[0]["lr"]
        model2_metric_logger.update(loss=loss_self_2.item(), lr=lr_2)

    return model1_metric_logger.meters["loss"].global_avg, lr, model2_metric_logger.meters["loss"].global_avg, lr_2
```

KL散度计算相关代码

```
def criterion_kl(inputs, target):
    logp_x = nn.functional.log_softmax(inputs, dim=-1)
    p_y = nn.functional.softmax(target, dim=-1)
    kl_sum = nn.functional.kl_div(logp_x, p_y, reduction='sum') / 480
    return kl_sum
```

### 2.7 复现流程4-在未标记的目标域上的协作学习复现

这一部分所添加的损失同样是在单轮训练中，其代码在2.6中已经进行了展示，这里不额外展示。

在该部分需要进行伪标签生成，其相关代码如下所示。

```
def soft_argmax(input1, input2):
    print(input1.shape, input2.shape)
    soft = input1 + input2
    soft = nn.functional.softmax(soft/2, dim=-1)
    argmax = torch.argmax(soft, dim=0)
    return argmax
```

以上为本项目的复现步骤，至此基本实现了两源域加目标域的协作学习。

## 3 mindspore实现

复现过程主要是在pytorch进行的复现，mindspore的实现主要是借用MSAdapter进行硬件环境的迁移。

MSAdapter为匹配用户习惯，其设计目的是在用户不感知的情况下，能适配PyTorch代码运行在昇腾（Ascend）设备上。MSAdapter以PyTorch的接口为标准，为用户提供一套和PyTorch一样（接口和功能完全一致）的中高阶模型构建接口和数据处理接口。

在迁移过程中主要是对导入包进行了修改，代码内部逻辑并未进行修改，因此不做额外展示。

修改示例，将

```
import torch
```

改为

```
import msadapter.pytorch as torch
```

## 4 结果说明

整个实验训练过程主要是在pytorch环境下进行的，但受限于论文中部分参数不明，在复现过程中只能根据自己的感觉进行设置，因此实验过程中出现kl散度计算值不合理、目标函数损失值不合理、图像风格空间统一效果不理想等多种问题，因此最终实验的效果不理想。

- 基于LAB色彩格式的图像空间统一化，在论文提及公式下，部分处理效果并不好

良好情况

![image-20240614031552712](/home/kolo/.config/Typora/typora-user-images/image-20240614031552712.png)

![image-20240614031532660](/home/kolo/.config/Typora/typora-user-images/image-20240614031532660.png)

不佳情况

![image-20240614031611851](/home/kolo/.config/Typora/typora-user-images/image-20240614031611851.png)

![image-20240614031635445](/home/kolo/.config/Typora/typora-user-images/image-20240614031635445.png)

- 单源域训练过程：在单个源域，不涉及协作学习的情况下，训练能够有序的进行，训练执行和训练信息如下

```
python train_new.py --num-classes 21
```

```
[epoch: 0]
train_loss: 4.9028
lr: 0.000100
global correct: 76.3
average row correct: ['46.5', '68.9', '0.0', '0.0', '0.6', '0.0', '0.0', '94.0', '57.6', '0.2', '0.0', '96.6', '9.0', '0.6', '0.0', '0.0', 'nan', '0.0', '0.0', '0.0', '0.0']
IoU: ['0.8', '50.8', '0.0', '0.0', '0.6', '0.0', '0.0', '80.9', '40.7', '0.1', '0.0', '78.1', '4.6', '0.6', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
mean IoU: 12.2

[epoch: 1]
train_loss: 1.2218
lr: 0.000097
global correct: 80.2
average row correct: ['41.5', '84.9', '2.0', '0.1', '1.6', '0.0', '0.0', '95.8', '77.8', '0.6', '0.3', '98.4', '9.5', '4.4', '0.0', '0.0', 'nan', '5.8', '0.0', '0.3', '0.3']
IoU: ['2.4', '62.8', '1.9', '0.1', '1.4', '0.0', '0.0', '86.2', '50.8', '0.4', '0.2', '78.2', '6.2', '3.9', '0.0', '0.0', '0.0', '5.3', '0.0', '0.2', '0.3']
mean IoU: 14.3

[epoch: 2]
train_loss: 0.9692
lr: 0.000094
global correct: 81.5
average row correct: ['24.2', '86.0', '12.6', '0.1', '6.6', '0.0', '0.0', '97.2', '73.0', '0.2', '0.2', '98.5', '8.9', '11.0', '0.0', '17.0', 'nan', '29.0', '0.0', '0.2', '9.7']
IoU: ['3.7', '65.2', '11.6', '0.1', '5.5', '0.0', '0.0', '86.7', '53.1', '0.2', '0.1', '79.5', '7.2', '9.3', '0.0', '12.4', '0.0', '20.3', '0.0', '0.2', '9.2']
mean IoU: 17.4

[epoch: 3]
train_loss: 0.8726
lr: 0.000091
global correct: 82.4
average row correct: ['11.6', '84.0', '34.4', '0.3', '6.3', '0.0', '1.3', '96.3', '81.2', '2.0', '0.1', '98.0', '10.5', '26.8', '0.0', '26.3', 'nan', '34.7', '0.0', '0.2', '24.2']
IoU: ['1.9', '68.7', '28.4', '0.3', '5.5', '0.0', '1.2', '87.5', '55.4', '1.9', '0.1', '80.7', '8.4', '18.0', '0.0', '21.8', '0.0', '24.0', '0.0', '0.2', '20.9']
mean IoU: 20.2
```

- 在多源域协作学习并加入目标域协作学习下训练效果并不好，meanIoU数值并未有明显的上升，而且甚至会导致Nan的情况（可能是训练轮次不够）

```
[epoch: 0]
train_loss: 6.2526
lr: 0.000100
global correct: 0.7
average row correct: ['0.0', '0.0', '0.0', '13.8', '0.0', '2.8', '1.0', '0.0', '0.5', '0.1', '0.0', '0.1', '22.3', '6.6', '0.0', '0.0', 'nan', '0.0', '0.0', '0.6', '0.0', 'nan']
IoU: ['0.0', '0.0', '0.0', '1.5', '0.0', '0.3', '0.9', '0.0', '0.4', '0.1', '0.0', '0.1', '1.1', '0.7', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
mean IoU: 0.2

[epoch: 1]
train_loss: 5.5274
lr: 0.000097
global correct: 0.8
average row correct: ['0.0', '0.0', '0.0', '22.1', '0.0', '1.8', '0.4', '0.0', '0.7', '0.1', '0.3', '0.0', '13.6', '12.1', '0.0', '0.0', 'nan', '0.0', '0.0', '0.7', '0.0', 'nan']
IoU: ['0.0', '0.0', '0.0', '1.7', '0.0', '0.2', '0.3', '0.0', '0.6', '0.1', '0.1', '0.0', '0.9', '1.1', '0.0', '0.0', '0.0', '0.0', '0.0', '0.1', '0.0', '0.0']
mean IoU: 0.2

[epoch: 2]
train_loss: 5.4295
lr: 0.000094
global correct: 0.6
average row correct: ['0.0', '0.0', '0.0', '8.1', '0.0', '7.6', '0.5', '0.1', '0.2', '0.1', '0.3', '0.0', '17.1', '12.4', '0.0', '0.0', 'nan', '0.0', '0.0', '0.6', '0.0', 'nan']
IoU: ['0.0', '0.0', '0.0', '1.2', '0.0', '0.2', '0.4', '0.1', '0.2', '0.1', '0.0', '0.0', '1.0', '0.9', '0.0', '0.0', '0.0', '0.0', '0.0', '0.1', '0.0', '0.0']
mean IoU: 0.2

[epoch: 3]
train_loss: 5.3779
lr: 0.000091
global correct: 0.5
average row correct: ['0.0', '0.0', '0.0', '9.8', '0.0', '8.4', '0.4', '0.0', '0.6', '0.0', '0.0', '0.1', '13.8', '4.6', '0.0', '0.1', 'nan', '0.0', '0.0', '0.4', '0.0', 'nan']
IoU: ['0.0', '0.0', '0.0', '1.3', '0.0', '0.2', '0.3', '0.0', '0.5', '0.0', '0.0', '0.1', '0.8', '0.6', '0.0', '0.0', '0.0', '0.0', '0.0', '0.1', '0.0', '0.0']
mean IoU: 0.2
```
