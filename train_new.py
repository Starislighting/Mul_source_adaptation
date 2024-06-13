import os
import time
import datetime

import torch

from src import deeplabv3_resnet101
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, train_one_epoch_double_run
from datasets_set.VOC_dataset import VOCSegmentation
from datasets_set.cityscapes_datasets import CityscapesSegmentation
from datasets_set.GTA5_datasets import GTA5Segmentation
import transforms as T


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))

        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    base_size = 520
    crop_size = 480

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)


def create_model(aux, num_classes, pretrain=True):
    model = deeplabv3_resnet101(aux=aux, num_classes=num_classes)

    if pretrain:
        my_path = '/home/kolo/PycharmProjects/weights/deeplabv3/deeplabv3_resnet101_coco-586e9e4e.pth'
        weights_dict = torch.load(my_path, map_location='cpu')

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model

def dataset_load(category, data_path, batch_size):
    train_loader = None
    val_loader = None
    if category == 'VOC':
        # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
        VOC_train_dataset = VOCSegmentation(data_path,
                                            year="2012",
                                            transforms=get_transform(train=True),
                                            txt_name="train.txt")

        # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
        VOC_val_dataset = VOCSegmentation(data_path,
                                          year="2012",
                                          transforms=get_transform(train=False),
                                          txt_name="val.txt")

        num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        train_loader = torch.utils.data.DataLoader(VOC_train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   collate_fn=VOC_train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(VOC_val_dataset,
                                                 batch_size=1,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 collate_fn=VOC_val_dataset.collate_fn)

    elif category == 'GTA5':
        GTA5_train_dataset = GTA5Segmentation(data_path, transforms=get_transform(train=True),
                                              txt_name="train.txt")

        GTA5_val_dataset = GTA5Segmentation(data_path, transforms=get_transform(train=False),
                                            txt_name="val.txt")

        num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        train_loader = torch.utils.data.DataLoader(GTA5_train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   collate_fn=GTA5_train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(GTA5_val_dataset,
                                                 batch_size=1,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 collate_fn=GTA5_val_dataset.collate_fn)
    elif category == 'cityscapes':
        city_train_dataset = CityscapesSegmentation(data_path, 'train',
                                                    transforms=get_transform(train=True))

        city_val_dataset = CityscapesSegmentation(data_path, 'val',
                                                    transforms=get_transform(train=False))

        num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        train_loader = torch.utils.data.DataLoader(city_train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   collate_fn=city_train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(city_val_dataset,
                                                 batch_size=2,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 collate_fn=city_val_dataset.collate_fn)
    else:
        pass

    return train_loader, val_loader

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_loader, val_loader = dataset_load(args.datatypes, args.data_path, batch_size)

    model = create_model(aux=args.aux, num_classes=num_classes)
    model.to(device)

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    # import matplotlib.pyplot as plt
    # lr_list = []
    # for _ in range(args.epochs):
    #     for _ in range(len(train_loader)):
    #         lr_scheduler.step()
    #         lr = optimizer.param_groups[0]["lr"]
    #         lr_list.append(lr)
    # plt.plot(range(len(lr_list)), lr_list)
    # plt.show()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def model_inits(model, train_loader):
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # import matplotlib.pyplot as plt
    # lr_list = []
    # for _ in range(args.epochs):
    #     for _ in range(len(train_loader)):
    #         lr_scheduler.step()
    #         lr = optimizer.param_groups[0]["lr"]
    #         lr_list.append(lr)
    # plt.plot(range(len(lr_list)), lr_list)
    # plt.show()

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    return [optimizer, lr_scheduler, scaler]

# 训练结果的写入
def model_result_save(model_results_file, model, model_run_sets, epoch, mean_loss, val_info, lr, model_name):
    # 将训练信息写入txt
    with open(model_results_file, "a") as f:
        # 记录每个epoch对应的train_loss、lr以及验证集各指标
        train_info = f"[epoch: {epoch}]\n" \
                     f"train_loss: {mean_loss:.4f}\n" \
                     f"lr: {lr:.6f}\n"
        f.write(train_info + val_info + "\n\n")

    save_file = {"model": model.state_dict(),
                 "optimizer": model_run_sets[0].state_dict(),
                 "lr_scheduler": model_run_sets[1].state_dict(),
                 "epoch": epoch,
                 "args": args}
    if args.amp:
        save_file["scaler"] = model_run_sets[2].state_dict()
    torch.save(save_file, "save_weights/model_{}".format(epoch) + model_name + ".pth")

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


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 training")

    parser.add_argument("--data-path", default="/data/", help="VOCdevkit root")
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--datatypes", default="None", type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main_double_model(args)
    # main(args)