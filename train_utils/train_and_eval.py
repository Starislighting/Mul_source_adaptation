import torch
from torch import nn
from torch.nn.functional import log_softmax,softmax,kl_div
import train_utils.distributed_utils as utils

# 交叉熵计算
def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

# kl散度计算
def criterion_kl(inputs, target):
    logp_x = nn.functional.log_softmax(inputs, dim=-1)
    p_y = nn.functional.softmax(target, dim=-1)
    kl_sum = nn.functional.kl_div(logp_x, p_y, reduction='sum') / 480
    return kl_sum

# 伪标签生成
def soft_argmax(input1, input2):
    print(input1.shape, input2.shape)
    soft = input1 + input2
    soft = nn.functional.softmax(soft/2, dim=-1)
    argmax = torch.argmax(soft, dim=0)
    return argmax

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = "Model" + ' Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)


        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


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
    image_list1 = []
    target_list1 = []

    image_list2 = []
    target_list2 = []

    # for image, target in model1_metric_logger.log_every(model1_train_loader, print_freq, header):
    #     image_list1.append(image)
    #     target_list1.append(target)
    #
    # for index in range(len(image_list1)):
    #     image_1 = image_list1[index]
    #     target_1 = target_list1[index]

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


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
