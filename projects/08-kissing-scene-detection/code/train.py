# 改编自PyTorch教程
import copy
import time
from typing import List, Tuple, Optional

import torch
import torch.optim as optim
from torch import nn

from data import AudioVideo, AudioVideo3D
from kissing_detector import KissingDetector, KissingDetector3DConv

ExperimentResults = Tuple[Optional[nn.Module], List[float], List[float]]


def _get_params_to_update(model: nn.Module,
                          feature_extract: bool) -> List[nn.parameter.Parameter]:
    params_to_update = model.parameters()
    if feature_extract:
        print('Params to update')
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                print("*", name)
    else:
        print('Updating ALL params')
    return params_to_update


def train_kd(data_path_base: str,
             conv_model_name: Optional[str],
             num_epochs: int,
             feature_extract: bool,
             batch_size: int,
             use_vggish: bool = True,
             num_workers: int = 4,
             shuffle: bool = True,
             lr: float = 0.001,
             momentum: float = 0.9,
             use_3d: bool = False) -> ExperimentResults:
    num_classes = 2
    try:
        if use_3d:
            kd = KissingDetector3DConv(num_classes, feature_extract, use_vggish)
        else:
            kd = KissingDetector(conv_model_name, num_classes, feature_extract, use_vggish=use_vggish)
    except ValueError:
        # if the combination is not valid
        return None, [-1.0], [-1.0]

    params_to_update = _get_params_to_update(kd, feature_extract)

    av = AudioVideo3D if use_3d else AudioVideo

    datasets = {set_: av(f'{data_path_base}/{set_}') for set_ in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(datasets[x],
                                                       batch_size=batch_size,
                                                       shuffle=shuffle, num_workers=num_workers)
                        for x in ['train', 'val']}
    # optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=momentum) # 使用SGD优化器
    optimizer_ft = optim.Adam(params_to_update, lr=lr)

    # 设置损失函数
    criterion = nn.CrossEntropyLoss()

    return train_model(kd,
                       dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                       is_inception=(conv_model_name == "inception"))


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    val_f1_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0

    # 检测是否有可用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch包含训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_tp = 0
            running_fp = 0
            running_fn = 0

            # 遍历数据
            for a, v, labels in dataloaders[phase]:
                a = a.to(device)
                v = v.to(device)
                labels = labels.to(device)

                # 清零参数梯度
                optimizer.zero_grad()

                # 前向传播
                # 仅在训练时跟踪历史记录
                with torch.set_grad_enabled(phase == 'train'):
                    # 获取模型输出并计算损失
                    # Inception模型的特殊情况，因为在训练时它有一个辅助输出。在训练
                    #   模式下，我们通过将最终输出和辅助输出相加来计算损失
                    #   但在测试时，我们只考虑最终输出
                    if is_inception and phase == 'train':
                        # https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(a, v)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(a, v)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # 仅在训练阶段进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计信息
                running_loss += loss.item() * a.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_tp += torch.sum((preds == labels.data)[labels.data == 1])
                running_fp += torch.sum((preds != labels.data)[labels.data == 1])
                running_fn += torch.sum((preds != labels.data)[labels.data == 0])

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            n = len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / n
            tp = running_tp.double()
            fp = running_fp.double()
            fn = running_fn.double()
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            epoch_f1 = 2 * p * r / (p + r)

            print('{} Loss: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_f1, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(float(epoch_acc))
                val_f1_history.append(float(epoch_f1))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val F1  : {:4f}'.format(best_f1))
    print('Best val Acc : {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, val_f1_history
