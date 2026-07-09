import os
import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity


# ----------------------------------------
#    训练过程中的验证与采样
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255, save_format = 'png'):
    # 逐个保存图像
    for i in range(len(img_list)):
        img = img_list[i]
        # 恢复归一化：乘以 255，因为最后一层是 sigmoid 激活
        img = img * 255
        # 处理 img_copy，不破坏原始 img 数据
        img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        # 保存到指定路径
        save_img_name = sample_name + '_' + name_list[i] + '.' + save_format
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)


def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p


def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p


def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = structural_similarity(target, pred, channel_axis=2, data_range=1.0)
    return ssim


# ----------------------------------------
#             路径处理
# ----------------------------------------
def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)


def get_files(path):
    # 读取文件夹，返回完整路径
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret


def get_jpgs(path):
    # 读取文件夹，返回图像文件名
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret


def get_jpgs_once(paths):
    # 读取文件夹，返回去重后的图像文件名
    ret = set([])
    if isinstance(paths, str):
        paths = [paths]
    assert isinstance(paths, list)
    for path in paths:
        for root, dirs, files in os.walk(path):
            for filespath in files:
                filespath = filespath.split('_')[0] + '_' + filespath.split('_')[1]
                filespath = os.path.join(root, filespath)
                if filespath not in ret:
                    ret.add(filespath)
    ret = list(ret)
    return ret


def get_blur_file_once(paths):
    # 读取模糊图像文件夹，返回去重后的文件名
    ret = set([])
    if isinstance(paths, str):
        paths = [paths]
    assert isinstance(paths, list)
    for path in paths:
        for root, dirs, files in os.walk(path):
            for filespath in files:
                filespath = filespath.split('_')[0] + '_' + filespath.split('_')[1] + '_' + filespath.split('_')[2]
                filespath = os.path.join(root, filespath)
                if filespath not in ret:
                    ret.add(filespath)
    ret = list(ret)
    return ret


def text_readlines(filename):
    # 尝试读取 txt 文件并返回列表，出错时返回空列表
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # 循环删除每行末尾的换行符等 EOF 字符
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content


def text_save(content, filename, mode = 'a'):
    # 将列表保存到 txt 文件
    # 尝试将列表变量写入 txt 文件
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_file(path):
    return os.path.exists(path)


def normalize_ImageNet_stats(batch):
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch_out = (batch - mean) / std
    return batch_out
