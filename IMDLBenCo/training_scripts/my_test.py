import torch
import sys

sys.path.append(".")
import torch.nn.functional as F
from diffusers import DDIMScheduler
from torchvision import transforms
from PIL import Image
from IMDLBenCo.model_zoo import DiffIML


def image_to_tensor(image_path):
    # 定义转换器
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 读取图像
    image = Image.open(image_path).convert('RGB')

    # 转换为Tensor
    tensor = transform(image)

    return tensor


def f1_score(y_true, y_pred, threshold=0.5):
    # 将预测值转换为二值（1 或 0）
    y_pred = (y_pred > threshold).float()

    # 计算TP、FP、FN
    tp = (y_true * y_pred).sum().float()
    fp = ((1 - y_true) * y_pred).sum().float()
    fn = (y_true * (1 - y_pred)).sum().float()

    # 计算 Precision 和 Recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    # 计算F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return f1.item()


def make_list():
    path_list = [
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/CASIAv1/CASIAv1_image_1.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/CASIAv1/CASIAv1_mask_1.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/CASIAv1/CASIAv1_image_2.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/CASIAv1/CASIAv1_mask_2.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/CASIAv1/CASIAv1_image_3.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/CASIAv1/CASIAv1_mask_3.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/Columbia/Columbia_image_1.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/Columbia/Columbia_mask_1.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/Columbia/Columbia_image_2.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/Columbia/Columbia_mask_2.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/Columbia/Columbia_image_3.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/Columbia/Columbia_mask_3.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/coverage/coverage_image_1.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/coverage/coverage_mask_1.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/coverage/coverage_image_2.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/coverage/coverage_mask_2.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/coverage/coverage_image_3.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/coverage/coverage_mask_3.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/IMD20_1024/IMD20_1024_image_1.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/IMD20_1024/IMD20_1024_mask_1.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/IMD20_1024/IMD20_1024_image_2.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/IMD20_1024/IMD20_1024_mask_2.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/IMD20_1024/IMD20_1024_image_3.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/IMD20_1024/IMD20_1024_mask_3.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/NIST16_1024/NIST16_1024_image_1.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/NIST16_1024/NIST16_1024_mask_1.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/NIST16_1024/NIST16_1024_image_2.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/NIST16_1024/NIST16_1024_mask_2.png'
        },
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/NIST16_1024/NIST16_1024_image_3.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/NIST16_1024/NIST16_1024_mask_3.png'
        }
    ]
    path_list = [
        {
            'image_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/CASIAv1/CASIAv1_image_1.png',
            'mask_path': '/mnt/data0/dubo/workspace/IMDLBenCo/data/com_img/CASIAv1/CASIAv1_mask_1.png'
        }
    ]
    tensor_list = []
    for i in path_list:
        item = {'image': image_to_tensor(i['image_path']), 'mask': mask_to_tensor(i['mask_path'])}
        tensor_list.append(item)
    return tensor_list

def mask_to_tensor(mask_path):
    # 定义转换器，调整大小为 512x512 并转为单通道 Tensor
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 调整Mask大小为 512x512
        transforms.ToTensor(),  # 将灰度图像转换为 Tensor
    ])

    # 读取mask图像（灰度图）
    mask = Image.open(mask_path).convert('L')  # 'L' 模式表示灰度图

    # 转换为Tensor
    tensor = transform(mask)

    return tensor


if __name__ == '__main__':
    ckpt_path = '/mnt/data0/dubo/workspace/IMDLBenCo/log/ckpt_dir/checkpoint-36.pth'
    device = 'cuda:4'
    # device = 'cpu'
    step = 13
    model = DiffIML(num_inference_steps=step, infer_time=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
    print(f"Load {ckpt_path} weight success.")
    model.eval()


    list = make_list()
    print(len(list))
    f1_list = []
    for item in list:
        image = item['image'].to(device).unsqueeze(0)
        mask = item['mask'].to(device).unsqueeze(0)
        pred_mask = model(image=image, mask=mask)['pred_mask']
        f1_list.append(f1_score(mask, pred_mask))
    print(sum(f1_list) / len(f1_list))
