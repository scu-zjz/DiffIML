import os
import sys
import torch.nn.functional as F
import json
from diffusers import DDIMScheduler
from PIL import Image

from tqdm import tqdm
import numpy as np

sys.path.append(".")
import torch
from IMDLBenCo.registry import MODELS, POSTFUNCS
from IMDLBenCo.model_zoo import DiffIML, MantraNet, Cat_Net, MVSSNet, PSCC_Net, Trufor, ObjectFormer
from IMDLBenCo.datasets import ManiDataset, JsonDataset, BalancedDataset
from IMDLBenCo.transforms import get_albu_transforms


def f1_score(y_true, y_pred, epsilon=1e-7):
    """
    计算 F1 分数
    Args:
        y_true (torch.Tensor): 真实标签, 形状 (N, C) 或 (N,)，其中 N 是样本数，C 是类别数
        y_pred (torch.Tensor): 预测标签, 形状 (N, C) 或 (N,)，其中 N 是样本数，C 是类别数
        epsilon (float): 防止除以零的一个小常数

    Returns:
        float: F1 分数
    """
    if y_true.dim() == 2:
        y_true = y_true.argmax(dim=1)
    if y_pred.dim() == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1.item()


def load_weight(model, ckpt):
    dict = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(dict['model'])
    return model


def get_models_and_load_weights(ckpt, device):
    models = {}
    sizes = {}
    models['DiffIML'] = DiffIML().to(device).eval()
    sizes['DiffIML'] = 512
    models['MantraNet'] = MantraNet().to(device).eval()
    sizes['MantraNet'] = 512
    models['Cat_Net'] = Cat_Net().to(device).eval()
    sizes['Cat_Net'] = 512
    models['PSCC_Net'] = PSCC_Net().to(device).eval()
    sizes['PSCC_Net'] = 256
    models['MVSSNet'] = MVSSNet().to(device)
    sizes['MVSSNet'] = 512
    models['Trufor'] = Trufor().to(device).eval()
    sizes['Trufor'] = 512
    models['ObjectFormer'] = ObjectFormer().to(device).eval()
    sizes['ObjectFormer'] = 224

    for model_name in models.keys():
        if ckpt[model_name] != "":
            load_weight(models[model_name], ckpt[model_name])
            print(f'Load {model_name} success.')

    return models, sizes


def operate(mask):
    return torch.where(mask > 0.5, 1., 0.)


def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """denormalize image with mean and std
    """
    image = image.cpu()
    image = image * torch.tensor(std).view(1, -1, 1, 1)
    image = image + torch.tensor(mean).view(1, -1, 1, 1)
    return image


def save_mask_to_png(mask_tensor, file_path):
    # 将 PyTorch Tensor 转换为 NumPy 数组
    if mask_tensor.shape[0] == 1:
        mask_array = mask_tensor.squeeze(0).cpu().numpy()  # 如果 mask_tensor 在 GPU 上，需要先转移到 CPU
        mask_array = (mask_array * 255).astype(np.uint8)
    else:
        mask_tensor = mask_tensor.permute(1, 2, 0)
        mask_array = (mask_tensor * 255).cpu().numpy().astype(np.uint8)

    # 创建 PIL 图像对象
    mask_image = Image.fromarray(mask_array)

    # 保存为 PNG 文件
    mask_image.save(file_path)


def run():
    dataset_paths = {
        "Columbia": "/mnt/data0/public_datasets/IML/Columbia.json",
        "NIST16_1024": "/mnt/data0/public_datasets/IML/NIST16_1024",
        "coverage": "/mnt/data0/public_datasets/IML/coverage.json",
        "CASIAv1": "/mnt/data0/public_datasets/IML/CASIA1.0",
        "IMD20_1024": "/mnt/data0/public_datasets/IML/IMD_20_1024"
    }
    ckpt_path = '/mnt/data0/yunfei/workspace/IMDLBenCo/data/find_pred.json'
    with open(ckpt_path, 'r') as f:
        ckpt = json.load(f)
    device = 'cuda:0'
    models, sizes = get_models_and_load_weights(ckpt, device)
    test_transform = get_albu_transforms('test')
    batch_size = 8
    find_count = 3
    lower_threshold = 0.7
    upper_threshold = 0.9
    save_img_dir = '/mnt/data0/yunfei/workspace/IMDLBenCo/data/com_img'

    for dataset_name in dataset_paths.keys():
        print(f'Starting searching {dataset_name}.')
        os.mkdir(os.path.join(save_img_dir, dataset_name))
        dataset_path = dataset_paths[dataset_name]
        # ---- dataset with crop augmentation ----
        post_function_name = f"cat_net_post_func".lower()
        print(f"Post function check: {post_function_name}")
        post_function = POSTFUNCS.get(post_function_name)
        if os.path.isdir(dataset_path):
            dataset_test = ManiDataset(
                dataset_path,
                is_resizing=True,
                output_size=(512, 512),
                common_transforms=test_transform,
                edge_width=7,
                post_funcs=post_function
            )
        else:
            dataset_test = JsonDataset(
                dataset_path,
                is_resizing=True,
                output_size=(512, 512),
                common_transforms=test_transform,
                edge_width=7,
                post_funcs=post_function
            )
        import torch.utils.data
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size,
            num_workers=8,
            shuffle=True
        )

        cur_count = 0
        for data_iter_step, data_dict in tqdm(enumerate(data_loader_test)):
            image, mask = data_dict['image'].to(device).float(), data_dict['mask'].to(device).float()

            tot_pred_mask = {}
            with torch.no_grad():
                for model_name in models.keys():
                    # move to device
                    for key in data_dict.keys():
                        if isinstance(data_dict[key], torch.Tensor):
                            data_dict[key] = data_dict[key].to(device)
                    data_dict['image'] = F.interpolate(data_dict['image'], (sizes[model_name], sizes[model_name])).to(
                        device)
                    data_dict['edge_mask'] = F.interpolate(data_dict['edge_mask'],
                                                           (sizes[model_name], sizes[model_name])).to(device)
                    data_dict['mask'] = F.interpolate(data_dict['mask'], (sizes[model_name], sizes[model_name])).to(
                        device)
                    pred_mask = models[model_name](**data_dict)['pred_mask']
                    pred_mask = F.interpolate(pred_mask, (512, 512)).to(device)
                    tot_pred_mask[model_name] = []
                    for idx in range(0, image.shape[0]):
                        tot_pred_mask[model_name].append(pred_mask[idx])

            image = denormalize(image)
            for idx in range(0, image.shape[0]):
                for j, model_name in enumerate(tot_pred_mask.keys()):
                    if model_name == 'DiffIML' and f1_score(mask[idx],
                                                            operate(tot_pred_mask[model_name][idx])) < upper_threshold:
                        break
                    if model_name != 'DiffIML' and f1_score(mask[idx],
                                                            operate(tot_pred_mask[model_name][idx])) > lower_threshold:
                        break
                    if j == len(tot_pred_mask) - 1:
                        cur_count += 1
                        image_path = os.path.join(save_img_dir, dataset_name, f'{dataset_name}_mask_{cur_count}.png')
                        save_mask_to_png(mask[idx], image_path)
                        image_path = os.path.join(save_img_dir, dataset_name, f'{dataset_name}_image_{cur_count}.png')
                        save_mask_to_png(image[idx], image_path)
                        for name in tot_pred_mask.keys():
                            image_path = os.path.join(save_img_dir, dataset_name,
                                                      f'{dataset_name}_{name}_{cur_count}.png')
                            if name == 'DiffIML':
                                save_mask_to_png(operate(tot_pred_mask[name][idx]), image_path)
                            else:
                                save_mask_to_png(tot_pred_mask[name][idx], image_path)
                    if cur_count >= find_count:
                        break
                if cur_count >= find_count:
                    break

            if cur_count >= find_count:
                break


if __name__ == '__main__':
    run()
