import os
import sys
import torch.nn.functional as F
import json
from diffusers import DDIMScheduler
from tqdm import tqdm
import torch.utils.data

sys.path.append(".")
import torch
from IMDLBenCo.model_zoo import DiffIML
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


def run(step):
    dataset_path = '/mnt/data0/public_datasets/IML/CASIA1.0'
    ckpt_path = '/mnt/data0/yunfei/workspace/IMDLBenCo/log/ckpt_dir/checkpoint-36.pth'
    device = 'cuda:0'
    test_transform = get_albu_transforms('test')

    # ---- dataset with crop augmentation ----
    if os.path.isdir(dataset_path):
        dataset_test = ManiDataset(
            dataset_path,
            is_resizing=True,
            output_size=(512, 512),
            common_transforms=test_transform,
        )

    else:
        dataset_test = JsonDataset(
            dataset_path,
            is_resizing=True,
            output_size=(512, 512),
            common_transforms=test_transform,
        )
    # ------------------------------------
    print(dataset_test)
    print(f"Test dataset total: {len(dataset_test)}.")

    model = DiffIML(num_inference_steps=step).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
    print(f"Load {ckpt_path} weight success.")
    model.eval()

    batch_size = 20
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=8
    )
    print(f"Totol {len(dataset_test) / batch_size} batch.")

    map_path = f'/mnt/data0/dubo/workspace/IMDLBenCo/data/eb_data_{step}_step.json'
    map = {"step": step, "count": 0, "train_l1": {}, "test_l1": {}}
    scheduler = DDIMScheduler(prediction_type='sample')
    scheduler.set_timesteps(model.num_inference_steps)
    for data_iter_step, data_dict in tqdm(enumerate(data_loader_test)):
        image, mask = data_dict['image'].to(device).float(), data_dict['mask'].to(device).float()
        print(data_dict['label'].shape)
        with torch.no_grad():
            features = model.extractor(image)
            latent_mask = model.vae.encode_mask(mask)
            test_y_t = torch.randn(image.shape[0], 4, 64, 64).to(device)
            train_y_t = test_y_t.clone()
            timesteps = scheduler.timesteps
            iterable = enumerate(timesteps)
            for i, t in iterable:
                test_noise_pred = model.unet(test_y_t, t, features)[:, 0:4, ...]
                train_noise_pred = model.unet(
                    scheduler.add_noise(latent_mask, torch.randn_like(latent_mask, device=device), timesteps=t), t,
                    features)[:, 0:4, ...]

                test_y_t = scheduler.step(test_noise_pred, t, test_y_t).prev_sample
                train_y_t = scheduler.step(latent_mask, t, train_y_t).prev_sample

                for idx in range(0, image.shape[0]):
                    test_l1 = F.mse_loss(test_noise_pred[idx], latent_mask[idx]).item()
                    train_l1 = F.mse_loss(train_noise_pred[idx], latent_mask[idx]).item()
                    if i == 0:
                        map['count'] += 1
                    if map['train_l1'].get(t.item()) is None:
                        map['train_l1'][t.item()] = train_l1
                    else:
                        map['train_l1'][t.item()] += train_l1
                    if map['test_l1'].get(t.item()) is None:
                        map['test_l1'][t.item()] = test_l1
                    else:
                        map['test_l1'][t.item()] += test_l1

    print(f"Test dataset total: {map['count']}.")
    for key in map['train_l1']:
        map['train_l1'][key] /= map['count']
    for key in map['test_l1']:
        map['test_l1'][key] /= map['count']

    with open(map_path, 'w') as file:
        json.dump(map, file, indent=4)
    print('Create json file success.')


if __name__ == '__main__':
    for step in [5, 10, 20]:
        print(step)
        run(step)
