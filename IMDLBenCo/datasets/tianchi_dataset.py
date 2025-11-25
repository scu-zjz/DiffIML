import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import ast


class TamperedImageDataset(Dataset):
    def __init__(self, csv_file, img_size=(512, 512), transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with image paths and polygons.
            img_size (tuple): The size to which the images and masks will be resized.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.df = pd.read_csv(csv_file)
        self.img_size = img_size
        self.transform = transform
        self.abs_path = "/mnt/data0/public_datasets/tianTi/train/"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. 获取图片路径和多边形坐标
        img_path = self.df.iloc[idx]['Path']
        polygon_str = self.df.iloc[idx]['Polygons']

        # 2. 加载图像
        img_path = self.abs_path + img_path
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # 获取原始图像尺寸

        # 3. 解析多边形坐标
        polygons = ast.literal_eval(polygon_str)

        # 4. 创建空的掩码 (初始尺寸与图像相同)
        mask = Image.new('L', original_size, 0)
        draw = ImageDraw.Draw(mask)

        # 5. 绘制每个多边形到掩码上
        for polygon in polygons:
            polygon = [tuple(point) for point in polygon]
            draw.polygon(polygon, outline=1, fill=1)  # 绘制多边形，内部填充

        # 6. 调整图像和掩码的尺寸到目标大小
        image = image.resize(self.img_size)
        mask = mask.resize(self.img_size)

        # 7. 转换为 PyTorch 张量
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()  # (C, H, W)
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float()  # (1, H, W)

        # 8. 如果有变换，应用变换
        if self.transform:
            image = self.transform(image)

        return image, mask, polygons, original_size  # 返回图像、生成的掩码以及多边形顶点列表
