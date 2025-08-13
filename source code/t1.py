import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
import os
import re
import numpy as np
from PIL import Image
import trimesh
import shutil
import random
import torchvision
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import traceback
import math
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练VGG用于特征损失
vgg_feature_extractor = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
for param in vgg_feature_extractor.parameters():
    param.requires_grad = False


def scan_training_data_with_texture(obj_folder, mtl_folder, tga_folder=None):
    """扫描文件夹，自动匹配OBJ、MTL和TGA文件"""
    data_triplets = []

    if not os.path.exists(obj_folder):
        print(f"OBJ folder does not exist: {obj_folder}")
        return data_triplets

    if not os.path.exists(mtl_folder):
        print(f"MTL folder does not exist: {mtl_folder}")
        return data_triplets

    # 获取所有OBJ文件
    obj_files = [f for f in os.listdir(obj_folder) if f.lower().endswith('.obj')]

    for obj_file in obj_files:
        obj_path = os.path.join(obj_folder, obj_file)
        obj_basename = os.path.splitext(obj_file)[0]

        # 生成对应的MTL文件名
        mtl_file = obj_basename + '.mtl'
        mtl_path = os.path.join(mtl_folder, mtl_file)

        # 生成对应的TGA文件名
        tga_path = None
        if tga_folder and os.path.exists(tga_folder):
            tga_file = obj_basename + '.tga'
            potential_tga_path = os.path.join(tga_folder, tga_file)
            if os.path.exists(potential_tga_path):
                tga_path = potential_tga_path

        # 检查文件是否存在
        if os.path.exists(mtl_path):
            data_triplets.append((obj_path, mtl_path, tga_path))
        else:
            print(f"Warning: No corresponding MTL found for {obj_file}")

    print(f"Total training triplets found: {len(data_triplets)}")
    return data_triplets


def extract_mesh_features(verts, faces):
    """增强的网格几何特征提取"""
    try:
        # 基础统计特征
        mean = torch.mean(verts, dim=0)
        min_vals = torch.min(verts, dim=0).values
        max_vals = torch.max(verts, dim=0).values
        std = torch.std(verts, dim=0)
        bbox_size = max_vals - min_vals

        # 顶点分布特征
        centroid = mean
        dist_to_centroid = torch.norm(verts - centroid, dim=1)
        dist_mean = torch.mean(dist_to_centroid)
        dist_std = torch.std(dist_to_centroid)

        # 创建trimesh对象计算高级特征
        verts_np = verts.detach().cpu().numpy()
        faces_np = faces.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)

        # 计算表面积和体积
        area = mesh.area
        volume = mesh.volume

        # 确保面索引形状正确 [F, 3]
        if faces.dim() == 1:
            faces = faces.view(-1, 3)
        if faces.shape[1] != 3:
            faces = faces.reshape(-1, 3)

        # 创建PyTorch3D网格对象
        mesh_torch = Meshes(verts=[verts], faces=[faces])

        # 计算顶点法向量统计特征
        _, normals = sample_points_from_meshes(
            mesh_torch,
            return_normals=True,
            num_samples=min(5000, len(verts))
        )# 修正：移除多余括号
        normals = normals.squeeze(0)
        normals_mean = torch.mean(normals, dim=0)
        normals_std = torch.std(normals, dim=0)
        normals_min = torch.min(normals, dim=0).values
        normals_max = torch.max(normals, dim=0).values

        # 曲率特征
        curvature = torch.norm(normals - normals_mean, dim=1)
        curvature_mean = torch.mean(curvature)
        curvature_std = torch.std(curvature)

        features = torch.cat([
            mean, min_vals, max_vals, std, bbox_size,  # 15维 (5*3)
            torch.tensor([area, volume, dist_mean, dist_std], dtype=torch.float32),  # 4维
            normals_mean, normals_std, normals_min, normals_max,  # 12维 (4*3)
            torch.tensor([curvature_mean, curvature_std], dtype=torch.float32)  # 2维
        ])  # 15+4+12+2=33维

        # 添加特征归一化
        features = (features - features.mean()) / (features.std() + 1e-8)

        return features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return torch.zeros(33)  # 修正为实际维度


def load_obj_mtl_tga_triplet(obj_path, mtl_path=None, tga_folder=None):
    try:
        # 加载OBJ
        verts, faces, aux = load_obj(obj_path)
        faces_tensor = faces.verts_idx

        # 提取增强特征
        input_features = extract_mesh_features(verts, faces_tensor)

    except Exception as e:
        print(f"Error loading OBJ {obj_path}: {e}")
        input_features = torch.zeros(33)
        faces_tensor = None

    # 解析MTL
    target_kd = torch.zeros(3)
    if mtl_path and os.path.exists(mtl_path):
        try:
            with open(mtl_path, 'r', encoding='utf-8') as f:
                mtl_content = f.read()
            kd_match = re.search(r'Kd\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)', mtl_content)
            if kd_match:
                target_kd = torch.tensor([float(x) for x in kd_match.groups()])
        except Exception as e:
            print(f"Error parsing MTL {mtl_path}: {e}")

    # 改进纹理加载
    target_texture = None
    if tga_folder:
        obj_filename = os.path.splitext(os.path.basename(obj_path))[0]
        tga_path = os.path.join(tga_folder, f"{obj_filename}.tga")
        if os.path.exists(tga_path):
            try:
                img = Image.open(tga_path).convert('RGB')

                # 使用高质量下采样
                img = img.resize((256, 256), Image.LANCZOS)

                # 转换为张量并标准化 (0-1范围)
                img_array = np.array(img).astype(np.float32) / 255.0
                target_texture = torch.tensor(img_array).permute(2, 0, 1)

            except Exception as e:
                print(f"Error loading texture {tga_path}: {e}")
                target_texture = torch.zeros(3, 256, 256)
        else:
            print(f"TGA not found: {tga_path}")
            target_texture = torch.zeros(3, 256, 256)
    else:
        target_texture = torch.zeros(3, 256, 256)

    # 检测全黑纹理
    if target_texture.abs().max() < 0.01:
        print(f"WARNING: Zero texture detected for {obj_path}")

    # 检查无效特征
    if torch.isnan(input_features).any() or torch.isinf(input_features).any():
        print(f"ERROR: Invalid features in {obj_path}")
        input_features = torch.zeros_like(input_features)

    return input_features, target_kd, target_texture, ""


class MeshDataset(Dataset):
    def __init__(self, triplets, cache_size=100):
        self.triplets = triplets
        self.cache = {}
        self.cache_size = cache_size

        # 增强的数据增强
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
        self.gaussian_blur = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
        self.random_rotation = transforms.RandomRotation(10)
        self.random_flip = transforms.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        # 检查缓存
        if idx in self.cache:
            return self.cache[idx]

        obj_path, mtl_path, tga_path = self.triplets[idx]
        inputs, target_kd, target_texture, _ = load_obj_mtl_tga_triplet(
            obj_path, mtl_path,
            os.path.dirname(tga_path) if tga_path else None
        )

        # 处理缺失纹理
        if target_texture is None:
            target_texture = torch.zeros(3, 256, 256)

        # 增强的数据增强
        if random.random() > 0.5:  # 50%概率应用颜色增强
            target_texture = self.color_jitter(target_texture)

        if random.random() > 0.5:  # 50%概率应用旋转
            target_texture = self.random_rotation(target_texture)

        if random.random() > 0.5:  # 50%概率应用翻转
            target_texture = self.random_flip(target_texture)

        if random.random() > 0.5:  # 50%概率应用轻微模糊
            target_texture = self.gaussian_blur(target_texture)

        # 添加到缓存
        if len(self.cache) < self.cache_size:
            self.cache[idx] = (inputs, target_kd, target_texture)

        return inputs, target_kd, target_texture


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features)
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.activation(out)


class TextureDiscriminator(nn.Module):
    """修正的纹理判别器"""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 输入: 3 x 256 x 256
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # 128x128
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 修正：使用1x1卷积压缩通道
            nn.Conv2d(512, 1, 1),  # 16x16 -> 16x16
            nn.Flatten(),
            nn.Linear(16 * 16, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)  # 输出形状: [batch]


class EnhancedTexturePredictor(nn.Module):
    def __init__(self, input_dim=33, mtl_output_dim=3, texture_size=256):
        super().__init__()
        self.texture_size = texture_size

        # MTL分支
        self.mtl_branch = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, mtl_output_dim),
            nn.Sigmoid()
        )

        # 纹理编码器
        self.texture_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 重构解码器 - 确保精确输出256x256
        self.decoder = nn.Sequential(
            # 初始4x4
            nn.ConvTranspose2d(256, 192, 4, 2, 1),  # 4x4 -> 8x8
            ResidualBlock(192),

            # 上采样到16x16
            nn.ConvTranspose2d(192, 128, 4, 2, 1),  # 8x8 -> 16x16
            ResidualBlock(128),

            # 上采样到32x32
            nn.ConvTranspose2d(128, 96, 4, 2, 1),  # 16x16 -> 32x32
            ResidualBlock(96),

            # 上采样到64x64
            nn.ConvTranspose2d(96, 64, 4, 2, 1),  # 32x32 -> 64x64
            ResidualBlock(64),

            # 上采样到128x128
            nn.ConvTranspose2d(64, 48, 4, 2, 1),  # 64x64 -> 128x128
            ResidualBlock(48),

            # 上采样到256x256 - 使用转置卷积确保精确尺寸
            nn.ConvTranspose2d(48, 32, 4, 2, 1),  # 128x128 -> 256x256
            nn.LeakyReLU(0.1, inplace=True),

            # 最终输出层
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

        # 从特征向量到初始特征图的映射
        self.fc_to_features = nn.Linear(4096, 256 * 4 * 4)

    def forward(self, x):
        mtl_pred = self.mtl_branch(x)

        # 纹理生成
        x_tex = self.texture_encoder(x)
        x_tex = self.fc_to_features(x_tex)
        x_tex = x_tex.view(-1, 256, 4, 4)  # [B, 256, 4, 4]
        texture_pred = self.decoder(x_tex)

        return mtl_pred, texture_pred


def ssim(img1, img2, window_size=11, size_average=True):
    """计算两个图像的SSIM（支持RGB输入）"""
    # 参数设置
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    # 创建窗口（支持多通道）
    window = torch.ones((1, 1, window_size, window_size)) / (window_size ** 2)
    window = window.to(img1.device)

    # 计算每个通道的SSIM并取平均
    ssim_per_channel = []
    for channel in range(img1.size(1)):  # 遍历每个通道
        img1_ch = img1[:, channel:channel + 1, :, :]  # 提取单通道 [B, 1, H, W]
        img2_ch = img2[:, channel:channel + 1, :, :]  # 提取单通道 [B, 1, H, W]

        mu1 = F.conv2d(img1_ch, window, padding=window_size // 2)
        mu2 = F.conv2d(img2_ch, window, padding=window_size // 2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1_ch * img1_ch, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2_ch * img2_ch, window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1_ch * img2_ch, window, padding=window_size // 2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            ssim_per_channel.append(ssim_map.mean())
        else:
            ssim_per_channel.append(ssim_map.mean(1).mean(1).mean(1))

    # 平均所有通道的SSIM
    return torch.mean(torch.stack(ssim_per_channel))


def gradient_penalty(y, x):
    """计算梯度惩罚项"""
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


def gram_matrix(feature):
    """计算Gram矩阵用于风格损失"""
    b, c, h, w = feature.size()
    features = feature.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    gram = gram.div(c * h * w)
    return gram


def train_model_with_texture(data_triplets, epochs=200, lr=0.0002, batch_size=32):
    # 确保批大小至少为2
    if batch_size < 2:
        print(f"Warning: Batch size increased from {batch_size} to 2 for BatchNorm compatibility")
        batch_size = 2

    if len(data_triplets) == 0:
        print("No training data found!")
        return None

    # 创建数据集和数据加载
    dataset = MeshDataset(data_triplets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, persistent_workers=False,
                            pin_memory=False, drop_last=True)

    # 初始化模型
    generator = EnhancedTexturePredictor().to(device)
    discriminator = TextureDiscriminator().to(device)

    # 优化器
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.81 , 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr * 0.5, betas=(0.81, 0.999))

    # 学习率调度器
    g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        g_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        d_optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    mtl_loss_fn = nn.MSELoss()
    adversarial_loss_fn = nn.BCEWithLogitsLoss()

    # 改进的纹理损失函数
    def texture_loss_function(pred, target, pred_features, target_features):
        # L1损失 - 保留细节
        l1_loss = F.l1_loss(pred, target)

        # 归一化到[0,1]范围 (VGG需要0-1输入)
        pred_norm = (pred + 1) / 2.0
        target_norm = (target + 1) / 2.0

        # 确保在[0,1]范围内
        pred_norm = torch.clamp(pred_norm, 0, 1)
        target_norm = torch.clamp(target_norm, 0, 1)

        # SSIM损失 - 结构相似性
        try:
            ssim_value = ssim(pred_norm, target_norm, window_size=11)
            ssim_loss = 1 - ssim_value
        except Exception as e:
            print(f"SSIM error: {e}")
            ssim_loss = F.mse_loss(pred_norm, target_norm)

        # 特征空间损失
        feat_loss = F.l1_loss(pred_features, target_features)

        # 风格损失 (Gram矩阵)
        pred_gram = gram_matrix(pred_features)
        target_gram = gram_matrix(target_features)
        style_loss = F.mse_loss(pred_gram, target_gram)

        # 调整权重
        return (0.4 * l1_loss +
                0.2 * ssim_loss +
                0.2 * feat_loss +
                0.2 * style_loss)

    # 学习率预热参数
    warmup_epochs = 5
    min_lr = 1e-6

    # 跟踪最佳模型
    best_val_loss = float('inf')
    best_model = None
    epochs_since_improvement = 0
    early_stopping_patience = 10

    print(f"Starting training with {len(data_triplets)} triplets...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        generator.train()
        discriminator.train()

        total_mtl_loss = 0.0
        total_texture_loss = 0.0
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_gp = 0.0
        total_samples = 0

        # 学习率预热
        if epoch < warmup_epochs:
            warmup_factor = min(1.0, (epoch + 1) / warmup_epochs)
            current_g_lr = lr * warmup_factor
            current_d_lr = lr * 0.5 * warmup_factor

            for param_group in g_optimizer.param_groups:
                param_group['lr'] = current_g_lr
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = current_d_lr

        try:
            for batch_idx, batch in enumerate(dataloader):
                inputs, target_kd, target_texture = batch
                inputs = inputs.to(device)
                target_kd = target_kd.to(device)
                target_texture = target_texture.to(device)
                batch_size = inputs.size(0)

                # 真实和假标签
                real_labels = torch.ones(batch_size, device=device)
                fake_labels = torch.zeros(batch_size, device=device)

                # ---------------------
                #  训练判别器
                # ---------------------
                d_optimizer.zero_grad()

                # 真实图像
                real_output = discriminator(target_texture)
                d_loss_real = adversarial_loss_fn(real_output, real_labels)

                # 生成图像
                mtl_pred, texture_pred = generator(inputs)
                fake_output = discriminator(texture_pred.detach())
                d_loss_fake = adversarial_loss_fn(fake_output, fake_labels)

                # 梯度惩罚
                alpha = torch.rand(batch_size, 1, 1, 1, device=device)
                interpolated = (alpha * target_texture + (1 - alpha) * texture_pred.detach()).requires_grad_(True)
                d_output = discriminator(interpolated)
                gp = gradient_penalty(d_output, interpolated)

                # 总判别器损失
                d_loss = d_loss_real + d_loss_fake + 10.0 * gp
                d_loss.backward()
                d_optimizer.step()

                # ---------------------
                #  训练生成器
                # ---------------------
                g_optimizer.zero_grad()

                # 对抗损失
                fake_output = discriminator(texture_pred)
                g_loss_adv = adversarial_loss_fn(fake_output, real_labels)

                # MTL损失
                mtl_loss = mtl_loss_fn(mtl_pred, target_kd)

                # 特征损失
                with torch.no_grad():
                    target_features = vgg_feature_extractor((target_texture + 1) / 2.0)
                pred_features = vgg_feature_extractor((texture_pred + 1) / 2.0)
                texture_loss = texture_loss_function(texture_pred, target_texture, pred_features, target_features)

                # 总生成器损失
                g_loss = mtl_loss + 1.5 * texture_loss + 0.5 * g_loss_adv
                g_loss.backward()
                g_optimizer.step()

                # 累计损失
                total_mtl_loss += mtl_loss.item() * batch_size
                total_texture_loss += texture_loss.item() * batch_size
                total_g_loss += g_loss.item() * batch_size
                total_d_loss += d_loss.item() * batch_size
                total_gp += gp.item() * batch_size
                total_samples += batch_size

                # 清理GPU内存
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # 每10个batch打印一次进度
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(dataloader)}: "
                          f"G Loss: {g_loss.item():.4f} (MTL: {mtl_loss.item():.4f}, Tex: {texture_loss.item():.4f}, Adv: {g_loss_adv.item():.4f}) | "
                          f"D Loss: {d_loss.item():.4f} (Real: {d_loss_real.item():.4f}, Fake: {d_loss_fake.item():.4f}, GP: {gp.item():.4f})")

        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            traceback.print_exc()
            continue

        # 计算平均损失
        if total_samples > 0:
            avg_mtl_loss = total_mtl_loss / total_samples
            avg_texture_loss = total_texture_loss / total_samples
            avg_g_loss = total_g_loss / total_samples
            avg_d_loss = total_d_loss / total_samples
            avg_gp = total_gp / total_samples
        else:
            avg_mtl_loss = avg_texture_loss = avg_g_loss = avg_d_loss = avg_gp = 0

        # 验证步骤
        generator.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            # 使用随机10%的数据作为验证集
            val_indices = random.sample(range(len(dataset)), max(1, len(dataset) // 10))
            for idx in val_indices:
                inputs, target_kd, target_texture = dataset[idx]
                inputs = inputs.unsqueeze(0).to(device)
                target_kd = target_kd.unsqueeze(0).to(device)
                target_texture = target_texture.unsqueeze(0).to(device)

                val_mtl, val_tex = generator(inputs)
                val_mtl_loss = mtl_loss_fn(val_mtl, target_kd)

                # 验证阶段使用简化的纹理损失
                val_texture_loss = F.l1_loss(val_tex, target_texture)

                val_loss += (val_mtl_loss + 1.5 * val_texture_loss).item()
                val_samples += 1

        if val_samples > 0:
            val_loss /= val_samples

        # 更新学习率
        g_scheduler.step(val_loss)
        d_scheduler.step(avg_d_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = generator.state_dict()
            epochs_since_improvement = 0
            print(f"New best model with val loss: {val_loss:.4f}")
            # 保存模型检查点
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, f"best_model_epoch_{epoch + 1}.pth")
        else:
            epochs_since_improvement += 1

        # 检查是否早停
        if epochs_since_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch + 1} due to no improvement for {early_stopping_patience} epochs")
            break

        epoch_time = time.time() - epoch_start_time
        # 打印epoch结果
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Train G Loss: {avg_g_loss:.4f} | "
              f"MTL Loss: {avg_mtl_loss:.4f} | "
              f"Texture Loss: {avg_texture_loss:.4f} | "
              f"D Loss: {avg_d_loss:.4f} | "
              f"GP: {avg_gp:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s | "
              f"Since Imp: {epochs_since_improvement}/{early_stopping_patience}")

    # 加载最佳模型
    if best_model is not None:
        generator.load_state_dict(best_model)
        print(f"Loaded best model with val loss: {best_val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds!")
    return generator


def generate_mtl_and_texture(model, new_obj_path, output_folder):
    """生成完整的OBJ+MTL+TGA文件 - 修复MTL与OBJ匹配问题"""
    model.eval()

    os.makedirs(output_folder, exist_ok=True)

    # 获取基础文件名
    obj_filename = os.path.splitext(os.path.basename(new_obj_path))[0]
    material_name = f"{obj_filename.lower()}_Material"

    # 创建输出文件名
    output_obj_path = os.path.join(output_folder, f"{obj_filename.lower()}.obj")
    output_mtl_path = os.path.join(output_folder, f"{material_name}.mtl")
    output_tga_path = os.path.join(output_folder, f"{material_name}.tga")

    try:
        # 1. 复制原始OBJ文件
        shutil.copy(new_obj_path, output_obj_path)

        # 2. 生成预测结果
        with torch.no_grad():
            # 加载几何数据
            verts, faces, aux = load_obj(
                new_obj_path,
                load_textures=False,
                create_texture_atlas=False
            )
            faces_tensor = faces.verts_idx

            # 提取特征
            input_features = extract_mesh_features(verts, faces_tensor)
            inputs = input_features.unsqueeze(0).to(device)

            # 模型预测
            mtl_pred, texture_pred = model(inputs)

            # 处理纹理输出
            texture_pred = texture_pred.squeeze(0).cpu()
            texture_pred = (texture_pred / 2.0) + 0.5
            texture_pred = torch.clamp(texture_pred, 0, 1)

            mtl_pred = mtl_pred.squeeze(0).cpu()

        # 3. 生成MTL文件
        pred_kd = mtl_pred.detach().numpy()
        mtl_content = f"""newmtl {material_name}
Ka 0.1000 0.1000 0.1000
Kd {pred_kd[0]:.4f} {pred_kd[1]:.4f} {pred_kd[2]:.4f}
Ks 0.5000 0.5000 0.5000
Ns 96.0784
Ni 1.4500
d 1.0000
illum 2
map_Kd {material_name}.tga
"""
        with open(output_mtl_path, 'w', encoding='utf-8') as f:
            f.write(mtl_content)

        # 4. 生成TGA文件
        texture_np = texture_pred.detach().numpy()
        texture_np = np.transpose(texture_np, (1, 2, 0))
        texture_np = (texture_np * 255).astype(np.uint8)
        img = Image.fromarray(texture_np, 'RGB')
        img.save(output_tga_path, 'TGA')

        # 5. 更新OBJ文件 - 修复的关键部分
        with open(output_obj_path, 'r', encoding='utf-8') as f:
            obj_lines = f.readlines()

        # 查找现有的mtllib行和usemtl行
        mtllib_index = -1
        usemtl_indices = []

        for i, line in enumerate(obj_lines):
            if line.startswith('mtllib'):
                mtllib_index = i
            elif line.startswith('usemtl'):
                usemtl_indices.append(i)

        # 更新或添加mtllib行
        mtllib_line = f"mtllib {material_name}.mtl\n"
        if mtllib_index != -1:
            obj_lines[mtllib_index] = mtllib_line
        else:
            # 如果没有mtllib行，添加到文件开头
            obj_lines.insert(0, mtllib_line)

        # 添加usemtl行到每个物体组前
        if not usemtl_indices:
            # 如果没有现有usemtl行，找到第一个顶点或面定义之前的位置
            insert_index = 0
            for i, line in enumerate(obj_lines):
                if line.startswith('v ') or line.startswith('f '):
                    insert_index = i
                    break

            # 在第一个几何元素前添加usemtl
            if insert_index > 0:
                obj_lines.insert(insert_index, f"usemtl {material_name}\n")
            else:
                # 如果没有几何元素，添加到文件末尾
                obj_lines.append(f"usemtl {material_name}\n")
        else:
            # 替换所有现有的usemtl行
            for i in usemtl_indices:
                obj_lines[i] = f"usemtl {material_name}\n"

        # 写回更新后的OBJ
        with open(output_obj_path, 'w', encoding='utf-8') as f:
            f.writelines(obj_lines)

        return True, f"Generated: {obj_filename}.obj + {material_name}.mtl + {material_name}.tga"

    except Exception as e:
        return False, f"Error generating files for {obj_filename}: {str(e)}"


def batch_generate_mtl_and_texture(model, obj_folder, output_folder):
    """为文件夹中的所有OBJ文件批量生成OBJ+MTL+TGA"""
    model.eval()

    if not os.path.exists(obj_folder):
        print(f"Input OBJ folder does not exist: {obj_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    # 获取所有OBJ文件
    obj_files = [f for f in os.listdir(obj_folder) if f.lower().endswith('.obj')]

    if len(obj_files) == 0:
        print(f"No OBJ files found in {obj_folder}")
        return

    print(f"Found {len(obj_files)} OBJ files to process...")

    success_count = 0
    error_count = 0

    for i, obj_file in enumerate(obj_files, 1):
        obj_path = os.path.join(obj_folder, obj_file)
        success, message = generate_mtl_and_texture(model, obj_path, output_folder)

        if success:
            success_count += 1
            print(f"[{i}/{len(obj_files)}] ✓ {message}")
        else:
            error_count += 1
            print(f"[{i}/{len(obj_files)}] ✗✗ {message}")

    print(f"\nBatch processing completed!")
    print(f"Success: {success_count}, Errors: {error_count}")


def save_model(model, save_path):
    """保存训练好的模型"""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model(model_path):
    """加载训练好的模型"""
    model = EnhancedTexturePredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


# 主程序
if __name__ == "__main__":
    # ========== 第一步：准备训练数据 =========
    training_obj_folder = '/root/miniconda3/envs/pytorch3d/objtraindata'  # 训练用OBJ文件夹
    training_mtl_folder = '/root/miniconda3/envs/pytorch3d/mtltraindata'  # 训练用MTL文件夹
    training_tga_folder = '/root/miniconda3/envs/pytorch3d/tgatraindata'  # 训练用TGA文件夹

    # 自动扫描并匹配训练数据
    print("Scanning for training data...")
    data_triplets = scan_training_data_with_texture(training_obj_folder, training_mtl_folder, training_tga_folder)

    if len(data_triplets) == 0:
        print("No training data found. Please check your folder paths.")
        exit()

    # 测试模型输出尺寸
    print("\nTesting model output size...")
    test_model = EnhancedTexturePredictor().to(device)
    test_model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 33).to(device)
        mtl_pred, tex_pred = test_model(test_input)
        print(f"Model output texture size: {tex_pred.shape}")

        # 检查尺寸并打印详细信息
        if tex_pred.shape[2] != 256 or tex_pred.shape[3] != 256:
            print(f"错误: 模型输出尺寸应为256x256，实际为{tex_pred.shape[2]}x{tex_pred.shape[3]}")
            print("解码器结构:")
            for i, layer in enumerate(test_model.decoder):
                print(f"层 {i}: {layer}")
            exit()
        else:
            print("模型输出尺寸验证通过!")

    # ========== 第二步：训练模型 ==========
    print("\n" + "=" * 50)
    print("TRAINING PHASE")
    print("=" * 50)

    # 使用300个epochs进行训练
    model = train_model_with_texture(data_triplets, epochs=300, batch_size=16)

    if model is None:
        print("Training failed!")
        exit()

    # 保存训练好的模型
    model_save_path = '/root/miniconda3/envs/pytorch3d/trained_model_with_texture.pth'
    save_model(model, model_save_path)

    # ========== 第三步：批量生成新的MTL和TGA文件 ==========
    print("\n" + "=" * 50)
    print("BATCH GENERATION PHASE")
    print("=" * 50)

    model.eval()

    new_obj_folder = '/root/miniconda3/envs/pytorch3d/newobj'  # 新OBJ文件夹
    output_folder = '/root/miniconda3/envs/pytorch3d/results'  # 输出文件夹

    # 批量生成MTL和TGA文件
    batch_generate_mtl_and_texture(model, new_obj_folder, output_folder)

    print("\n" + "=" * 50)
    print("ALL PROCESSING COMPLETED!")
    print("=" * 50)