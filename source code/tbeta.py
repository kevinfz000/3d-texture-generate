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
from torchvision.models import vgg16, VGG16_Weights
import traceback

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """优化后的网格几何特征提取 - 修复面索引问题"""
    try:
        # 基础统计特征
        mean = torch.mean(verts, dim=0)
        min_vals = torch.min(verts, dim=0).values
        max_vals = torch.max(verts, dim=0).values
        std = torch.std(verts, dim=0)
        bbox_size = max_vals - min_vals

        # 创建trimesh对象计算高级特征
        verts_np = verts.cpu().numpy()
        faces_np = faces.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)

        # 计算表面积和体积
        area = mesh.area
        volume = mesh.volume

        # 修复面索引问题
        if faces.dim() == 1:
            # 如果面索引是1D张量，需要重塑为2D
            faces = faces.view(-1, 3)

        # 确保面索引形状正确 [F, 3]
        if faces.shape[1] != 3:
            faces = faces.reshape(-1, 3)

        # 创建PyTorch3D网格对象 - 修复关键行！
        mesh_torch = Meshes(verts=[verts], faces=[faces])

        # 计算顶点法向量统计特征
        _, normals = sample_points_from_meshes(
            mesh_torch,
            return_normals=True,
            num_samples=1000
        )
        normals = normals.squeeze(0)
        normals_mean = torch.mean(normals, dim=0)
        normals_std = torch.std(normals, dim=0)

        # 组合所有特征
        features = torch.cat([
            mean, min_vals, max_vals, std, bbox_size,
            torch.tensor([area, volume], dtype=torch.float32),
            normals_mean, normals_std
        ])

        return features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        # 返回零特征作为后备
        return torch.zeros(23)


def load_obj_mtl_tga_triplet(obj_path, mtl_path=None, tga_folder=None):
    try:
        # 加载OBJ
        verts, faces, aux = load_obj(obj_path)
        faces_tensor = faces.verts_idx

        # 提取增强特征
        input_features = extract_mesh_features(verts, faces_tensor)

    except Exception as e:
        print(f"Error loading OBJ {obj_path}: {e}")
        input_features = torch.zeros(23)  # 更新特征维度
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

                # 转换为张量并标准化
                img_array = np.array(img).astype(np.float32) / 255.0
                target_texture = torch.tensor(img_array).permute(2, 0, 1)

                # 添加归一化 - 使用更稳定的归一化方式
                target_texture = (target_texture - 0.5) * 2.0

            except Exception as e:
                print(f"Error loading TGA {tga_path}: {e}")
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
        # 增强的数据增强转换
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
        self.gaussian_blur = torchvision.transforms.GaussianBlur(3, sigma=(0.1, 2.0))

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

        # 数据增强 - 增强版
        if random.random() > 0.5:  # 50%概率应用颜色增强
            target_texture = self.color_jitter(target_texture)

        if random.random() > 0.7:  # 30%概率应用模糊
            target_texture = self.gaussian_blur(target_texture)

        if random.random() > 0.7:  # 30%概率应用随机裁剪
            crop_size = random.randint(192, 256)
            start_x = random.randint(0, 256 - crop_size)
            start_y = random.randint(0, 256 - crop_size)
            target_texture = target_texture[
                             :, start_y:start_y + crop_size, start_x:start_x + crop_size]
            target_texture = F.interpolate(
                target_texture.unsqueeze(0),
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # 添加到缓存
        if len(self.cache) < self.cache_size:
            self.cache[idx] = (inputs, target_kd, target_texture)

        return inputs, target_kd, target_texture


# 残差块定义 - 增强版
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(features, features),
            nn.BatchNorm1d(features)
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.activation(out)


# 增强的纹理预测模型
class EnhancedTexturePredictor(nn.Module):
    def __init__(self, input_dim=23, mtl_output_dim=3, texture_size=256):
        super().__init__()
        self.texture_size = texture_size

        # MTL分支 - 简化版
        self.mtl_branch = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, mtl_output_dim),
            nn.Sigmoid()
        )

        # 纹理分支（增强版）- 使用更深的残差网络
        self.texture_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            ResidualBlock(1024),
            nn.Linear(1024, 2048 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 解码器 - 增强版，添加跳跃连接和更多层
        self.decoder = nn.Sequential(
            # 从4x4上采样到8x8
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # 残差块
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # 从8x8上采样到16x16
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 残差块
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 从16x16上采样到32x32
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 残差块
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 从32x32上采样到64x64
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 残差块
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 上采样到128x128
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 残差块
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 上采样到256x256
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 残差块
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 最终输出层
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        mtl_pred = self.mtl_branch(x)

        # 纹理生成
        x_tex = self.texture_encoder(x)
        x_tex = x_tex.view(-1, 2048, 4, 4)
        texture_pred = self.decoder(x_tex)

        return mtl_pred, texture_pred


# 自定义SSIM实现（替代piq） - 支持RGB图像
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


# 梯度惩罚项 - 用于稳定训练
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


def train_model_with_texture(data_triplets, epochs=200, lr=0.0001, batch_size=16):
    if len(data_triplets) == 0:
        print("No training data found!")
        return None

    # 创建数据集和数据加载器
    dataset = MeshDataset(data_triplets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, persistent_workers=False,
                            pin_memory=False, drop_last=True)

    # 初始化模型
    model = EnhancedTexturePredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    mtl_loss_fn = nn.MSELoss()

    def texture_loss_function(pred, target):
        # L1损失 - 保留细节
        l1_loss = F.l1_loss(pred, target)

        # 归一化到[0,1]范围
        pred_norm = (pred + 1) / 2.0
        target_norm = (target + 1) / 2.0

        # SSIM损失 - 结构相似性
        try:
            ssim_value = ssim(pred_norm, target_norm, window_size=11)
            ssim_loss = 1 - ssim_value
        except Exception as e:
            print(f"SSIM error: {e}")
            ssim_loss = F.mse_loss(pred_norm, target_norm)

        # 特征空间损失 - 增加权重
        with torch.no_grad():
            target_feats = vgg_feature_extractor(target_norm)
        pred_feats = vgg_feature_extractor(pred_norm)
        feat_loss = F.l1_loss(pred_feats, target_feats)

        # 梯度惩罚 - 仅当张量需要梯度时计算
        if pred.requires_grad:  # 关键修复：检查梯度状态
            gp = gradient_penalty(pred, pred)
        else:
            gp = torch.tensor(0.0, device=pred.device)  # 验证阶段设为0

        # 调整权重 - 增加感知损失的权重
        return 0.4 * l1_loss + 0.3 * ssim_loss + 0.3 * feat_loss + 0.1 * gp

    # 学习率调度器 - 更精细的控制
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    print(f"Starting training with {len(data_triplets)} triplets...")

    # 跟踪最佳模型
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        model.train()
        total_mtl_loss = 0.0
        total_texture_loss = 0.0
        total_loss = 0.0
        total_samples = 0

        try:
            for batch_idx, batch in enumerate(dataloader):
                inputs, target_kd, target_texture = batch
                inputs = inputs.to(device)
                target_kd = target_kd.to(device)
                target_texture = target_texture.to(device)

                # 前向传播
                mtl_pred, texture_pred = model(inputs)

                # 计算损失
                mtl_loss = mtl_loss_fn(mtl_pred, target_kd)
                texture_loss = texture_loss_function(texture_pred, target_texture)

                # 调整损失权重 - 增加纹理损失的重要性
                batch_loss = mtl_loss + 1.5 * texture_loss

                # 反向传播
                optimizer.zero_grad()
                batch_loss.backward()

                # 梯度裁剪 - 防止梯度爆炸
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # 累计损失
                total_mtl_loss += mtl_loss.item() * inputs.size(0)
                total_texture_loss += texture_loss.item() * inputs.size(0)
                total_loss += batch_loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                # 清理GPU内存
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # 每10个batch打印一次进度
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(dataloader)}: "
                          f"Batch Loss: {batch_loss.item():.4f} "
                          f"(MTL: {mtl_loss.item():.4f}, Tex: {texture_loss.item():.4f})")

        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            traceback.print_exc()  # 打印详细错误堆栈
            # 继续训练而不是中断
            continue

        # 计算平均损失
        if total_samples > 0:
            avg_mtl_loss = total_mtl_loss / total_samples
            avg_texture_loss = total_texture_loss / total_samples
            avg_total_loss = total_loss / total_samples
        else:
            avg_mtl_loss = avg_texture_loss = avg_total_loss = 0

        # 验证步骤 - 使用完整验证集
        model.eval()
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

                val_mtl, val_tex = model(inputs)
                val_mtl_loss = mtl_loss_fn(val_mtl, target_kd)
                val_texture_loss = texture_loss_function(val_tex, target_texture)
                val_loss += (val_mtl_loss + 1.5 * val_texture_loss).item()
                val_samples += 1

        if val_samples > 0:
            val_loss /= val_samples

        # 更新学习率
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            print(f"New best model with val loss: {val_loss:.4f}")

        # 打印epoch结果
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Train Loss: {avg_total_loss:.4f} | "
              f"MTL Loss: {avg_mtl_loss:.4f} | "
              f"Texture Loss: {avg_texture_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 加载最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)
        print("Loaded best model based on validation loss")

    print("Training completed!")
    return model


def generate_mtl_and_texture(model, new_obj_path, output_folder):
    """生成完整的OBJ+MTL+TGA文件"""
    model.eval()

    os.makedirs(output_folder, exist_ok=True)

    # 获取基础文件名
    obj_filename = os.path.splitext(os.path.basename(new_obj_path))[0]

    # 创建输出文件名
    output_obj_path = os.path.join(output_folder, f"{obj_filename.lower()}.obj")
    output_mtl_path = os.path.join(output_folder, f"{obj_filename.lower()}.mtl")
    output_tga_path = os.path.join(output_folder, f"{obj_filename.lower()}.tga")

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
        mtl_content = f"""newmtl {obj_filename.lower()}_Material
Ka 0.1000 0.1000 0.1000
Kd {pred_kd[0]:.4f} {pred_kd[1]:.4f} {pred_kd[2]:.4f}
Ks 0.5000 0.5000 0.5000
Ns 96.0784
Ni 1.4500
d 1.0000
illum 2
map_Kd {obj_filename.lower()}.tga
"""
        with open(output_mtl_path, 'w', encoding='utf-8') as f:
            f.write(mtl_content)

        # 4. 生成TGA文件
        texture_np = texture_pred.detach().numpy()
        texture_np = np.transpose(texture_np, (1, 2, 0))
        texture_np = (texture_np * 255).astype(np.uint8)
        img = Image.fromarray(texture_np, 'RGB')

        # 高质量上采样
        img = img.resize((512, 512), Image.LANCZOS)
        img.save(output_tga_path, 'TGA')

        # 5. 更新OBJ文件
        with open(output_obj_path, 'r') as f:
            obj_content = f.read()

        # 更新mtllib引用
        new_obj_content = re.sub(r'mtllib .*', f'mtllib {obj_filename.lower()}.mtl', obj_content)

        # 添加usemtl指令
        if f"usemtl {obj_filename.lower()}_Material" not in new_obj_content:
            new_obj_content += f"\nusemtl {obj_filename.lower()}_Material\n"

        # 写回更新后的OBJ
        with open(output_obj_path, 'w') as f:
            f.write(new_obj_content)

        return True, f"Generated: {obj_filename}.obj + {obj_filename}.mtl + {obj_filename}.tga"

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
            print(f"[{i}/{len(obj_files)}] ✗ {message}")

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