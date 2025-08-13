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
from torchvision import transforms  # 添加这行
from torchvision.models import vgg16, VGG16_Weights
import traceback
import math
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练VGG用于特征损失
vgg_feature_extractor = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
for param in vgg_feature_extractor.parameters():
    param.requires_grad = False

# ... [其余代码保持不变] ...

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
            num_samples=5000  # 增加采样点
        )
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

        return features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return torch.zeros(33)  # 修正为实际维度


# 修改纹理加载和预处理
def load_obj_mtl_tga_triplet(obj_path, mtl_path=None, tga_folder=None):
    try:
        # 加载OBJ
        verts, faces, aux = load_obj(obj_path)
        faces_tensor = faces.verts_idx

        # 提取增强特征
        input_features = extract_mesh_features(verts, faces_tensor)

    except Exception as e:
        print(f"Error loading OBJ {obj_path}: {e}")
        input_features = torch.zeros(35)  # 更新特征维度
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

                print(f"Error loading OBJ {obj_path}: {e}")

                input_features = torch.zeros(33)  # 修正为33维
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


# 数据集类 - 简化数据增强
class MeshDataset(Dataset):
    def __init__(self, triplets, cache_size=100):
        self.triplets = triplets
        self.cache = {}
        self.cache_size = cache_size
        # 使用 torchvision.transforms.GaussianBlur
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
        self.gaussian_blur = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

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

        # 更温和的数据增强
        if random.random() > 0.7:  # 30%概率应用颜色增强
            target_texture = self.color_jitter(target_texture)

        if random.random() > 0.8:  # 20%概率应用轻微模糊
            target_texture = self.gaussian_blur(target_texture)

        # 添加到缓存
        if len(self.cache) < self.cache_size:
            self.cache[idx] = (inputs, target_kd, target_texture)

        return inputs, target_kd, target_texture
        print(f"Loaded texture size: {target_texture.shape}")  # 应为[3,256,256]
        return inputs, target_kd, target_texture



# 增强的残差块
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


# 重构的纹理预测模型
class EnhancedTexturePredictor(nn.Module):
    def __init__(self, input_dim=33, mtl_output_dim=3, texture_size=256):
        super().__init__()
        self.texture_size = texture_size

        # 修改MTL分支
        self.mtl_branch = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.InstanceNorm1d(64) if input_dim == 1 else nn.BatchNorm1d(64),  # 自适应归一化
            nn.Linear(64, mtl_output_dim),
            nn.Sigmoid()
        )

        # 纹理分支 - 重构为更合理的结构
        # 在纹理编码器中添加自适应归一化
        self.texture_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.InstanceNorm1d(512) if input_dim == 1 else nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 修改后的解码器部分
        # 重构后的解码器 - 使用逐步上采样
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4x4 -> 8x8
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8x8 -> 16x16
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16x16 -> 32x32
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 32x32 -> 64x64
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 64x64 -> 128x128
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 128x128 -> 256x256
            nn.Conv2d(16, 3, 3, padding=1),
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


def train_model_with_texture(data_triplets, epochs=200, lr=0.001, batch_size=16):
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
    model = EnhancedTexturePredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    mtl_loss_fn = nn.MSELoss()

    # ... [其余代码保持不变] ...
    # 改进的纹理损失函数 - 调整权重
    def texture_loss_function(pred, target, training=True):
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

        # 特征空间损失 - 使用更深的VGG特征
        with torch.no_grad():
            target_feats = vgg_feature_extractor(target_norm)
        pred_feats = vgg_feature_extractor(pred_norm)
        feat_loss = F.l1_loss(pred_feats, target_feats)

        # 梯度惩罚 - 仅当张量需要梯度时计算（训练阶段）
        gp = torch.tensor(0.0, device=pred.device)
        if training and pred.requires_grad:
            gp = gradient_penalty(pred, pred)

        # 调整权重 - 增加L1和SSIM损失的权重
        return 0.4 * l1_loss + 0.3 * ssim_loss + 0.3 * feat_loss + 0.1 * gp

    # 学习率预热参数
    warmup_epochs = 10
    warmup_factor = 0.01
    min_lr = 1e-6

    # 跟踪最佳模型
    best_val_loss = float('inf')
    best_model = None
    epochs_since_improvement = 0
    early_stopping_patience = 15

    print(f"Starting training with {len(data_triplets)} triplets...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_mtl_loss = 0.0
        total_texture_loss = 0.0
        total_loss = 0.0
        total_samples = 0
        total_grad_norm = 0.0

        # 学习率预热
        if epoch < warmup_epochs:
            warmup_factor = min(1.0, (epoch + 1) / warmup_epochs)
            current_lr = lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            # 余弦退火学习率
            cos_factor = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
            current_lr = min_lr + (lr - min_lr) * cos_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

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
                texture_loss = texture_loss_function(texture_pred, target_texture, training=True)
                batch_loss = mtl_loss + 1.5 * texture_loss

                # 反向传播
                optimizer.zero_grad()
                batch_loss.backward()

                # 梯度裁剪 - 防止梯度爆炸
                total_grad_norm += nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
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
                          f"(MTL: {mtl_loss.item():.4f}, Tex: {texture_loss.item():.4f}) "
                          f"LR: {current_lr:.6f}")

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
            avg_grad_norm = total_grad_norm / len(dataloader)
        else:
            avg_mtl_loss = avg_texture_loss = avg_total_loss = avg_grad_norm = 0

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

                # 验证阶段使用简化的纹理损失（L1损失）
                val_texture_loss = F.l1_loss(val_tex, target_texture)

                val_loss += (val_mtl_loss + 1.5 * val_texture_loss).item()
                val_samples += 1

        if val_samples > 0:
            val_loss /= val_samples

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            epochs_since_improvement = 0
            print(f"New best model with val loss: {val_loss:.4f}")
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
              f"Train Loss: {avg_total_loss:.4f} | "
              f"MTL Loss: {avg_mtl_loss:.4f} | "
              f"Texture Loss: {avg_texture_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Grad Norm: {avg_grad_norm:.4f} | "
              f"Time: {epoch_time:.1f}s | "
              f"Since Imp: {epochs_since_improvement}/{early_stopping_patience}")

    # 加载最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)
        print(f"Loaded best model with val loss: {best_val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds!")
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


def test_decoder_sizes():
    """系统测试解码器每一层的输出尺寸"""
    print("\n" + "=" * 50)
    print("DECODER SIZE TESTING")
    print("=" * 50)

    # 创建测试模型
    test_model = EnhancedTexturePredictor()
    test_model.debug_mode = True  # 启用调试模式

    # 模拟输入
    test_input = torch.randn(1, 256, 4, 4)
    print(f"初始输入尺寸: {test_input.shape}")

    # 运行前向传播（会打印每一层输出）
    with torch.no_grad():
        _ = test_model.decoder(test_input)

    # 完整模型测试
    print("\n完整模型测试:")
    test_model_full = EnhancedTexturePredictor()
    test_model_full.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 33)
        mtl_pred, tex_pred = test_model_full(test_input)
        print(f"最终输出尺寸: {tex_pred.shape}")
        assert tex_pred.shape[2:] == (256, 256), \
            f"输出尺寸应为256x256，实际为{tex_pred.shape[2]}x{tex_pred.shape[3]}"

    print("解码器尺寸验证通过！")




# 主程序
if __name__ == "__main__":
    print("测试解码器尺寸...")
    test_decoder_sizes()
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
    # 在主程序训练前添加
    test_model = EnhancedTexturePredictor()
    test_model.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        test_input = torch.randn(1, 33)
        mtl_pred, tex_pred = test_model(test_input)
        print(f"Model output texture size: {tex_pred.shape}")  # 应为[1,3,256,256]


        # 在主程序训练前添加
        def test_model_output_size():
            test_model = EnhancedTexturePredictor()
            test_model.eval()
            with torch.no_grad():
                test_input = torch.randn(1, 33)
                mtl_pred, tex_pred = test_model(test_input)
                print(f"Model output texture size: {tex_pred.shape}")
                assert tex_pred.shape[2] == 256 and tex_pred.shape[3] == 256, \
                    f"模型输出尺寸应为256x256，实际为{tex_pred.shape[2]}x{tex_pred.shape[3]}"
            print("模型输出尺寸验证通过！")


        print("测试模型输出尺寸...")
        test_model_output_size()
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