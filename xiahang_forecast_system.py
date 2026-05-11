#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
import pvlib
import pickle
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
import joblib
from scipy.interpolate import griddata
from scipy.spatial import cKDTree  # 使用KD树加速
from scipy.spatial.distance import cdist

from airport_visibility_common import (
    build_airport_model,
    build_inference_matrix,
    build_improved_pmst_model,
    build_pmst_inference_matrix,
    predict_classes_from_probs,
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visibility_forecast.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------
# 模型定义
# ------------------------------
class EnhancedAttentionBlock(nn.Module):
    """增强的注意力机制，专注于低能见度相关特征"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 多头自注意力
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )

        # 特征重要性门控机制
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 残差连接和层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # 自注意力
        attn_out, attn_weights = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)

        # 特征门控
        x_flat = x.squeeze(1)
        gate = self.feature_gate(x_flat)
        x_gated = x_flat * gate

        # 前馈网络
        ffn_out = self.ffn(x_gated)
        x_final = self.norm2(x_gated + ffn_out)

        return x_final, attn_weights

class BalancedLowVisibilityClassifier(nn.Module):
    """
    平衡的低能见度分类器 - 改进版本
    """
    def __init__(self, input_dim, hidden_dim=512, num_classes=3, dropout=0.25):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),

            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 残差块
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim, dropout) for _ in range(4)
        ])

        # 注意力机制
        self.attention = EnhancedAttentionBlock(hidden_dim, num_heads=12, dropout=dropout)

        # 分离的分类头
        # 低能见度检测分支
        self.low_vis_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(hidden_dim // 4, 1)
        )

        # 精细分类分支
        self.fine_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(hidden_dim // 4, num_classes)
        )

        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(num_classes + 1, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_dim // 4, num_classes)
        )

        # 权重初始化
        self.apply(self._init_weights)

    def _make_residual_block(self, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)

        # 残差块处理
        for block in self.residual_blocks:
            residual = features
            features = block(features)
            features = F.relu(features + residual)

        # 注意力机制
        features, attention_weights = self.attention(features)

        # 双分支预测
        low_vis_confidence = self.low_vis_detector(features)
        fine_logits = self.fine_classifier(features)

        # 融合预测
        combined_features = torch.cat([fine_logits, low_vis_confidence], dim=1)
        final_logits = self.fusion_net(combined_features)

        return final_logits, low_vis_confidence, attention_weights

# ------------------------------
# 工具函数
# ------------------------------
def calculate_zenith_angle(latitudes, longitudes, times):
    """计算给定站点的太阳天顶角（输入已是UTC时间）"""
    try:
        times = pd.DatetimeIndex(times)
        # 如果时间没有时区信息，直接打上 UTC 标签；如果有，就确保它是 UTC
        if times.tz is None:
            times = times.tz_localize('UTC')
        else:
            times = times.tz_convert('UTC')

        broadcasted_times = np.repeat(times, len(latitudes))
        broadcasted_lats = np.tile(latitudes, len(times))
        broadcasted_lons = np.tile(longitudes, len(times))
        solar_position = pvlib.solarposition.get_solarposition(
            time=broadcasted_times, latitude=broadcasted_lats, longitude=broadcasted_lons
        )
        zenith_angles = solar_position['apparent_zenith'].values
        zenith_angles = zenith_angles.reshape(len(times), len(latitudes))
        return zenith_angles
    except Exception as e:
        logger.warning(f"天顶角计算失败: {e}")
        return np.zeros((len(times), len(latitudes)))

def get_nearest_vegetation_type(latitudes, longitudes, vegetation_data):
    """获取离站点最近的植被类型"""
    vegetation_type = []
    lat_idx = np.digitize(latitudes, vegetation_data.latitude.values) - 1
    lon_idx = np.digitize(longitudes, vegetation_data.longitude.values) - 1
    lat_idx = np.clip(lat_idx, 0, len(vegetation_data.latitude) - 1)
    lon_idx = np.clip(lon_idx, 0, len(vegetation_data.longitude) - 1)

    for lat, lon in zip(lat_idx, lon_idx):
        veg_type = vegetation_data.isel(latitude=lat, longitude=lon)['htcc']
        vegetation_type.append(veg_type.values)
    return np.array(vegetation_type)

def create_vegetation_embeddings(latitudes, longitudes, vegetation_data, time_coords, station_coords):
    """创建植被嵌入特征"""
    try:
        vegetation_data = vegetation_data.sortby('latitude').sortby('longitude')
        vegetation_data = vegetation_data.interpolate_na(dim='latitude', method='nearest')
        vegetation_data = vegetation_data.interpolate_na(dim='longitude', method='nearest')
        vegetation_type = get_nearest_vegetation_type(latitudes, longitudes, vegetation_data)
        unique_vegetation_types = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20])
        type_to_index = {v:i for i,v in enumerate(unique_vegetation_types)}
        vegetation_type_indices = np.array([type_to_index.get(v, 0) for v in vegetation_type])
        vegetation_type_indices = torch.tensor(vegetation_type_indices, dtype=torch.long)
        embedding_dim = 8
        embedding_layer = nn.Embedding(len(unique_vegetation_types), embedding_dim)
        vegetation_embeddings = embedding_layer(vegetation_type_indices)
        vegetation_embeddings_np = vegetation_embeddings.detach().numpy()
        vegetation_embeddings_broadcasted = np.tile(
            vegetation_embeddings_np[:, np.newaxis, :], (1, len(time_coords), 1)
        ).transpose(1,0,2)
        vegetation_da = xr.DataArray(
            vegetation_embeddings_broadcasted,
            coords={'time': time_coords, 'num_station': station_coords,
                    'variable':[f'veg_emb_{i}' for i in range(embedding_dim)]},
            dims=['time','num_station','variable']
        )
        return vegetation_da
    except Exception as e:
        logger.warning(f"植被嵌入计算失败: {e}")
        # 返回零填充的嵌入
        embedding_dim = 8
        zero_embeddings = np.zeros((len(time_coords), len(station_coords), embedding_dim))
        vegetation_da = xr.DataArray(
            zero_embeddings,
            coords={'time': time_coords, 'num_station': station_coords,
                    'variable':[f'veg_emb_{i}' for i in range(embedding_dim)]},
            dims=['time','num_station','variable']
        )
        return vegetation_da


def idw_interpolation(grid_data, grid_lats, grid_lons, station_lats, station_lons, power=2.0, max_distance=None):
    """
    优化的IDW插值 - 使用KD树加速（10-50倍性能提升）
    """
    start_time = time.time()

    try:
        # 验证输入数据
        if grid_data.size == 0:
            logger.warning("输入的格点数据为空")
            return np.zeros((1, len(station_lats)))

        if len(grid_lats) == 0 or len(grid_lons) == 0:
            logger.warning("格点经纬度数据为空")
            return np.zeros((grid_data.shape[0], len(station_lats)))

        # 创建网格点坐标
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lons, grid_lats)
        grid_points = np.column_stack((grid_lat_2d.ravel(), grid_lon_2d.ravel()))
        station_points = np.column_stack((station_lats, station_lons))

        if grid_points.size == 0 or station_points.size == 0:
            logger.warning("坐标点数据为空")
            return np.zeros((grid_data.shape[0], len(station_lats)))

        # 【关键优化1】使用KD树进行空间索引
        # 确定最近邻数量
        n_neighbors = min(10, len(grid_points))  # 最多使用10个最近邻点
        if max_distance is None:
            max_distance = 5.0  # 默认5度

        interpolated_data = []

        for t in range(grid_data.shape[0]):
            grid_values = grid_data[t, :, :].ravel()

            # 检查有效数据
            finite_mask = np.isfinite(grid_values)
            valid_count = np.sum(finite_mask)

            if valid_count == 0:
                logger.warning(f"时间步{t}没有有效数据点")
                station_values = np.zeros(len(station_lats))
                interpolated_data.append(station_values)
                continue

            if valid_count < 4:
                logger.warning(f"时间步{t}有效数据点太少({valid_count}个)，使用均值填充")
                mean_value = np.nanmean(grid_values)
                if np.isnan(mean_value):
                    mean_value = 0.0
                station_values = np.full(len(station_lats), mean_value)
                interpolated_data.append(station_values)
                continue

            # 【关键优化2】只对有效点建立KD树
            valid_points = grid_points[finite_mask]
            valid_values = grid_values[finite_mask]

            # 构建KD树（一次性，不是对每个站点）
            tree = cKDTree(valid_points)

            # 【关键优化3】批量查询所有站点的最近邻
            actual_n_neighbors = min(n_neighbors, len(valid_points))
            distances, indices = tree.query(
                station_points,
                k=actual_n_neighbors,
                distance_upper_bound=max_distance,
                workers=-1  # 使用所有CPU核心
            )

            # 【关键优化4】向量化计算权重和插值
            station_values = np.zeros(len(station_lats))

            for i in range(len(station_lats)):
                # 过滤无效邻居（超出距离限制的）
                valid_neighbors = indices[i] < len(valid_points)

                if not np.any(valid_neighbors):
                    # 如果没有有效邻居，使用全局均值
                    station_values[i] = np.mean(valid_values)
                    continue

                neighbor_dists = distances[i][valid_neighbors]
                neighbor_idx = indices[i][valid_neighbors]

                # 处理非常近的点（<0.01度约1km）
                very_close = neighbor_dists < 0.01
                if np.any(very_close):
                    station_values[i] = np.mean(valid_values[neighbor_idx[very_close]])
                    continue

                # IDW插值
                neighbor_dists = np.maximum(neighbor_dists, 1e-10)  # 避免除零
                weights = 1.0 / (neighbor_dists ** power)
                weights = weights / np.sum(weights)

                station_values[i] = np.sum(weights * valid_values[neighbor_idx])

            interpolated_data.append(station_values)

        result = np.array(interpolated_data)

        # 最终验证
        if result.size == 0:
            logger.warning("插值结果为空，返回零数组")
            return np.zeros((grid_data.shape[0], len(station_lats)))

        # 检查并修复异常值
        finite_mask = np.isfinite(result)
        if not np.all(finite_mask):
            logger.warning(f"插值结果包含{np.sum(~finite_mask)}个非有限值，用零替换")
            result[~finite_mask] = 0.0

        elapsed = time.time() - start_time
        logger.debug(f"IDW插值完成，耗时{elapsed:.2f}秒，结果形状: {result.shape}, "
                    f"数据范围: [{np.min(result):.6f}, {np.max(result):.6f}]")

        return result

    except Exception as e:
        logger.error(f"IDW插值严重失败: {e}")
        return np.zeros((grid_data.shape[0] if grid_data.ndim > 0 else 1, len(station_lats)))


def process_ssrd_accumulation_for_stations(station_cumulative_data):
    """
    对已插值到站点的SSRD累积数据进行逐小时差值计算
    输入: list of (stations,) arrays，每个array代表一个时效的累积值
    输出: (time, stations) array，每行代表一个小时的辐射通量
    """
    try:
        if len(station_cumulative_data) <= 1:
            logger.warning("站点累积数据点太少，无法计算逐小时差值")
            return None

        logger.debug(f"处理站点SSRD累积数据，时间点数: {len(station_cumulative_data)}, 站点数: {len(station_cumulative_data[0])}")

        hourly_flux_data = []

        for i in range(1, len(station_cumulative_data)):
            # 计算第i小时的辐射通量
            current_cumulative = station_cumulative_data[i]
            previous_cumulative = station_cumulative_data[i-1]

            # 差值计算
            hourly_diff = current_cumulative - previous_cumulative

            # 转换为W/m²
            hourly_flux = hourly_diff / 3600.0

            # 确保非负
            hourly_flux = np.maximum(hourly_flux, 0)

            # 处理异常值
            finite_mask = np.isfinite(hourly_flux)
            if not np.all(finite_mask):
                invalid_count = np.sum(~finite_mask)
                logger.warning(f"第{i}小时有{invalid_count}个站点的SSRD值无效，设为0")
                hourly_flux[~finite_mask] = 0.0

            hourly_flux_data.append(hourly_flux)

        # 转换为numpy数组
        result = np.array(hourly_flux_data)  # shape: (time, stations)

        logger.debug(f"站点SSRD差值计算完成: 形状{result.shape}, 范围[{np.min(result):.3f}, {np.max(result):.3f}] W/m²")

        return result

    except Exception as e:
        logger.error(f"站点SSRD差值计算失败: {e}")
        return None

def load_and_interpolate_ssrd_data(base_dir, timestamp, station_lats, station_lons, forecast_hours=36):
    """
    SSRD数据处理优化版：先插值到站点，再进行逐小时累积差值计算
    大幅提升处理速度，特别是当站点数远少于格点数时
    """
    start_time = time.time()

    try:
        filepaths = build_grid_data_path(base_dir, timestamp, 'DSWRFsfc')
        if filepaths is None:
            logger.warning("无法构建SSRD文件路径列表")
            return None

        if len(filepaths) != forecast_hours + 1:  # 应该是37个文件
            logger.warning(f"SSRD文件数量不正确，期望{forecast_hours + 1}个，实际{len(filepaths)}个")
            return None

        logger.debug(f"准备读取{len(filepaths)}个SSRD文件进行优化处理")

        # 优化：先读取所有累积数据并插值到站点，再进行逐小时计算
        station_cumulative_data = []  # 存储每个时效插值到站点的累积数据
        grid_lats = None
        grid_lons = None

        logger.info(f"开始分时效插值SSRD累积数据到{len(station_lats)}个站点...")

        for i, filepath in enumerate(filepaths):
            try:
                if not os.path.exists(filepath):
                    # 用前一个时间点的值填充
                    if station_cumulative_data:
                        station_cumulative_data.append(station_cumulative_data[-1].copy())
                        logger.debug(f"时效{i}文件缺失，使用前一时间点数据填充")
                    else:
                        logger.warning(f"起报时间SSRD文件缺失，无法处理")
                        return None
                    continue

                ds = xr.open_dataset(filepath)

                # 获取变量
                data_vars = [v for v in ds.data_vars if len(ds[v].dims) >= 2]
                if not data_vars:
                    logger.warning(f"文件{filepath}中无有效数据变量")
                    ds.close()
                    continue

                data_var = data_vars[0]
                grid_data = ds[data_var]

                # 第一次读取时获取坐标
                if grid_lats is None:
                    lat_dim_names = ['latitude', 'lat', 'y', 'grid_yt']
                    lon_dim_names = ['longitude', 'lon', 'x', 'grid_xt']

                    lat_dim = lon_dim = None
                    for name in lat_dim_names:
                        if name in grid_data.dims:
                            lat_dim = name
                            break

                    for name in lon_dim_names:
                        if name in grid_data.dims:
                            lon_dim = name
                            break

                    if lat_dim is None or lon_dim is None:
                        logger.warning(f"无法识别SSRD文件的经纬度维度")
                        ds.close()
                        return None

                    grid_lats = ds[lat_dim].values
                    grid_lons = ds[lon_dim].values

                    # 处理经度范围
                    if np.any(grid_lons > 180):
                        grid_lons = np.where(grid_lons > 180, grid_lons - 360, grid_lons)
                        lon_sorted_idx = np.argsort(grid_lons)
                        grid_lons = grid_lons[lon_sorted_idx]
                        grid_data = grid_data.isel({lon_dim: lon_sorted_idx})

                # 转换为numpy并移除时间维度
                values = grid_data.values
                if values.ndim == 3:
                    values = values[0]  # 取第一个时间点
                elif values.ndim != 2:
                    logger.warning(f"时效{i}数据维度异常: {values.ndim}")
                    ds.close()
                    continue

                # 关键优化：直接插值单个时间点的2D数据到站点
                # 添加时间维度以适配IDW函数
                grid_values_3d = values[np.newaxis, :, :]  # (1, lat, lon)

                interpolated_values = idw_interpolation(
                    grid_values_3d, grid_lats, grid_lons, station_lats, station_lons,
                    power=2.0, max_distance=5.0
                )

                if interpolated_values is not None and interpolated_values.size > 0:
                    # 取出单个时间点的结果 (1, stations) -> (stations,)
                    station_values = interpolated_values[0, :]
                    station_cumulative_data.append(station_values)
                    logger.debug(f"时效{i}插值成功，站点累积值范围[{np.min(station_values):.0f}, {np.max(station_values):.0f}] J/m²")
                else:
                    logger.warning(f"时效{i}插值失败")
                    if station_cumulative_data:
                        station_cumulative_data.append(station_cumulative_data[-1].copy())
                    else:
                        ds.close()
                        return None

                ds.close()

            except Exception as e:
                logger.warning(f"读取SSRD文件{filepath}失败: {e}")
                if station_cumulative_data:
                    station_cumulative_data.append(station_cumulative_data[-1].copy())
                else:
                    return None

        if len(station_cumulative_data) < 2:
            logger.error("SSRD站点累积数据不足，无法计算逐小时差值")
            return None

        # 优化后的逐小时计算：只针对站点数据
        logger.info(f"开始对{len(station_lats)}个站点进行逐小时差值计算...")

        hourly_station_data = []

        for i in range(1, len(station_cumulative_data)):
            # 计算第i小时的辐射通量：累积(i) - 累积(i-1)
            cumulative_current = station_cumulative_data[i]
            cumulative_previous = station_cumulative_data[i-1]

            # 计算差值并转换为W/m²
            hourly_diff = cumulative_current - cumulative_previous
            hourly_flux = hourly_diff / 3600.0  # J/m² -> W/m²

            # 确保为正值
            hourly_flux = np.maximum(hourly_flux, 0)

            hourly_station_data.append(hourly_flux)

            if i % 12 == 0:  # 每12小时输出一次日志
                logger.debug(f"第{i}小时站点辐射通量范围: [{np.min(hourly_flux):.3f}, {np.max(hourly_flux):.3f}] W/m²")

        # 确保数据长度
        if len(hourly_station_data) < forecast_hours:
            need_padding = forecast_hours - len(hourly_station_data)
            for _ in range(need_padding):
                hourly_station_data.append(hourly_station_data[-1].copy())
            logger.warning(f"SSRD数据不足{forecast_hours}小时，用最后值填充{need_padding}个时间点")

        # 截取到目标长度并转换为最终格式
        hourly_station_data = hourly_station_data[:forecast_hours]

        # 转换为 (time, station) 格式的numpy数组
        interpolated_data = np.array(hourly_station_data)  # (time, station)

        elapsed = time.time() - start_time
        logger.info(f"SSRD优化处理完成，耗时{elapsed:.2f}秒，最终形状: {interpolated_data.shape}")
        logger.info(f"整体辐射通量范围: [{np.min(interpolated_data):.3f}, {np.max(interpolated_data):.3f}] W/m²")

        return interpolated_data

    except Exception as e:
        logger.error(f"SSRD优化处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_grid_data_path(base_dir, timestamp, var_name):
    """构建格点数据文件路径（支持返回时序文件列表）
    - 对于 DSWRFsfc 保持原有特殊处理（包含起报时刻，共 forecast_hours+1 个）
    - 对于其他单层变量（如 lcc/cldl）返回按时效排列的文件列表（比 SSRD 少读最后一小时，默认36小时）
    """
    date_str = timestamp.strftime('%Y%m%d')
    hour_str = f"t{timestamp.strftime('%H')}z"

    # 变量名映射
    var_mapping = {
        'cldl': 'lcc',
        'DSWRFsfc': 'ssrd',
    }

    if var_name not in var_mapping:
        return None

    file_var = var_mapping[var_name]
    init_time_part = timestamp.strftime('%m%d%H%M')

    # SSRD 的特殊处理保持不变（需要时效0）
    if var_name == 'DSWRFsfc':
        ssrd_files = []
        # 起报时间文件 (时效0，起报时间后1分钟)
        forecast_time_init = timestamp + timedelta(minutes=1)
        forecast_time_str = forecast_time_init.strftime('%m%d%H%M')
        filename_init = f"A5D{init_time_part}{forecast_time_str}1_ssrd.nc"
        filepath_init = os.path.join(base_dir, '0P125', date_str, hour_str, 'Single_level', file_var, filename_init)
        ssrd_files.append(filepath_init)
        # 1-36小时预报文件 (整点)
        for hour in range(1, 37):
            forecast_time = timestamp + timedelta(hours=hour)
            forecast_time_str = forecast_time.strftime('%m%d%H%M')
            filename = f"A5D{init_time_part}{forecast_time_str}1_ssrd.nc"
            filepath = os.path.join(base_dir, '0P125', date_str, hour_str, 'Single_level', file_var, filename)
            ssrd_files.append(filepath)
        return ssrd_files

    # 其他单层变量（如 lcc）——返回一组按时效排列的文件路径（默认36个时效，少于SSRD的37）
    prefixes = ['A4D', 'A5D']
    forecast_hours = 36  # 比SSRD少读取最后一小时（SSRD读取0..36共37个），这里读取0..35共36个
    files = []
    forecast_time = timestamp + timedelta(minutes=1)
    forecast_time_part = forecast_time.strftime('%m%d%H%M')
    for prefix in prefixes:
        filename = f"{prefix}{init_time_part}{forecast_time_part}1_{file_var}.nc"
        filepath = os.path.join(base_dir, '0P125', date_str, hour_str, 'Single_level', file_var, filename)
        if os.path.exists(filepath):
            files.append(filepath)
    for fh in range(1, forecast_hours):
        found = False
        forecast_time = timestamp + timedelta(hours=fh)
        forecast_time_part = forecast_time.strftime('%m%d%H%M')
        for prefix in prefixes:
            filename = f"{prefix}{init_time_part}{forecast_time_part}1_{file_var}.nc"
            filepath = os.path.join(base_dir, '0P125', date_str, hour_str, 'Single_level', file_var, filename)
            if os.path.exists(filepath):
                files.append(filepath)
                found = True
                break
        if not found:
            # 如果在任何前缀下都没找到，仍然返回首选前缀的推测路径（便于后续诊断/补齐）
            filename = f"{prefixes[0]}{init_time_part}{forecast_time_part}1_{file_var}.nc"
            filepath = os.path.join(base_dir, '0P125', date_str, hour_str, 'Single_level', file_var, filename)
            files.append(filepath)

    return files


def build_pressure_level_data_path(base_dir, timestamp, var_name):
    """构建气压层数据文件路径列表（修复时间处理问题）
    - 对于气压层变量（r, q 等）返回按时效排列的文件列表（默认36小时，少于SSRD最后一小时）
    - 文件名构建与SSRD类似：prefix + init_time_part + forecast_time_part + '1' + '_' + file_var + '.nc'
    """
    date_str = timestamp.strftime('%Y%m%d')
    hour_str = f"t{timestamp.strftime('%H')}z"

    # 变量名映射
    var_mapping = {
        'rh925': 'r',    # 相对湿度 -> r
        'q1000': 'q',    # 比湿 -> q
        'q925': 'q'      # 比湿 -> q
    }

    if var_name not in var_mapping:
        return None

    file_var = var_mapping[var_name]
    init_time_part = timestamp.strftime('%m%d%H%M')

    prefixes = ['A2D', 'A3D', 'A4D', 'A5D']  # 尝试更多可能的前缀
    forecast_hours = 36  # 比SSRD少1小时，读取0..35共36个文件

    file_list = []
    forecast_time = timestamp + timedelta(minutes=1)
    forecast_time_part = forecast_time.strftime('%m%d%H%M')
    for prefix in prefixes:
        filename = f"{prefix}{init_time_part}{forecast_time_part}1_{file_var}.nc"
        filepath = os.path.join(base_dir, '0P125', date_str, hour_str, 'Pressure_levels', file_var, filename)
        if os.path.exists(filepath):
            file_list.append(filepath)
    for fh in range(1, forecast_hours):
        forecast_time = timestamp + timedelta(hours=fh)
        forecast_time_part = forecast_time.strftime('%m%d%H%M')
        found = False
        for prefix in prefixes:
            filename = f"{prefix}{init_time_part}{forecast_time_part}1_{file_var}.nc"
            filepath = os.path.join(
                base_dir,
                '0P125',
                date_str,
                hour_str,
                'Pressure_levels',
                file_var,
                filename
            )
            if os.path.exists(filepath):
                file_list.append(filepath)
                found = True
                break
        if not found:
            # 当未找到时，仍将首选前缀的推测路径加入列表，便于后续补齐/诊断
            filename = f"{prefixes[0]}{init_time_part}{forecast_time_part}1_{file_var}.nc"
            filepath = os.path.join(
                base_dir,
                '0P125',
                date_str,
                hour_str,
                'Pressure_levels',
                file_var,
                filename
            )
            file_list.append(filepath)

    return file_list


def load_and_interpolate_grid_data(base_dir, timestamp, var_name, station_lats, station_lons, forecast_hours=36):
    """
    加载格点数据并使用IDW插值到站点（优化版）
    """
    start_time = time.time()

    try:
        if var_name == 'DSWRFsfc':
            return load_and_interpolate_ssrd_data(base_dir, timestamp, station_lats, station_lons, forecast_hours)

        filepaths = build_grid_data_path(base_dir, timestamp, var_name)
        if filepaths is None:
            logger.warning(f"无法构建{var_name}的文件路径")
            return None

        # 如果返回的是单个字符串（兼容旧逻辑）
        if isinstance(filepaths, str):
            filepath = filepaths
            if not os.path.exists(filepath):
                logger.warning(f"格点数据文件不存在: {filepath}")
                return None
            try:
                ds = xr.open_dataset(filepath)
            except Exception as e:
                logger.warning(f"无法打开NetCDF文件 {filepath}: {e}")
                return None

            data_vars = [v for v in ds.data_vars if len(ds[v].dims) >= 2]
            if not data_vars:
                logger.warning(f"未找到有效的数据变量: {filepath}")
                ds.close()
                return None
            data_var = data_vars[0]
            grid_data = ds[data_var]

            # 识别经纬度维度
            lat_dim_names = ['latitude', 'lat', 'y', 'grid_yt']
            lon_dim_names = ['longitude', 'lon', 'x', 'grid_xt']
            time_dim_names = ['time', 'valid_time', 't']
            lat_dim = lon_dim = time_dim = None
            for name in lat_dim_names:
                if name in grid_data.dims:
                    lat_dim = name
                    break
            for name in lon_dim_names:
                if name in grid_data.dims:
                    lon_dim = name
                    break
            for name in time_dim_names:
                if name in grid_data.dims:
                    time_dim = name
                    break

            if lat_dim is None or lon_dim is None:
                logger.warning(f"无法识别经纬度维度: {grid_data.dims}")
                ds.close()
                return None

            grid_lats = ds[lat_dim].values
            grid_lons = ds[lon_dim].values

            if np.any(grid_lons > 180):
                grid_lons = np.where(grid_lons > 180, grid_lons - 360, grid_lons)
                lon_sorted_idx = np.argsort(grid_lons)
                grid_lons = grid_lons[lon_sorted_idx]
                grid_data = grid_data.isel({lon_dim: lon_sorted_idx})

            # 处理时间切片（尽量取 forecast_hours 个时效，从时效1开始，除非只有一个时间点）
            if time_dim is not None:
                time_len = len(ds[time_dim])
                if time_len > forecast_hours:
                    grid_data = grid_data.isel({time_dim: slice(1, forecast_hours + 1)})
                elif time_len > 1:
                    grid_data = grid_data.isel({time_dim: slice(1, None)})

            try:
                grid_values = grid_data.values
                if grid_values.ndim == 2:
                    grid_values = grid_values[np.newaxis, :, :]
            except Exception as e:
                logger.warning(f"数据转换为numpy失败: {e}")
                ds.close()
                return None

            ds.close()

            # 特殊变量处理
            if var_name == 'cldl':
                grid_values = np.clip(grid_values, 0, 1)

            interpolated_data = idw_interpolation(
                grid_values, grid_lats, grid_lons, station_lats, station_lons,
                power=2.0, max_distance=5.0
            )

            if interpolated_data is not None and interpolated_data.size > 0:
                elapsed = time.time() - start_time
                logger.info(f"成功加载和插值{var_name}: 形状{interpolated_data.shape}, 耗时{elapsed:.2f}秒")
                return interpolated_data
            else:
                logger.warning(f"{var_name}插值结果为空")
                return None

        # 如果返回列表，则按时序读取多个文件并拼接成 (time, lat, lon)
        if isinstance(filepaths, (list, tuple)):
            grid_lats = grid_lons = None
            time_slices = []

            for i, fp in enumerate(filepaths):
                try:
                    if not os.path.exists(fp):
                        # 使用前一个时间点的值填充
                        if time_slices:
                            time_slices.append(time_slices[-1].copy())
                            logger.debug(f"时效{i}文件缺失，使用前一时间点数据填充: {fp}")
                        else:
                            logger.warning(f"起报时间文件缺失且无前值可用: {fp}")
                            return None
                        continue

                    ds = xr.open_dataset(fp)
                    data_vars = [v for v in ds.data_vars if len(ds[v].dims) >= 2]
                    if not data_vars:
                        logger.warning(f"文件{fp}中无有效数据变量")
                        ds.close()
                        if time_slices:
                            time_slices.append(time_slices[-1].copy())
                        else:
                            return None
                        continue
                    data_var = data_vars[0]
                    grid_data = ds[data_var]

                    # 首次获取坐标
                    if grid_lats is None or grid_lons is None:
                        lat_dim = next((n for n in ['latitude','lat','y','grid_yt'] if n in grid_data.dims), None)
                        lon_dim = next((n for n in ['longitude','lon','x','grid_xt'] if n in grid_data.dims), None)
                        if lat_dim is None or lon_dim is None:
                            logger.warning(f"无法识别经纬度维度: {grid_data.dims} in {fp}")
                            ds.close()
                            return None
                        grid_lats = ds[lat_dim].values
                        grid_lons = ds[lon_dim].values
                        if np.any(grid_lons > 180):
                            grid_lons = np.where(grid_lons > 180, grid_lons - 360, grid_lons)
                            lon_sorted_idx = np.argsort(grid_lons)
                            grid_lons = grid_lons[lon_sorted_idx]
                            grid_data = grid_data.isel({lon_dim: lon_sorted_idx})

                    values = grid_data.values
                    # 如果有时间维度，取第一个时间切片
                    if values.ndim == 3:
                        values = values[0]
                    elif values.ndim != 2:
                        logger.warning(f"时效{i}数据维度异常 ({values.ndim}) in {fp}")
                        ds.close()
                        continue

                    time_slices.append(values)
                    ds.close()

                except Exception as e:
                    logger.warning(f"读取格点文件{fp}失败: {e}")
                    if time_slices:
                        time_slices.append(time_slices[-1].copy())
                    else:
                        return None

            if not time_slices:
                logger.warning(f"未能读取到任何时效的格点数据: {var_name}")
                return None

            grid_values = np.stack(time_slices, axis=0)  # (time, lat, lon)

            # 变量特殊处理
            if var_name == 'cldl':
                grid_values = np.clip(grid_values, 0, 1)

            interpolated_data = idw_interpolation(
                grid_values, grid_lats, grid_lons, station_lats, station_lons,
                power=2.0, max_distance=5.0
            )

            if interpolated_data is not None and interpolated_data.size > 0:
                elapsed = time.time() - start_time
                logger.info(f"成功加载和插值{var_name}（多文件）: 形状{interpolated_data.shape}, 耗时{elapsed:.2f}秒")
                return interpolated_data
            else:
                logger.warning(f"{var_name}插值结果为空（多文件）")
                return None

        logger.warning(f"未知的文件路径类型: {type(filepaths)}")
        return None

    except Exception as e:
        logger.error(f"加载格点数据严重失败{var_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_and_interpolate_pressure_level_data(base_dir, timestamp, var_name, station_lats, station_lons, forecast_hours=36):
    """
    加载气压层数据并插值到站点（优化版）
    """
    start_time = time.time()

    try:
        filepaths = build_pressure_level_data_path(base_dir, timestamp, var_name)
        if filepaths is None:
            logger.warning(f"无法构建{var_name}的气压层文件路径")
            return None

        # 如果单个字符串（兼容旧逻辑）
        if isinstance(filepaths, str):
            filepath = filepaths
            if not os.path.exists(filepath):
                logger.warning(f"气压层数据文件不存在: {filepath}")
                return None
            # 保持原有单文件处理逻辑（参照以前实现）
            filepath_list = [filepath]
        else:
            filepath_list = list(filepaths)

        grid_lats = grid_lons = None
        selected_level_idx = None
        time_slices = []
        target_level_map = {
            'rh925': 925.0,
            'q1000': 1000.0,
            'q925': 925.0
        }
        if var_name not in target_level_map:
            logger.warning(f"未知的气压层变量: {var_name}")
            return None
        target_level = target_level_map[var_name]

        for i, fp in enumerate(filepath_list):
            try:
                if not os.path.exists(fp):
                    if time_slices:
                        time_slices.append(time_slices[-1].copy())
                        logger.debug(f"时效{i}气压层文件缺失，使用前一时间点数据填充: {fp}")
                    else:
                        logger.warning(f"起报时间气压层文件缺失且无前值可用: {fp}")
                        return None
                    continue

                ds = xr.open_dataset(fp)
                data_vars = [v for v in ds.data_vars if len(ds[v].dims) >= 3]
                if not data_vars:
                    logger.warning(f"未找到有效的气压层数据变量: {fp}")
                    ds.close()
                    if time_slices:
                        time_slices.append(time_slices[-1].copy())
                    else:
                        return None
                    continue
                data_var = data_vars[0]
                grid_data = ds[data_var]

                # 首次获取坐标与层信息
                if grid_lats is None or grid_lons is None or selected_level_idx is None:
                    lat_dim = next((n for n in ['lat','latitude','y'] if n in grid_data.dims), None)
                    lon_dim = next((n for n in ['lon','longitude','x'] if n in grid_data.dims), None)
                    level_dim = next((n for n in ['level','plev','pressure_level','lev'] if n in grid_data.dims), None)
                    time_dim = next((n for n in ['time','valid_time','t'] if n in grid_data.dims), None)
                    if lat_dim is None or lon_dim is None or level_dim is None:
                        logger.warning(f"无法识别气压层数据的必要维度: {grid_data.dims} in {fp}")
                        ds.close()
                        return None
                    grid_lats = ds[lat_dim].values
                    grid_lons = ds[lon_dim].values
                    levels = ds[level_dim].values
                    # 找到最接近的层
                    level_diff = np.abs(levels - target_level)
                    selected_level_idx = int(np.argmin(level_diff))
                    actual_level = levels[selected_level_idx]
                    if abs(actual_level - target_level) > 50:
                        logger.warning(f"{var_name}: 目标气压层{target_level}hPa不存在，使用最接近的{actual_level}hPa")
                    ds_sel = grid_data.isel({level_dim: selected_level_idx})
                    # 处理经度
                    if np.any(grid_lons > 180):
                        grid_lons = np.where(grid_lons > 180, grid_lons - 360, grid_lons)
                        lon_sorted_idx = np.argsort(grid_lons)
                        grid_lons = grid_lons[lon_sorted_idx]
                        ds_sel = ds_sel.isel({lon_dim: lon_sorted_idx})
                else:
                    # 尝试按已确定的 level_dim 名称选择（如果层结构一致）
                    # 这里安全起见重新识别层维度名并选取最接近的索引（如果不同文件层次顺序略有差异）
                    level_dim_candidate = next((n for n in ['level','plev','pressure_level','lev'] if n in grid_data.dims), None)
                    if level_dim_candidate is not None:
                        levels = ds[level_dim_candidate].values
                        idx = int(np.argmin(np.abs(levels - target_level)))
                        ds_sel = grid_data.isel({level_dim_candidate: idx})
                    else:
                        logger.warning(f"文件{fp}未包含层维度，跳过")
                        ds.close()
                        if time_slices:
                            time_slices.append(time_slices[-1].copy())
                        else:
                            return None
                        continue

                values = ds_sel.values
                # 如果有时间维度，取第一个时间切片
                if values.ndim == 3:
                    values = values[0]
                elif values.ndim != 2:
                    logger.warning(f"气压层时效{i}数据维度异常 ({values.ndim}) in {fp}")
                    ds.close()
                    continue

                time_slices.append(values)
                ds.close()

            except Exception as e:
                logger.warning(f"读取气压层文件{fp}失败: {e}")
                if time_slices:
                    time_slices.append(time_slices[-1].copy())
                else:
                    return None

        if not time_slices:
            logger.warning(f"未能读取到任何气压层时效数据: {var_name}")
            return None

        grid_values = np.stack(time_slices, axis=0)  # (time, lat, lon)

        # 变量范围修正
        if var_name.startswith('rh'):
            grid_values = np.clip(grid_values, 0, 100)
        elif var_name.startswith('q'):
            grid_values = np.maximum(grid_values, 0)

        interpolated_data = idw_interpolation(
            grid_values, grid_lats, grid_lons, station_lats, station_lons,
            power=2.0, max_distance=5.0
        )

        if interpolated_data is not None and interpolated_data.size > 0:
            elapsed = time.time() - start_time
            logger.info(f"成功加载和插值气压层数据{var_name}: 形状{interpolated_data.shape}, 耗时{elapsed:.2f}秒")
            return interpolated_data
        else:
            logger.warning(f"气压层数据{var_name}插值结果为空")
            return None

    except Exception as e:
        logger.error(f"加载气压层数据严重失败{var_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_and_interpolate_slp_from_station_dir(data_dir, station_lats, station_lons, target_time_len):
    """
    从站点数据目录读取slp格点数据并插值
    """
    try:
        slp_filepath = os.path.join(data_dir, 'slp.nc')
        if not os.path.exists(slp_filepath):
            logger.warning(f"SLP文件不存在: {slp_filepath}")
            return None

        logger.debug(f"读取SLP格点文件: {slp_filepath}")

        # 读取slp格点数据
        ds = xr.open_dataset(slp_filepath)

        # 获取变量名 - 寻找slp变量
        if 'slp' not in ds.data_vars:
            logger.warning(f"SLP文件中未找到slp变量")
            ds.close()
            return None

        grid_data = ds['slp']

        logger.debug(f"SLP数据变量: slp, 维度: {grid_data.dims}, 形状: {grid_data.shape}")

        # 根据已知的维度结构，直接使用固定的维度名称
        # SLP数据结构: (time, grid_yt, grid_xt) 或类似结构
        lat_dim = 'grid_yt'
        lon_dim = 'grid_xt'
        time_dim = 'time'

        # 验证维度是否存在
        missing_dims = []
        if lat_dim not in grid_data.dims:
            missing_dims.append(lat_dim)
        if lon_dim not in grid_data.dims:
            missing_dims.append(lon_dim)
        if time_dim not in grid_data.dims:
            missing_dims.append(time_dim)

        if missing_dims:
            logger.warning(f"SLP数据缺少必要维度: {missing_dims}, 实际维度: {list(grid_data.dims)}")
            ds.close()
            return None

        # 获取坐标值
        grid_lats = ds[lat_dim].values
        grid_lons = ds[lon_dim].values

        # 处理经度范围
        if np.any(grid_lons > 180):
            grid_lons = np.where(grid_lons > 180, grid_lons - 360, grid_lons)
            lon_sorted_idx = np.argsort(grid_lons)
            grid_lons = grid_lons[lon_sorted_idx]
            grid_data = grid_data.isel({lon_dim: lon_sorted_idx})

        # 时间维度处理
        if time_dim is not None:
            time_len = len(ds[time_dim])
            if time_len > 1:
                grid_data = grid_data.isel({time_dim: slice(1, None)})  # 从预报时效1开始

        # 转换为numpy
        grid_values = grid_data.values

        # 确保维度顺序 (time, lat, lon)
        if time_dim is not None:
            dims = list(grid_data.dims)
            if len(dims) == 3:
                try:
                    time_idx = dims.index(time_dim)
                    lat_idx = dims.index(lat_dim)
                    lon_idx = dims.index(lon_dim)

                    if time_idx != 0 or lat_idx != 1 or lon_idx != 2:
                        axis_order = [time_idx, lat_idx, lon_idx]
                        grid_values = np.transpose(grid_values, axis_order)
                except:
                    pass
        else:
            grid_values = grid_values[np.newaxis, :, :]

        # IDW插值
        interpolated_data = idw_interpolation(
            grid_values, grid_lats, grid_lons, station_lats, station_lons,
            power=2.0, max_distance=5.0
        )

        ds.close()

        if interpolated_data is not None:
            # 时间维度对齐
            if interpolated_data.shape[0] != target_time_len:
                if interpolated_data.shape[0] > target_time_len:
                    interpolated_data = interpolated_data[:target_time_len, :]
                elif interpolated_data.shape[0] < target_time_len:
                    need_padding = target_time_len - interpolated_data.shape[0]
                    last_values = interpolated_data[-1:, :]
                    padding = np.tile(last_values, (need_padding, 1))
                    interpolated_data = np.concatenate([interpolated_data, padding], axis=0)

            logger.info(f"成功从站点目录加载和插值SLP: 形状{interpolated_data.shape}, "
                       f"范围[{np.nanmin(interpolated_data):.1f}, {np.nanmax(interpolated_data):.1f}] hPa")

        return interpolated_data

    except Exception as e:
        logger.error(f"从站点目录加载SLP数据失败: {e}")
        return None

def add_additional_features(ds_merged, vegetation_data=None, grid_data_base_dir=None, timestamp=None, data_dir=None):
    """
    添加天顶角、经纬度、植被嵌入、格点数据等额外特征（优化版）
    """
    start_time = time.time()

    latitudes = ds_merged['station_lat'].values
    longitudes = ds_merged['station_lon'].values
    time_coords = ds_merged.coords['time']
    station_coords = ds_merged.coords['num_station']
    target_time_len = len(time_coords)

    # 获取参考坐标系统 - 从ds_merged中提取完整的坐标信息
    reference_coords = {
        'time': time_coords,
        'num_station': station_coords,
        'station_lat': ds_merged.coords['station_lat'],
        'station_lon': ds_merged.coords['station_lon']
    }

    # 基础气象特征顺序 - 修改为包含风相关变量
    ordered_feature_vars = [
        'rh2m', 'TMP2m', 'prer', 'slp', 'DSWRFsfc',
        'UGRD10m', 'wind_speed', 'VGRD10m', 'wd10m', 'cape', 'cldl', 't925', 'rh925',
        'u925', 'wind_speed_925', 'v925', 'dp1000', 'dp925', 'q1000', 'q925','omg925','omg1000','gust'
    ]

    feature_arrays = []

    logger.info(f"开始添加{len(ordered_feature_vars)}个特征...")

    # 处理基础特征
    for idx, var in enumerate(ordered_feature_vars):
        if (idx + 1) % 5 == 0:
            logger.info(f"处理特征进度: {idx + 1}/{len(ordered_feature_vars)}")

        if var in ['cldl', 'DSWRFsfc'] and grid_data_base_dir is not None and timestamp is not None:
            # cldl和DSWRFsfc从格点数据获取
            interpolated_data = load_and_interpolate_grid_data(
                grid_data_base_dir, timestamp, var, latitudes, longitudes
            )

            if interpolated_data is not None:
                # 时间维度对齐处理
                if interpolated_data.shape[0] != target_time_len:
                    if interpolated_data.shape[0] > target_time_len:
                        interpolated_data = interpolated_data[:target_time_len, :]
                    elif interpolated_data.shape[0] < target_time_len:
                        need_padding = target_time_len - interpolated_data.shape[0]
                        last_values = interpolated_data[-1:, :]
                        padding = np.tile(last_values, (need_padding, 1))
                        interpolated_data = np.concatenate([interpolated_data, padding], axis=0)

                # 创建具有完整坐标信息的xarray DataArray
                var_data = xr.DataArray(
                    interpolated_data,
                    coords=reference_coords,  # 使用参考坐标系统
                    dims=['time', 'num_station']
                )
            else:
                # 如果格点数据加载失败，创建零填充但保持坐标一致
                var_data = xr.zeros_like(ds_merged[list(ds_merged.data_vars.keys())[0]])
                var_data = var_data.assign_coords(reference_coords)  # 确保坐标一致
                logger.warning(f"格点数据 {var} 加载失败，使用零填充")

            var_data = var_data.expand_dims('variable').assign_coords(variable=[var])
            feature_arrays.append(var_data)

        elif var in ['wind_speed', 'wd10m', 'wind_speed_925']:
            # 处理计算的风相关变量
            if var == 'wind_speed':
                # 计算10m风速
                if 'UGRD10m' in ds_merged.data_vars and 'VGRD10m' in ds_merged.data_vars:
                    u_wind = ds_merged['UGRD10m']
                    v_wind = ds_merged['VGRD10m']
                    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
                    wind_speed = wind_speed.assign_coords(reference_coords)
                else:
                    wind_speed = xr.zeros_like(ds_merged[list(ds_merged.data_vars.keys())[0]])
                    wind_speed = wind_speed.assign_coords(reference_coords)
                    logger.warning("缺少UGRD10m或VGRD10m，风速使用零填充")

                wind_speed = wind_speed.expand_dims('variable').assign_coords(variable=[var])
                feature_arrays.append(wind_speed)

            elif var == 'wd10m':
                # 计算10m风向
                if 'UGRD10m' in ds_merged.data_vars and 'VGRD10m' in ds_merged.data_vars:
                    u_wind = ds_merged['UGRD10m']
                    v_wind = ds_merged['VGRD10m']
                    # 风向计算 (从北顺时针，以度为单位)
                    wind_dir = (270 - np.arctan2(v_wind, u_wind) * 180 / np.pi) % 360
                    wind_dir = wind_dir.assign_coords(reference_coords)
                else:
                    wind_dir = xr.zeros_like(ds_merged[list(ds_merged.data_vars.keys())[0]])
                    wind_dir = wind_dir.assign_coords(reference_coords)
                    logger.warning("缺少UGRD10m或VGRD10m，风向使用零填充")

                wind_dir = wind_dir.expand_dims('variable').assign_coords(variable=[var])
                feature_arrays.append(wind_dir)

            elif var == 'wind_speed_925':
                # 计算925hPa风速
                if 'u925' in ds_merged.data_vars and 'v925' in ds_merged.data_vars:
                    u_wind_925 = ds_merged['u925']
                    v_wind_925 = ds_merged['v925']
                    wind_speed_925 = np.sqrt(u_wind_925**2 + v_wind_925**2)
                    wind_speed_925 = wind_speed_925.assign_coords(reference_coords)
                else:
                    wind_speed_925 = xr.zeros_like(ds_merged[list(ds_merged.data_vars.keys())[0]])
                    wind_speed_925 = wind_speed_925.assign_coords(reference_coords)
                    logger.warning("缺少u925或v925，925hPa风速使用零填充")

                wind_speed_925 = wind_speed_925.expand_dims('variable').assign_coords(variable=[var])
                feature_arrays.append(wind_speed_925)

        elif var == 'slp' and data_dir is not None:
            # slp从站点目录的格点文件获取
            interpolated_data = load_and_interpolate_slp_from_station_dir(
                data_dir, latitudes, longitudes, target_time_len
            )

            if interpolated_data is not None:
                # 创建具有完整坐标信息的xarray DataArray
                var_data = xr.DataArray(
                    interpolated_data,
                    coords=reference_coords,  # 使用参考坐标系统
                    dims=['time', 'num_station']
                )
            else:
                # 如果从站点目录读取失败，尝试从格点数据目录读取
                if grid_data_base_dir is not None and timestamp is not None:
                    interpolated_data = load_and_interpolate_grid_data(
                        grid_data_base_dir, timestamp, var, latitudes, longitudes
                    )
                    if interpolated_data is not None:
                        # 时间对齐处理
                        if interpolated_data.shape[0] != target_time_len:
                            if interpolated_data.shape[0] > target_time_len:
                                interpolated_data = interpolated_data[:target_time_len, :]
                            elif interpolated_data.shape[0] < target_time_len:
                                need_padding = target_time_len - interpolated_data.shape[0]
                                last_values = interpolated_data[-1:, :]
                                padding = np.tile(last_values, (need_padding, 1))
                                interpolated_data = np.concatenate([interpolated_data, padding], axis=0)

                        var_data = xr.DataArray(
                            interpolated_data,
                            coords=reference_coords,  # 使用参考坐标系统
                            dims=['time', 'num_station']
                        )
                    else:
                        var_data = xr.zeros_like(ds_merged[list(ds_merged.data_vars.keys())[0]])
                        var_data = var_data.assign_coords(reference_coords)
                        logger.warning(f"SLP数据加载失败，使用零填充")
                else:
                    var_data = xr.zeros_like(ds_merged[list(ds_merged.data_vars.keys())[0]])
                    var_data = var_data.assign_coords(reference_coords)
                    logger.warning(f"SLP数据不可用，使用零填充")

            var_data = var_data.expand_dims('variable').assign_coords(variable=[var])
            feature_arrays.append(var_data)

        elif var in ['rh925', 'q1000', 'q925'] and grid_data_base_dir is not None and timestamp is not None:
            # 处理气压层数据
            interpolated_data = load_and_interpolate_pressure_level_data(
                grid_data_base_dir, timestamp, var, latitudes, longitudes
            )

            if interpolated_data is not None:
                # 时间维度对齐处理
                if interpolated_data.shape[0] != target_time_len:
                    if interpolated_data.shape[0] > target_time_len:
                        interpolated_data = interpolated_data[:target_time_len, :]
                    elif interpolated_data.shape[0] < target_time_len:
                        need_padding = target_time_len - interpolated_data.shape[0]
                        last_values = interpolated_data[-1:, :]
                        padding = np.tile(last_values, (need_padding, 1))
                        interpolated_data = np.concatenate([interpolated_data, padding], axis=0)

                # 创建具有完整坐标信息的xarray DataArray
                var_data = xr.DataArray(
                    interpolated_data,
                    coords=reference_coords,  # 使用参考坐标系统
                    dims=['time', 'num_station']
                )
            else:
                # 如果气压层数据加载失败，创建零填充
                var_data = xr.zeros_like(ds_merged[list(ds_merged.data_vars.keys())[0]])
                var_data = var_data.assign_coords(reference_coords)
                logger.warning(f"气压层数据 {var} 加载失败，使用零填充")

            var_data = var_data.expand_dims('variable').assign_coords(variable=[var])
            feature_arrays.append(var_data)

        elif var in ds_merged.data_vars:
            # 处理站点数据 - 确保坐标一致
            var_data = ds_merged[var]
            # 确保坐标完全一致
            var_data = var_data.assign_coords(reference_coords)
            var_data = var_data.expand_dims('variable').assign_coords(variable=[var])
            feature_arrays.append(var_data)
        else:
            # 创建零填充的占位符，使用参考坐标系统
            placeholder = xr.zeros_like(ds_merged[list(ds_merged.data_vars.keys())[0]])
            placeholder = placeholder.assign_coords(reference_coords)
            placeholder = placeholder.expand_dims('variable').assign_coords(variable=[var])
            feature_arrays.append(placeholder)
            logger.warning(f"变量 {var} 不存在，使用零填充")

    logger.info("合并特征数组...")
    features = xr.concat(feature_arrays, dim='variable')

    # 添加天顶角 - 使用参考坐标系统
    logger.info("计算天顶角...")
    zenith_angles = calculate_zenith_angle(latitudes, longitudes, time_coords.values)
    zenith_da = xr.DataArray(
        zenith_angles,
        coords=reference_coords,  # 使用参考坐标系统
        dims=['time','num_station']
    ).expand_dims(dim={'variable':['zenith']}, axis=-1)
    features = xr.concat([features, zenith_da], dim='variable')

    # 添加经纬度 - 使用参考坐标系统
    logger.info("添加经纬度特征...")
    latitudes_broadcasted = np.tile(latitudes[:,np.newaxis],(1,len(time_coords))).T
    longitudes_broadcasted = np.tile(longitudes[:,np.newaxis],(1,len(time_coords))).T

    # 为经纬度数据创建带有参考坐标的DataArray
    lat_coords = reference_coords.copy()
    lat_coords['variable'] = ['lat']
    lat_da = xr.DataArray(
        latitudes_broadcasted[:,:,np.newaxis],
        coords=lat_coords,
        dims=['time','num_station','variable']
    )

    lon_coords = reference_coords.copy()
    lon_coords['variable'] = ['lon']
    lon_da = xr.DataArray(
        longitudes_broadcasted[:,:,np.newaxis],
        coords=lon_coords,
        dims=['time','num_station','variable']
    )

    features = xr.concat([features, lat_da, lon_da], dim='variable')

    # 添加植被嵌入 - 使用参考坐标系统
    if vegetation_data is not None:
        try:
            logger.info("创建植被嵌入...")
            vegetation_da = create_vegetation_embeddings(latitudes, longitudes, vegetation_data, time_coords, station_coords)
            # 确保植被嵌入数据也有完整的坐标信息
            veg_coords = reference_coords.copy()
            veg_coords['variable'] = [f'veg_emb_{i}' for i in range(vegetation_da.shape[2])]
            vegetation_da = vegetation_da.assign_coords(veg_coords)
            features = xr.concat([features, vegetation_da], dim='variable')
        except Exception as e:
            logger.warning(f"植被嵌入创建失败: {e}")
            # 创建零填充的植被嵌入
            embedding_dim = 8
            zero_embeddings = np.zeros((len(time_coords), len(station_coords), embedding_dim))
            veg_coords = reference_coords.copy()
            veg_coords['variable'] = [f'veg_emb_{i}' for i in range(embedding_dim)]
            vegetation_da = xr.DataArray(
                zero_embeddings,
                coords=veg_coords,
                dims=['time','num_station','variable']
            )
            features = xr.concat([features, vegetation_da], dim='variable')

    elapsed = time.time() - start_time
    logger.info(f"特征添加完成，耗时{elapsed:.2f}秒，最终特征数: {len(features.variable)}")

    return features

def check_data_completeness(data_dir):
    """检查数据目录的完整性 - 修复版本"""
    required_files = [
        'dp_plev.CH.nc',
        'omg_plev.CH.nc',
        'sfc.CH.nc',
        'slp.nc',  # slp格点数据文件
        't_plev.CH.nc',
        'u_plev.CH.nc',
        'v_plev.CH.nc'
    ]

    missing_files = []
    incomplete_files = []  # 添加这个列表

    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
        else:
            # 可以添加文件大小检查等
            try:
                file_size = os.path.getsize(filepath)
                if file_size == 0:
                    incomplete_files.append(f"{filename} (empty)")
            except Exception as e:
                incomplete_files.append(f"{filename} (access error: {e})")

    return {
        'complete': len(missing_files) == 0 and len(incomplete_files) == 0,
        'missing_files': missing_files,
        'incomplete_files': incomplete_files  # 确保返回这个字段
    }

def parse_timestamp_from_dirname(dirname):
    """从目录名解析时间戳"""
    try:
        return pd.to_datetime(dirname, format='%Y%m%d%H')
    except:
        return None

def convert_visibility_to_class(visibility, threshold1=0.5, threshold2=1.0):
    """将能见度转换为分类标签"""
    return np.select(
        [visibility < threshold1, visibility < threshold2, visibility >= threshold2],
        [0, 1, 2],
        default=2
    )

# ------------------------------
# 数据处理类
# ------------------------------
class DataProcessor:
    def __init__(self, vegetation_data=None, grid_data_base_dir=None):
        self.vegetation_data = vegetation_data
        self.grid_data_base_dir = grid_data_base_dir

    def merge_single_timestep_data(self, data_dir, timestamp):
        """合并单个时间点的数据 - 修复坐标不一致问题（增强版）"""
        overall_start = time.time()

        try:
            # 定义文件路径
            files_mapping = {
                'dp': 'dp_plev.CH.nc',
                'omg': 'omg_plev.CH.nc',
                'sfc': 'sfc.CH.nc',
                't': 't_plev.CH.nc',
                'u': 'u_plev.CH.nc',
                'v': 'v_plev.CH.nc'
            }

            datasets = {}
            missing_files = []

            logger.info(f"开始读取站点数据文件...")
            # 检查并加载文件
            for key, filename in files_mapping.items():
                filepath = os.path.join(data_dir, filename)
                if os.path.exists(filepath):
                    try:
                        datasets[key] = xr.open_dataset(filepath)
                        logger.debug(f"成功加载文件: {filename}")
                    except Exception as e:
                        logger.warning(f"无法读取文件 {filepath}: {e}")
                        missing_files.append(filename)
                else:
                    missing_files.append(filename)

            if missing_files:
                logger.warning(f"{data_dir} 缺少文件: {missing_files}")

            if len(datasets) == 0:
                logger.error(f"{data_dir} 没有可用的数据文件")
                return None

            # 第1步：获取统一的参考坐标系统
            reference_coords = None
            reference_dims = None

            # 优先使用sfc文件作为参考，因为它通常最完整
            coord_priority = ['sfc', 'dp', 't', 'u', 'v', 'omg']

            for key in coord_priority:
                if key in datasets:
                    ds = datasets[key]
                    if 'station_lat' in ds.coords and 'station_lon' in ds.coords:
                        reference_coords = {
                            'station_lat': ds.coords['station_lat'],
                            'station_lon': ds.coords['station_lon']
                        }
                        # 同时获取其他重要坐标
                        if 'time' in ds.coords:
                            reference_coords['time'] = ds.coords['time']
                        if 'num_station' in ds.coords:
                            reference_coords['num_station'] = ds.coords['num_station']

                        reference_dims = ds.dims
                        logger.info(f"使用 {key} 作为参考坐标系，站点数: {len(ds.coords['station_lat'])}")
                        break

            if reference_coords is None:
                logger.error("无法找到有效的参考坐标系")
                for ds in datasets.values():
                    ds.close()
                return None

            # 第2步：统一所有变量的坐标系统
            variable_mapping = {
                'dp': ['dp1000', 'dp925'],
                'omg': ['omg1000', 'omg925'],
                'sfc': ['cape', 'gust', 'prer', 'rh2m', 'TMP2m', 'UGRD10m', 'VGRD10m'],
                't': ['t925'],
                'u': ['u925'],
                'v': ['v925']
            }

            unified_data_vars = {}

            for ds_key, var_list in variable_mapping.items():
                if ds_key not in datasets:
                    logger.warning(f"缺少数据集: {ds_key}")
                    continue

                ds = datasets[ds_key]

                for var_name in var_list:
                    if var_name in ds.data_vars:
                        try:
                            var_data = ds[var_name]

                            # 关键修复：强制重新分配所有坐标
                            # 确保数据维度与参考坐标匹配
                            if var_data.dims == tuple(reference_dims.keys()):
                                # 维度完全匹配，直接分配参考坐标
                                unified_var = var_data.assign_coords(reference_coords)
                            else:
                                # 维度不匹配，尝试部分重新分配
                                new_coords = {}
                                for dim in var_data.dims:
                                    if dim in reference_coords:
                                        new_coords[dim] = reference_coords[dim]

                                if len(new_coords) >= 2:  # 至少有station和time坐标
                                    unified_var = var_data.assign_coords(new_coords)
                                else:
                                    logger.warning(f"变量 {var_name} 维度不兼容，跳过")
                                    continue

                            # 额外验证：确保关键坐标存在且一致
                            if 'station_lat' in unified_var.coords and 'station_lon' in unified_var.coords:
                                # 检查坐标数组是否完全相同（不仅仅是形状）
                                if not np.array_equal(unified_var.coords['station_lat'].values,
                                                    reference_coords['station_lat'].values):
                                    logger.warning(f"变量 {var_name} 的station_lat坐标不一致，强制对齐")
                                    unified_var = unified_var.assign_coords(station_lat=reference_coords['station_lat'])

                                if not np.array_equal(unified_var.coords['station_lon'].values,
                                                    reference_coords['station_lon'].values):
                                    logger.warning(f"变量 {var_name} 的station_lon坐标不一致，强制对齐")
                                    unified_var = unified_var.assign_coords(station_lon=reference_coords['station_lon'])

                            unified_data_vars[var_name] = unified_var
                            logger.debug(f"成功统一变量: {var_name} from {ds_key}, 形状: {unified_var.shape}")

                        except Exception as e:
                            logger.warning(f"处理变量 {var_name} from {ds_key} 失败: {e}")
                    else:
                        logger.warning(f"变量 {var_name} 不存在于 {ds_key}")

            if not unified_data_vars:
                logger.error(f"{data_dir} 没有可合并的变量")
                for ds in datasets.values():
                    ds.close()
                return None

            logger.info(f"准备合并 {len(unified_data_vars)} 个统一坐标的变量")

            # 第3步：创建统一的数据集
            try:
                # 使用统一坐标的变量创建数据集
                ds_merged = xr.Dataset(unified_data_vars)

                # 确保数据集有完整的坐标信息
                ds_merged = ds_merged.assign_coords(reference_coords)

                logger.info("数据集合并成功")

            except Exception as e:
                logger.error(f"数据集创建失败: {e}")
                for ds in datasets.values():
                    ds.close()
                return None

            # 确保时间维度正确（从起报时间后1小时开始，共36小时）
            if 'time' in ds_merged.dims:
                current_time_len = len(ds_merged['time'])
                if current_time_len > 36:
                    ds_merged = ds_merged.isel(time=slice(0, 36))
                    logger.info(f"截取时间维度从 {current_time_len} 到 36")
                elif current_time_len < 36:
                    logger.warning(f"时间维度不足: 只有{current_time_len}个时间点，期望36个")

            # 输出合并后的信息
            logger.info(f"数据合并成功:")
            logger.info(f"  数据变量: {list(ds_merged.data_vars.keys())}")
            logger.info(f"  维度: {dict(ds_merged.dims)}")

            # 添加额外特征（包括从slp.nc文件读取格点数据并插值）
            logger.info("开始添加额外特征（格点数据插值等）...")
            features = add_additional_features(ds_merged, self.vegetation_data, self.grid_data_base_dir, timestamp, data_dir)

            # 关闭数据集
            for ds in datasets.values():
                ds.close()

            elapsed = time.time() - overall_start
            logger.info(f"单时间步数据处理完成，总耗时{elapsed:.2f}秒")

            return features

        except Exception as e:
            logger.error(f"{data_dir} 合并失败: {e}")
            import traceback
            traceback.print_exc()

            # 确保关闭所有打开的数据集
            try:
                for ds in datasets.values():
                    ds.close()
            except:
                pass

            return None

# ------------------------------
# 预报系统类
# ------------------------------
class VisibilityForecastSystem:
    def __init__(self, config_path):
        """初始化预报系统"""
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessors = None
        self.data_processor = None
        self.model_type = self.config.get("model_type", "legacy_mlp")
        self.airport_thresholds = {"fog": 0.5, "mist": 0.5}
        self.airport_temperature = 1.0
        self.airport_station_order = None
        self.airport_season_thresholds = None
        self.processed_dirs = set()
        self.failed_dirs = {}
        self.forecasted_timestamps = set()  # 新增：记录已预报的时间戳

        logger.info(f"系统初始化，使用设备: {self.device}")

        # 新增：加载已预报的时间戳
        self.load_forecasted_timestamps()

    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_model(self):
        """加载模型和预处理器"""
        try:
            preprocessor_path = self.config.get('preprocessor_path')
            if preprocessor_path and os.path.exists(preprocessor_path):
                try:
                    self.preprocessors = joblib.load(preprocessor_path)
                except Exception:
                    with open(preprocessor_path, 'rb') as f:
                        self.preprocessors = pickle.load(f)
            else:
                self.preprocessors = None

            try:
                checkpoint = torch.load(
                    self.config['model_path'],
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(self.config['model_path'], map_location=self.device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            else:
                model_state = checkpoint

            model_config = {}
            if isinstance(checkpoint, dict):
                model_config = checkpoint.get('model_config', checkpoint.get('config', {}))
            if not model_config and isinstance(self.preprocessors, dict):
                model_config = self.preprocessors.get('model_config', {})
            self.model_type = (
                self.config.get('model_type')
                or model_config.get('model_type')
                or (checkpoint.get('model_type') if isinstance(checkpoint, dict) else None)
                or (self.preprocessors.get('model_type') if isinstance(self.preprocessors, dict) else None)
                or 'legacy_mlp'
            )

            if self.model_type in ('airport_pmst', 'improved_dual_stream_pmst'):
                if self.preprocessors is None:
                    self.preprocessors = {}
                self.airport_thresholds = (
                    self.config.get('thresholds')
                    or (checkpoint.get('thresholds') if isinstance(checkpoint, dict) else None)
                    or self.preprocessors.get('thresholds')
                    or {'fog': 0.5, 'mist': 0.5}
                )
                self.airport_season_thresholds = (
                    self.config.get('season_thresholds')
                    or (checkpoint.get('season_thresholds') if isinstance(checkpoint, dict) else None)
                    or self.preprocessors.get('season_thresholds')
                )
                self.airport_temperature = float(
                    self.config.get(
                        'temperature',
                        (checkpoint.get(
                            'temperature',
                            self.preprocessors.get('temperature', 1.0)
                        ) if isinstance(checkpoint, dict) else self.preprocessors.get('temperature', 1.0))
                    )
                )
                self.airport_station_order = (
                    self.config.get('station_names')
                    or (checkpoint.get('station_order') if isinstance(checkpoint, dict) else None)
                    or self.preprocessors.get('station_order')
                )
                if self.model_type == 'airport_pmst' and 'station_to_idx' not in self.preprocessors:
                    station_order = self.airport_station_order or (
                        checkpoint.get('station_order', []) if isinstance(checkpoint, dict) else []
                    )
                    self.preprocessors['station_to_idx'] = {
                        str(name): i for i, name in enumerate(station_order)
                    }
                if 'fill_values' not in self.preprocessors:
                    self.preprocessors['fill_values'] = checkpoint.get('fill_values', {}) if isinstance(checkpoint, dict) else {}
                if 'model_config' not in self.preprocessors:
                    self.preprocessors['model_config'] = model_config

                if self.model_type == 'improved_dual_stream_pmst':
                    self.model = build_improved_pmst_model(model_config).to(self.device)
                else:
                    self.model = build_airport_model(model_config).to(self.device)
                self.model.load_state_dict(model_state)
                logger.info(
                    f"机场PMST模型加载成功，type={self.model_type}, thresholds={self.airport_thresholds}, "
                    f"T={self.airport_temperature:.3f}"
                )
            else:
                if self.preprocessors is None:
                    raise FileNotFoundError(f"旧模型需要预处理器文件: {preprocessor_path}")
                self.model = BalancedLowVisibilityClassifier(
                    model_config['input_dim'],
                    model_config['hidden_dim'],
                    model_config['num_classes']
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("旧版MLP模型加载成功")

            self.model.eval()

            # 初始化数据处理器
            vegetation_data = None
            if (
                self.model_type != 'airport_pmst'
                and self.config.get('vegetation_file')
                and os.path.exists(self.config['vegetation_file'])
            ):
                vegetation_data = xr.open_dataset(self.config['vegetation_file'])

            grid_data_base_dir = self.config.get('grid_data_base_dir', '/sharedata/dataset/GroupData/GD001-EC_Forcasting')
            self.data_processor = DataProcessor(vegetation_data, grid_data_base_dir)

            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def get_airport_station_names(self, coords):
        """Return station ICAO names in the same order as the current forecast tensor."""
        if self.config.get('station_names'):
            names = [str(v) for v in self.config['station_names']]
        elif self.airport_station_order is not None:
            names = [str(v) for v in self.airport_station_order]
        elif 'station_name' in coords:
            names = [str(v) for v in coords['station_name'].values]
        elif 'station' in coords:
            names = [str(v) for v in coords['station'].values]
        elif 'num_station' in coords:
            names = [str(v) for v in coords['num_station'].values]
        else:
            names = []

        station_dim = None
        for key in ('station_name', 'station', 'num_station'):
            if key in coords:
                station_dim = len(coords[key])
                break
        if station_dim is not None and len(names) != station_dim:
            logger.warning(
                f"机场站点名数量与输入站点数不一致: names={len(names)}, station_dim={station_dim}; "
                "将回退为输入坐标值"
            )
            if 'num_station' in coords:
                names = [str(v) for v in coords['num_station'].values]
            elif 'station' in coords:
                names = [str(v) for v in coords['station'].values]
        return names

    def preprocess_data(self, features_ds):
        """预处理数据 - 修复维度处理"""
        try:
            if self.model_type in ('airport_pmst', 'improved_dual_stream_pmst'):
                station_names = self.get_airport_station_names(features_ds.coords)
                scaler = self.preprocessors.get('scaler')
                fill_values = self.preprocessors.get('fill_values', {})
                model_config = self.preprocessors.get('model_config', {})
                window_size = int(model_config.get('window_size', 12))
                local_time_offset_hours = float(
                    self.preprocessors.get(
                        'local_time_offset_hours',
                        self.config.get('local_time_offset_hours', 8)
                    )
                )
                use_source_zenith = bool(
                    self.preprocessors.get(
                        'use_source_zenith',
                        self.config.get('use_source_zenith', False)
                    )
                )
                if self.model_type == 'improved_dual_stream_pmst':
                    X_scaled, out_coords, unknown = build_pmst_inference_matrix(
                        features_ds,
                        station_names=station_names,
                        station_static=self.preprocessors.get('station_static', {}),
                        scaler=scaler,
                        fill_values=fill_values,
                        window_size=window_size,
                        local_time_offset_hours=local_time_offset_hours,
                        use_source_zenith=True,
                    )
                else:
                    station_to_idx = self.preprocessors.get('station_to_idx', {})
                    X_scaled, out_coords, unknown = build_inference_matrix(
                        features_ds,
                        station_names=station_names,
                        station_to_idx=station_to_idx,
                        scaler=scaler,
                        fill_values=fill_values,
                        window_size=window_size,
                        local_time_offset_hours=local_time_offset_hours,
                        use_source_zenith=use_source_zenith,
                    )
                if unknown:
                    logger.warning(f"存在未见过的机场站点，将使用默认静态/站点特征: {unknown[:10]}")
                logger.info(f"机场PMST预处理完成，type={self.model_type}, 输入形状: {X_scaled.shape}")
                return X_scaled, out_coords

            # 获取数据数组并添加调试信息
            X = features_ds.values  # shape: (variable, time, station)
            logger.info(f"原始features形状: {X.shape}")
            logger.info(f"features维度含义: (variable={X.shape[0]}, time={X.shape[1]}, station={X.shape[2]})")

            # 正确的维度转换: (variable, time, station) -> (time, station, variable) -> (samples, features)
            X = X.transpose(1, 2, 0)  # 转换为 (time, station, variable)
            logger.info(f"转置后形状: {X.shape} (time, station, variable)")

            # reshape: 将时间和站点维度合并为样本维度
            X = X.reshape(-1, X.shape[-1])  # reshape to (samples, features)
            logger.info(f"reshape后形状: {X.shape}")
            logger.info(f"reshape后维度含义: (samples={X.shape[0]}, features={X.shape[1]})")

            # 检查特征数量
            expected_features = 34
            if X.shape[1] != expected_features:
                logger.warning(f"特征数量不匹配，期望{expected_features}个，实际{X.shape[1]}个")
                # 如果特征不足，用零填充
                if X.shape[1] < expected_features:
                    padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                    X = np.concatenate([X, padding], axis=1)
                    logger.info(f"特征填充后形状: {X.shape}")
                else:
                    X = X[:, :expected_features]  # 截取前34个特征
                    logger.info(f"特征截取后形状: {X.shape}")

            # 预处理连续特征
            condition_scaler = self.preprocessors['condition_scaler']
            num_continuous_features = self.preprocessors.get('num_continuous_features', 26)

            X_continuous = X[:, :num_continuous_features]
            X_embedding = X[:, num_continuous_features:]
            X_continuous_scaled = condition_scaler.transform(X_continuous)
            X_scaled = np.concatenate([X_continuous_scaled, X_embedding], axis=1)

            logger.info(f"预处理完成，最终数据形状: {X_scaled.shape}")

            return X_scaled, features_ds.coords

        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def load_forecasted_timestamps(self):
        """从输出目录扫描已存在的预报文件，记录已预报的时间戳"""
        try:
            output_dir = self.config.get('output_dir', '.')
            if not os.path.exists(output_dir):
                logger.info("输出目录不存在，跳过已预报时间戳加载")
                return

            # 扫描所有预报结果文件
            forecast_files = glob.glob(os.path.join(output_dir, 'visibility_forecast_*.nc'))

            for filepath in forecast_files:
                try:
                    # 从文件名提取时间戳: visibility_forecast_2025091000.nc
                    filename = os.path.basename(filepath)
                    timestamp_str = filename.replace('visibility_forecast_', '').replace('.nc', '')

                    # 验证时间戳格式
                    timestamp = pd.to_datetime(timestamp_str, format='%Y%m%d%H')
                    self.forecasted_timestamps.add(timestamp_str)

                except Exception as e:
                    logger.debug(f"无法解析预报文件名 {filename}: {e}")
                    continue

            logger.info(f"加载已预报时间戳: {len(self.forecasted_timestamps)} 个")

        except Exception as e:
            logger.warning(f"加载已预报时间戳失败: {e}")

    def is_already_forecasted(self, timestamp):
        """检查该时间戳是否已经预报过"""
        timestamp_str = timestamp.strftime('%Y%m%d%H')

        # 方法1: 检查内存记录
        if timestamp_str in self.forecasted_timestamps:
            return True

        # 方法2: 检查文件是否存在（双重保险）
        output_file = os.path.join(
            self.config['output_dir'],
            f'visibility_forecast_{timestamp_str}.nc'
        )

        if os.path.exists(output_file):
            # 验证文件完整性
            try:
                ds = xr.open_dataset(output_file)
                required_vars = ['predicted_class', 'class_probabilities']
                has_required_vars = all(var in ds.data_vars for var in required_vars)
                ds.close()

                if has_required_vars:
                    # 文件有效，更新内存记录
                    self.forecasted_timestamps.add(timestamp_str)
                    return True
                else:
                    logger.warning(f"预报文件不完整: {output_file}")
                    return False

            except Exception as e:
                logger.warning(f"预报文件损坏: {output_file}, {e}")
                return False

        return False

    def mark_as_forecasted(self, timestamp):
        """标记该时间戳已完成预报"""
        timestamp_str = timestamp.strftime('%Y%m%d%H')
        self.forecasted_timestamps.add(timestamp_str)

        # 可选：追加到日志文件
        log_file = os.path.join(self.config.get('output_dir', '.'), 'forecasted_timestamps.log')
        try:
            with open(log_file, 'a') as f:
                f.write(f"{timestamp_str}\n")
        except Exception as e:
            logger.debug(f"无法写入预报时间戳日志: {e}")

    def predict_airport_classes(self, probabilities, coords):
        if not self.airport_season_thresholds:
            return predict_classes_from_probs(probabilities, self.airport_thresholds)

        time_values = pd.DatetimeIndex(pd.to_datetime(coords['time']))
        station_dim = len(coords['station_name'])
        predictions = np.empty(probabilities.shape[0], dtype=np.int64)

        def season_name(month):
            if month in (12, 1, 2):
                return 'DJF'
            if month in (3, 4, 5):
                return 'MAM'
            if month in (6, 7, 8):
                return 'JJA'
            return 'SON'

        for i, ts in enumerate(time_values):
            s_name = season_name(ts.month)
            rec = self.airport_season_thresholds.get(s_name, {})
            thresholds = {
                'fog': float(rec.get('fog_th', self.airport_thresholds.get('fog', 0.5))),
                'mist': float(rec.get('mist_th', self.airport_thresholds.get('mist', 0.5))),
            }
            sl = slice(i * station_dim, (i + 1) * station_dim)
            predictions[sl] = predict_classes_from_probs(probabilities[sl], thresholds)
        return predictions

    def predict(self, X_scaled, coords):
        """进行预测"""
        try:
            with torch.no_grad():
                if self.model_type in ('airport_pmst', 'improved_dual_stream_pmst'):
                    batch_size = int(self.config.get('batch_size', 4096))
                    probs_list = []
                    temp = max(float(self.airport_temperature), 1e-6)
                    for start in range(0, len(X_scaled), batch_size):
                        batch = torch.tensor(
                            X_scaled[start:start + batch_size],
                            dtype=torch.float32,
                            device=self.device
                        )
                        logits, _reg_pred, _low_vis = self.model(batch)
                        probs = F.softmax(logits / temp, dim=1)
                        probs_list.append(probs.cpu().numpy())
                    probabilities = np.concatenate(probs_list, axis=0)
                    predictions = self.predict_airport_classes(probabilities, coords)
                else:
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

                    main_logits, low_vis_confidence, _ = self.model(X_tensor)

                    # 后处理
                    main_prob = F.softmax(main_logits, dim=1)
                    aux_prob = torch.sigmoid(low_vis_confidence.squeeze())

                    adjusted_prob = main_prob.clone()

                    # 温和调整
                    high_conf_mask = aux_prob > 0.6
                    if high_conf_mask.any():
                        adjusted_prob[high_conf_mask, 0] *= 1.1
                        adjusted_prob[high_conf_mask, 2] *= 0.95

                    adjusted_prob = adjusted_prob / adjusted_prob.sum(dim=1, keepdim=True)
                    _, predicted = torch.max(adjusted_prob, 1)
                    probabilities = adjusted_prob.cpu().numpy()
                    predictions = predicted.cpu().numpy()

            return predictions, probabilities

        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None, None

    def save_forecast_results(self, predictions, probabilities, coords, timestamp):
        """保存预测结果 - 修复版本"""
        try:
            if self.model_type in ('airport_pmst', 'improved_dual_stream_pmst'):
                time_dim = len(coords['time'])
                station_names = [str(v) for v in coords['station_name']]
                station_dim = len(station_names)
                if predictions.size != time_dim * station_dim:
                    logger.error(
                        f"机场PMST predictions尺寸不匹配: {predictions.size} != {time_dim * station_dim}"
                    )
                    return None
                if probabilities.size != time_dim * station_dim * 3:
                    logger.error(
                        f"机场PMST probabilities尺寸不匹配: {probabilities.size} != "
                        f"{time_dim * station_dim * 3}"
                    )
                    return None
                predictions = predictions.reshape(time_dim, station_dim)
                probabilities = probabilities.reshape(time_dim, station_dim, 3)
                result_ds = xr.Dataset({
                    'predicted_class': (['time', 'station'], predictions.astype(np.int16)),
                    'class_probabilities': (['time', 'station', 'class'], probabilities.astype(np.float32))
                }, coords={
                    'time': coords['time'],
                    'station': station_names,
                    'class': ['<500m', '500-1000m', '>=1000m']
                }, attrs={
                    'model_type': self.model_type,
                    'fog_threshold': float(self.airport_thresholds.get('fog', 0.5)),
                    'mist_threshold': float(self.airport_thresholds.get('mist', 0.5)),
                    'temperature': float(self.airport_temperature),
                    'description': 'Airport METAR visibility class forecast'
                })

                timestamp_str = timestamp.strftime('%Y%m%d%H')
                output_file = os.path.join(
                    self.config['output_dir'],
                    f'visibility_forecast_{timestamp_str}.nc'
                )
                os.makedirs(self.config['output_dir'], exist_ok=True)
                result_ds.to_netcdf(output_file)

                class_counts = np.bincount(predictions.flatten(), minlength=3)
                total_samples = predictions.size
                logger.info(
                    f"机场PMST预测结果已保存: {output_file}; "
                    f"<500m: {class_counts[0]}/{total_samples} ({class_counts[0]/total_samples*100:.1f}%), "
                    f"500-1000m: {class_counts[1]}/{total_samples} ({class_counts[1]/total_samples*100:.1f}%), "
                    f">=1000m: {class_counts[2]}/{total_samples} ({class_counts[2]/total_samples*100:.1f}%)"
                )
                return output_file

            # 添加调试信息
            logger.info(f"保存结果 - predictions形状: {predictions.shape}")
            logger.info(f"保存结果 - probabilities形状: {probabilities.shape}")
            logger.info(f"保存结果 - coords['time']长度: {len(coords['time'])}")
            logger.info(f"保存结果 - coords['num_station']长度: {len(coords['num_station'])}")

            # 重塑预测结果
            time_dim = len(coords['time'])
            station_dim = len(coords['num_station'])

            logger.info(f"目标reshape尺寸: time={time_dim}, station={station_dim}")
            logger.info(f"predictions总元素数: {predictions.size}, 期望: {time_dim * station_dim}")
            logger.info(f"probabilities总元素数: {probabilities.size}, 期望: {time_dim * station_dim * 3}")

            # 验证尺寸匹配
            if predictions.size != time_dim * station_dim:
                logger.error(f"predictions尺寸不匹配: {predictions.size} != {time_dim * station_dim}")
                return None

            if probabilities.size != time_dim * station_dim * 3:
                logger.error(f"probabilities尺寸不匹配: {probabilities.size} != {time_dim * station_dim * 3}")
                return None

            predictions = predictions.reshape(time_dim, station_dim)
            probabilities = probabilities.reshape(time_dim, station_dim, 3)

            logger.info(f"reshape成功 - predictions: {predictions.shape}, probabilities: {probabilities.shape}")

            # 你的站点名列表，顺序需与num_station一致
            station_names = [
                'ZBHZ', 'ZBZL', 'ZPCW', 'ZPJM', 'ZSSR', 'ZWSC', 'ZYJS', 'ZYSQ', 'ZGYY', 'ZBYZ',
                'ZBZJ', 'ZGBH', 'ZGBS', 'ZGCD', 'ZGDY', 'ZGGG', 'ZGHA', 'ZGHY', 'ZSQZ', 'ZYMD',
                'ZYMH', 'ZYQQ', 'ZYTL', 'ZLXN', 'ZLXY', 'ZLYL', 'ZYJM', 'ZSSH', 'ZPTC', 'ZPPP',
                'ZSYC', 'ZSYN', 'ZSYT', 'ZSAM', 'ZSCG', 'ZSCN', 'ZBAD', 'ZSFZ', 'ZSGS', 'ZSGZ',
                'ZSHC', 'ZSJD', 'ZBUH', 'ZBUL', 'ZGMX', 'ZGNN', 'ZGOW', 'ZGSD', 'ZGSZ', 'ZGZH',
                None, 'ZHCC', 'ZHES', 'ZHHH', 'ZHLY', 'ZHNY', 'ZSLY', 'ZSNB', 'ZSNJ', 'ZSNT',
                'ZSOF', 'ZSPD', 'ZSQD', 'ZUAS', 'ZJSY', 'ZLDH', 'ZLGY', 'ZLHZ', 'ZUGY', 'ZBXZ',
                'ZBYC', 'ZBYN', 'ZLZY', 'ZPDL', 'ZPJH', 'ZPLC', 'ZPLJ', 'ZBAA', 'ZBAL', 'ZBAR',
                'ZBDH', 'ZBDS', 'ZYHB', 'ZYHE', 'ZHXF', 'ZSYW', 'ZSZS', 'ZBLF', 'ZBLL', 'ZBMZ',
                'ZBOW', 'ZUQJ', 'ZUTR', 'ZUUU', 'ZUWX', 'ZUXC', None, 'ZSJH', 'ZSJJ', 'ZSJN',
                'ZSJU', None, 'ZBTL', 'ZBUC', 'ZBER', 'ZBES', 'ZBHD', 'ZBHH', 'ZUBD', 'ZUBJ',
                'ZUCK', 'ZSLO', 'ZUZY', 'ZWAK', 'ZWAT', 'ZWHM', 'ZWHZ', 'ZLZW', 'ZULB', 'ZULS',
                'ZULZ', 'ZUMY', 'ZUNC', 'ZSRZ', 'ZHSY', 'ZSXZ', 'ZSYA', 'ZSSM', 'ZUGU', 'ZSSS',
                'ZSTX', 'ZSWF', 'ZSWH', 'ZSWX', 'ZSWY', 'ZSWZ', 'ZYCY', 'ZYDQ', 'ZUYB', 'ZBEN',
                'ZWKC', 'ZWKL', 'ZWKM', 'ZWSH', 'ZWTL', 'ZWTN', 'ZWWW', 'ZWYN', 'ZYAS', 'ZYCC',
                'ZHYC', 'ZJHK', 'ZJQH', 'ZLIC', 'ZLJC', 'ZLJQ', 'ZLLL', 'ZBSJ', 'ZBSN', 'ZBTJ',
                'ZGKL', 'ZUMT', 'ZBCD', 'ZBLA', 'ZUYI', 'ZYTX', 'ZYYJ', 'ZBDT', 'ZLQY', 'ZBXT',
                'ZHEC', 'ZHJZ', 'ZHXY', 'ZLAK', 'ZLLN', 'ZSHZ', 'ZSWA', 'ZUTF', None, 'ZYJZ',
                'ZYYK', 'ZZZZ'
            ]
            # 站点名数量需与num_station一致
            time_dim = len(coords['time'])
            station_dim = len(coords['num_station'])

            predictions = predictions.reshape(time_dim, station_dim)
            probabilities = probabilities.reshape(time_dim, station_dim, 3)

            # 判断是否可以用站点名替换
            if len(station_names) == station_dim:
                result_ds = xr.Dataset({
                    'predicted_class': (['time', 'station_name'], predictions),
                    'class_probabilities': (['time', 'station_name', 'class'], probabilities)
                }, coords={
                    'time': coords['time'],
                    'station_name': station_names,
                    'class': ['<500m', '500-1000m', '>1000m']
                })
            else:
                logger.warning("站点名数量与num_station数量不一致，仍用num_station保存")
                result_ds = xr.Dataset({
                    'predicted_class': (['time', 'num_station'], predictions),
                    'class_probabilities': (['time', 'num_station', 'class'], probabilities)
                }, coords={
                    'time': coords['time'],
                    'num_station': coords['num_station'],
                    'class': ['<500m', '500-1000m', '>1000m']
                })
            # 保存结果
            timestamp_str = timestamp.strftime('%Y%m%d%H')
            output_file = os.path.join(
                self.config['output_dir'],
                f'visibility_forecast_{timestamp_str}.nc'
            )
            os.makedirs(self.config['output_dir'], exist_ok=True)
            result_ds.to_netcdf(output_file)

            logger.info(f"预测结果已保存: {output_file}")

            # 打印预测摘要
            class_counts = np.bincount(predictions.flatten(), minlength=3)
            total_samples = predictions.size
            logger.info(f"预测摘要 - <500m: {class_counts[0]}/{total_samples} ({class_counts[0]/total_samples*100:.1f}%), "
                       f"500-1000m: {class_counts[1]}/{total_samples} ({class_counts[1]/total_samples*100:.1f}%), "
                       f">1000m: {class_counts[2]}/{total_samples} ({class_counts[2]/total_samples*100:.1f}%)")

            return output_file

        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            import traceback
            traceback.print_exc()
            return None


    def process_new_data(self, data_dir):
        """处理新数据目录"""
        overall_start = time.time()

        timestamp = parse_timestamp_from_dirname(os.path.basename(data_dir))
        if timestamp is None:
            logger.warning(f"无法解析时间戳: {data_dir}")
            return False

        # 新增：检查是否已预报
        if self.is_already_forecasted(timestamp):
            logger.info(f"跳过已预报数据: {data_dir} ({timestamp})")
            return True  # 返回True表示无需重新处理

        logger.info(f"=" * 80)
        logger.info(f"处理新数据: {data_dir} ({timestamp})")
        logger.info(f"=" * 80)

        # 合并数据
        merge_start = time.time()
        features_ds = self.data_processor.merge_single_timestep_data(data_dir, timestamp)
        if features_ds is None:
            logger.error(f"数据合并失败: {data_dir}")
            return False
        merge_elapsed = time.time() - merge_start
        logger.info(f"数据合并耗时: {merge_elapsed:.2f}秒")

        # 预处理
        preprocess_start = time.time()
        X_scaled, coords = self.preprocess_data(features_ds)
        if X_scaled is None:
            logger.error(f"数据预处理失败: {data_dir}")
            return False
        preprocess_elapsed = time.time() - preprocess_start
        logger.info(f"数据预处理耗时: {preprocess_elapsed:.2f}秒")

        # 预测
        predict_start = time.time()
        predictions, probabilities = self.predict(X_scaled, coords)
        if predictions is None:
            logger.error(f"预测失败: {data_dir}")
            return False
        predict_elapsed = time.time() - predict_start
        logger.info(f"模型预测耗时: {predict_elapsed:.2f}秒")

        # 保存结果
        save_start = time.time()
        output_file = self.save_forecast_results(predictions, probabilities, coords, timestamp)
        if output_file:
            # 新增：标记为已预报
            self.mark_as_forecasted(timestamp)
            save_elapsed = time.time() - save_start
            logger.info(f"结果保存耗时: {save_elapsed:.2f}秒")

            total_elapsed = time.time() - overall_start
            logger.info(f"=" * 80)
            logger.info(f"成功完成预报: {data_dir} -> {output_file}")
            logger.info(f"总耗时: {total_elapsed:.2f}秒 (合并:{merge_elapsed:.1f}s + 预处理:{preprocess_elapsed:.1f}s + 预测:{predict_elapsed:.1f}s + 保存:{save_elapsed:.1f}s)")
            logger.info(f"=" * 80)
            return True

        return False

    def load_processed_log(self):
        """加载已处理目录的日志"""
        log_file = os.path.join(self.config.get('output_dir', '.'), 'processed_dirs.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                self.processed_dirs = set(line.strip() for line in f if line.strip())
            logger.info(f"加载已处理目录日志: {len(self.processed_dirs)} 个目录")

    def save_processed_log(self, data_dir):
        """保存已处理目录到日志"""
        log_file = os.path.join(self.config.get('output_dir', '.'), 'processed_dirs.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(f"{data_dir}\n")

    def process_existing_data(self):
        """批量处理已存在的数据 - 跳过已预报的"""
        logger.info("开始批量处理现有数据...")
        existing_dirs = glob.glob(os.path.join(self.config['data_root_dir'], '*'))
        existing_dirs = [d for d in existing_dirs if os.path.isdir(d)]

        success_count = 0
        skipped_count = 0
        total_count = len(existing_dirs)

        logger.info(f"发现 {total_count} 个目录")
        logger.info(f"已预报 {len(self.forecasted_timestamps)} 个时间点")

        for data_dir in sorted(existing_dirs):
            dir_name = os.path.basename(data_dir)

            # 解析时间戳
            timestamp = parse_timestamp_from_dirname(dir_name)
            if timestamp is None:
                logger.debug(f"跳过无效目录名: {dir_name}")
                continue

            # 新增：检查是否已预报
            if self.is_already_forecasted(timestamp):
                logger.debug(f"跳过已预报: {dir_name}")
                skipped_count += 1
                continue

            if data_dir in self.processed_dirs:
                logger.debug(f"跳过已处理目录: {dir_name}")
                skipped_count += 1
                continue

            logger.info(f"正在检查目录: {dir_name}")

            # 检查数据完整性
            completeness = check_data_completeness(data_dir)

            if not completeness['complete']:
                logger.warning(f"数据不完整，跳过: {dir_name}")
                if completeness['missing_files']:
                    logger.warning(f"  缺失文件: {completeness['missing_files']}")
                if completeness['incomplete_files']:
                    logger.warning(f"  不完整文件: {completeness['incomplete_files']}")
                continue

            logger.info(f"数据完整性检查通过: {dir_name}")

            # 尝试处理
            try:
                success = self.process_new_data(data_dir)
                if success:
                    success_count += 1
                    self.processed_dirs.add(data_dir)
                    self.save_processed_log(data_dir)
                    logger.info(f"成功处理: {dir_name}")
                else:
                    logger.error(f"处理失败: {dir_name}")
            except Exception as e:
                logger.error(f"处理异常 {dir_name}: {e}")
                import traceback
                traceback.print_exc()

        logger.info(f"批量处理完成: 成功{success_count}个, 跳过{skipped_count}个, 共{total_count}个目录")

    def cleanup_old_forecasts(self, days_to_keep=7):
        """清理旧的预报结果文件（可选功能）"""
        try:
            output_dir = self.config.get('output_dir', '.')
            if not os.path.exists(output_dir):
                return

            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            forecast_files = glob.glob(os.path.join(output_dir, 'visibility_forecast_*.nc'))

            removed_count = 0
            for filepath in forecast_files:
                try:
                    filename = os.path.basename(filepath)
                    timestamp_str = filename.replace('visibility_forecast_', '').replace('.nc', '')
                    file_time = pd.to_datetime(timestamp_str, format='%Y%m%d%H')

                    if file_time.to_pydatetime() < cutoff_time:
                        os.remove(filepath)
                        removed_count += 1
                        # 同时从内存记录中移除
                        self.forecasted_timestamps.discard(timestamp_str)

                except Exception as e:
                    logger.debug(f"清理文件失败 {filename}: {e}")

            if removed_count > 0:
                logger.info(f"清理了 {removed_count} 个超过{days_to_keep}天的旧预报文件")

        except Exception as e:
            logger.warning(f"清理旧预报文件失败: {e}")

    def monitor_and_forecast(self, process_existing=False, scan_only_new=True):
        """
        持续监控并预报
        process_existing: 是否处理已存在的数据
        scan_only_new: 是否只扫描新文件
        """
        logger.info(f"开始监控目录: {self.config['data_root_dir']}")
        logger.info(f"检查间隔: {self.config['check_interval']}秒")

        # 加载已处理目录日志
        if scan_only_new:
            self.load_processed_log()

        # 批量处理现有数据
        if process_existing:
            self.process_existing_data()

        # 初始扫描
        if not process_existing:
            existing_dirs = set(glob.glob(os.path.join(self.config['data_root_dir'], '*')))
            if scan_only_new:
                # 只记录现有目录，不处理
                self.processed_dirs.update(existing_dirs)
                logger.info(f"初始扫描完成，记录 {len(existing_dirs)} 个现有目录（不处理）")
            else:
                # 处理所有现有目录
                self.processed_dirs.update(existing_dirs)
                logger.info(f"初始扫描完成，发现 {len(existing_dirs)} 个现有目录")

        while True:
            try:
                # 扫描新目录
                current_dirs = set(glob.glob(os.path.join(self.config['data_root_dir'], '*')))
                current_dirs = {d for d in current_dirs if os.path.isdir(d)}
                new_dirs = current_dirs - self.processed_dirs

                if new_dirs:
                    logger.info(f"发现 {len(new_dirs)} 个新目录")
                    for new_dir in sorted(new_dirs):
                        # 检查是否是最近失败的目录，避免频繁重试
                        if new_dir in self.failed_dirs:
                            last_fail_time = self.failed_dirs[new_dir]
                            if time.time() - last_fail_time < 300:  # 5分钟内不重试
                                continue

                        # 检查数据完整性
                        completeness = check_data_completeness(new_dir)
                        if not completeness['complete']:
                            logger.info(f"数据传输未完成，稍后重试: {new_dir}")
                            logger.debug(f"  缺失: {completeness['missing_files']}")
                            logger.debug(f"  不完整: {completeness['incomplete_files']}")
                            continue

                        success = self.process_new_data(new_dir)
                        if success:
                            self.processed_dirs.add(new_dir)
                            self.save_processed_log(new_dir)
                            # 清除失败记录
                            if new_dir in self.failed_dirs:
                                del self.failed_dirs[new_dir]
                        else:
                            # 记录失败时间
                            self.failed_dirs[new_dir] = time.time()
                            logger.warning(f"处理失败，将在5分钟后重试: {new_dir}")

                # 等待
                time.sleep(self.config['check_interval'])

            except KeyboardInterrupt:
                logger.info("收到停止信号，正在退出...")
                break
            except Exception as e:
                logger.error(f"监控过程中出错: {e}")
                time.sleep(10)

# ------------------------------
# 配置文件生成函数
# ------------------------------
def create_default_config():
    """创建默认配置文件"""
    config = {
        "data_root_dir": "/public/home/chenxi/PuTS/tianji",
        "model_type": "improved_dual_stream_pmst",
        "model_path": "/public/home/putianshu/vis_mlp/checkpoints/airport_metar_2025_full_from_scratch_Airport_Full_best_score.pt",
        "preprocessor_path": "/public/home/putianshu/vis_mlp/checkpoints/airport_metar_2025_full_from_scratch_airport_preprocessor.pkl",
        "vegetation_file": "/public/home/chenxi/PuTS/tianji/data_vegtype.nc",
        "grid_data_base_dir": "/sharedata/dataset/GroupData/GD001-EC_Forcasting",
        "output_dir": "/public/home/chenxi/PuTS/tianji/forecasts",
        "check_interval": 60,
        "batch_size": 4096,
        "idw_power": 2.0,
        "idw_max_distance": 5.0,
        "local_time_offset_hours": 8,
        "use_source_zenith": False
    }

    with open('forecast_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("已创建默认配置文件: forecast_config.json")
    print("请根据实际情况修改配置文件中的路径")
    print("新增配置项说明：")
    print("- grid_data_base_dir: 格点数据基础目录路径")
    print("- idw_power: IDW插值的幂次参数，默认为2.0")
    print("- idw_max_distance: IDW插值最大搜索距离（度），默认为5.0")

# ------------------------------
# 主函数
# ------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description='实时能见度预报系统（性能优化版）')
    parser.add_argument('--config', '-c', default='forecast_config.json',
                       help='配置文件路径')
    parser.add_argument('--create-config', action='store_true',
                       help='创建默认配置文件')
    parser.add_argument('--process-existing', action='store_true',
                       help='批量处理已存在的数据')
    parser.add_argument('--scan-all', action='store_true',
                       help='扫描所有文件（包括已存在的）')
    parser.add_argument('--cleanup-days', type=int, default=None,
                       help='清理N天前的旧预报文件（可选）')
    args = parser.parse_args()

    if args.create_config:
        create_default_config()
        return

    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        logger.info("使用 --create-config 创建默认配置文件")
        return

    # 初始化预报系统
    forecast_system = VisibilityForecastSystem(args.config)

    # 加载模型
    if not forecast_system.load_model():
        logger.error("模型加载失败，退出程序")
        return

    # 新增：清理旧文件（如果指定）
    if args.cleanup_days is not None:
        logger.info(f"开始清理 {args.cleanup_days} 天前的旧预报文件...")
        forecast_system.cleanup_old_forecasts(days_to_keep=args.cleanup_days)

    # 开始监控和预报
    try:
        forecast_system.monitor_and_forecast(
            process_existing=args.process_existing,
            scan_only_new=not args.scan_all
        )
    except Exception as e:
        logger.error(f"系统运行异常: {e}")

if __name__ == "__main__":
    main()
