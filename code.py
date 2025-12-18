# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pywt
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ==================== 参数配置 ====================
class Config:
    # S2: 阈值调整系数
    K_THRESHOLD = 1.2  # k ∈ [1.0, 1.5]
    
    # S4: 几何参数阈值（将通过统计计算）
    A_MIN = 1000  # 临时初始值，将通过训练数据计算
    A_MAX = 50000
    ALPHA_MIN = 2.0
    ALPHA_MAX = 15.0
    BETA_MIN = 0.1
    BETA_MAX = 0.9
    
    # S5: 极小值
    EPSILON = 1e-12
    
    # S7: MLP模型参数
    MLP_HIDDEN_LAYERS = (64, 32)
    MLP_MAX_ITER = 500
    MLP_RANDOM_STATE = 42
    
    # 图像处理参数
    MIN_LEAF_AREA = 500  # 最小叶片面积
    MORPH_KERNEL_SIZE = 5  # 形态学操作核大小


# ==================== 步骤S1: 图像获取 ====================
def load_infrared_image(image_path: str) -> np.ndarray:
    """
    S1: 通过中波红外相机获取玉米植株的红外图像X
    
    参数:
        image_path: 红外图像文件路径
        
    返回:
        X: 灰度图像数组，形状(H, W)
    """
    # 读取图像并转换为灰度
    X = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if X is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    # 归一化到0-255范围
    X = cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    print(f"S1: 成功加载红外图像，尺寸: {X.shape}")
    return X


# ==================== 步骤S2: 小波边界增强 ====================
def perform_wavelet_transform(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    S21: 执行二维离散小波变换，生成高频细节子带
    
    参数:
        X: 输入灰度图像
        
    返回:
        LH: 水平低频、垂直高频子带
        HL: 水平高频、垂直低频子带
        HH: 水平高频、垂直高频子带
    """
    # 进行1层二维离散小波变换
    coeffs = pywt.dwt2(X, 'db1')
    LL, (LH, HL, HH) = coeffs
    
    print(f"S21: 小波变换完成，各子带尺寸: LH={LH.shape}, HL={HL.shape}, HH={HH.shape}")
    return LH, HL, HH


def build_high_frequency_energy_map(LH: np.ndarray, HL: np.ndarray, HH: np.ndarray) -> np.ndarray:
    """
    S21: 构建高频能量图E(xi, yi)
    
    公式: E = sqrt(LH² + HL² + HH²)
    
    参数:
        LH, HL, HH: 三个高频子带
        
    返回:
        E: 高频能量图
    """
    # 将所有子带调整为相同尺寸
    min_h = min(LH.shape[0], HL.shape[0], HH.shape[0])
    min_w = min(LH.shape[1], HL.shape[1], HH.shape[1])
    
    LH_resized = cv2.resize(LH, (min_w, min_h))
    HL_resized = cv2.resize(HL, (min_w, min_h))
    HH_resized = cv2.resize(HH, (min_w, min_h))
    
    # 计算能量图
    E = np.sqrt(LH_resized**2 + HL_resized**2 + HH_resized**2)
    
    # 归一化
    E = cv2.normalize(E, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    print(f"S21: 高频能量图构建完成，尺寸: {E.shape}")
    return E


def calculate_threshold_Te(E: np.ndarray, k: float = Config.K_THRESHOLD) -> float:
    """
    S22: 计算阈值Te
    
    公式: Te = μE + k × σE
    
    参数:
        E: 高频能量图
        k: 阈值调整系数
        
    返回:
        Te: 计算得到的阈值
    """
    mu_E = np.mean(E)
    sigma_E = np.std(E)
    Te = mu_E + k * sigma_E
    
    print(f"S22: 能量统计 - 均值={mu_E:.4f}, 标准差={sigma_E:.4f}, 阈值Te={Te:.4f}")
    return Te


def generate_boundary_response_map(E: np.ndarray, Te: float) -> np.ndarray:
    """
    S23: 生成边界响应图B(xi, yi)
    
    公式: B(x,y) = 1 如果 E(x,y) >= Te, 否则为 0
    
    参数:
        E: 高频能量图
        Te: 阈值
        
    返回:
        B: 二值边界响应图
    """
    B = (E >= Te).astype(np.uint8) * 255
    
    print(f"S23: 边界响应图生成完成，非零像素数: {np.sum(B > 0)}")
    return B


def step_s2_wavelet_enhancement(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    S2总流程: 执行小波边界增强
    
    参数:
        X: 输入红外图像
        
    返回:
        E: 高频能量图
        B: 边界响应图
    """
    print("\n" + "="*50)
    print("开始执行S2: 小波边界增强")
    print("="*50)
    
    # S21: 小波变换和能量图
    LH, HL, HH = perform_wavelet_transform(X)
    E = build_high_frequency_energy_map(LH, HL, HH)
    
    # S22: 计算阈值
    Te = calculate_threshold_Te(E, Config.K_THRESHOLD)
    
    # S23: 生成边界响应图
    B = generate_boundary_response_map(E, Te)
    
    return E, B


# ==================== 步骤S3: 叶片区域分割 ====================
def segment_leaf_region(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    S3: 基于边界响应图分割叶片区域
    
    使用边界响应图作为指导，与原图像结合进行分割
    
    参数:
        X: 原始红外图像
        B: 边界响应图
        
    返回:
        X1: 叶片区域二值掩膜
    """
    # 使用边界图指导的区域生长或阈值分割
    # 这里采用改进的阈值分割：在原图像中，边界响应高的区域认为是叶片
    
    # 确保B是二值的
    _, B_binary = cv2.threshold(B, 127, 255, cv2.THRESH_BINARY)
    
    # 在原图像中，边界响应为1的区域保留
    masked_image = cv2.bitwise_and(X, X, mask=B_binary)
    
    # Otsu阈值分割
    _, X1 = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 清理小噪声
    kernel = np.ones((3, 3), np.uint8)
    X1 = cv2.morphologyEx(X1, cv2.MORPH_OPEN, kernel, iterations=1)
    
    print(f"S3: 叶片区域掩膜生成完成，尺寸: {X1.shape}")
    return X1


# ==================== 步骤S4: 连通域分析与单叶分割 ====================
def connected_component_analysis(X1: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    S41: 执行连通域标记
    
    参数:
        X1: 叶片区域二值掩膜
        
    返回:
        labels: 标记后的图像
        num_labels: 连通域数量
        stats: 每个连通域的统计信息
    """
    # 确保是二值图像
    if len(X1.shape) == 3:
        X1_gray = cv2.cvtColor(X1, cv2.COLOR_BGR2GRAY)
    else:
        X1_gray = X1.copy()
    
    # 连通域标记
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        X1_gray, connectivity=8
    )
    
    # 排除背景（标签0）
    num_labels -= 1
    labels = labels - 1
    
    print(f"S41: 连通域分析完成，检测到{num_labels}个区域")
    return labels, num_labels, stats, centroids


def calculate_geometric_parameters(region_mask: np.ndarray) -> Dict[str, float]:
    """
    S42: 计算单个区域的几何参数
    
    参数:
        region_mask: 单个区域的二值掩膜
        
    返回:
        dict: 包含面积、长宽比、紧致度的字典
    """
    # 面积
    area = np.sum(region_mask > 0)
    
    # 计算轮廓
    contours, _ = cv2.findContours(
        region_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return {'area': 0, 'alpha': 0, 'beta': 0}
    
    cnt = contours[0]
    
    # 外接矩形长宽比
    x, y, w, h = cv2.boundingRect(cnt)
    alpha = h / w if w > 0 else 0
    
    # 周长和紧致度
    perimeter = cv2.arcLength(cnt, True)
    beta = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    
    return {
        'area': area,
        'alpha': alpha,
        'beta': beta
    }


def filter_candidates(labels: np.ndarray, stats: np.ndarray, 
                     centroids: np.ndarray) -> List[int]:
    """
    S42: 筛选候选叶片
    
    参数:
        labels: 标记图像
        stats: 统计信息
        centroids: 质心坐标
        
    返回:
        valid_indices: 符合条件的区域索引列表
    """
    valid_indices = []
    
    # 计算所有区域的平均几何参数（模拟S421-S422）
    # 实际应用中应该基于标注数据计算
    all_params = {
        'areas': [],
        'alphas': [],
        'betas': []
    }
    
    for i in range(len(stats)):
        # 创建单个区域的掩膜
        region_mask = (labels == i).astype(np.uint8) * 255
        
        # 过滤掉背景和小区域
        if np.sum(region_mask) < Config.MIN_LEAF_AREA:
            continue
            
        params = calculate_geometric_parameters(region_mask)
        all_params['areas'].append(params['area'])
        all_params['alphas'].append(params['alpha'])
        all_params['betas'].append(params['beta'])
    
    # 计算平均值并确定阈值
    if not all_params['areas']:
        return []
    
    A_avg = np.mean(all_params['areas'])
    alpha_avg = np.mean(all_params['alphas'])
    beta_avg = np.mean(all_params['betas'])
    
    # 根据专利公式计算阈值（这里使用简化版本）
    Config.A_MIN = A_avg * 0.3
    Config.A_MAX = A_avg * 3.0
    Config.ALPHA_MIN = alpha_avg * 0.5
    Config.ALPHA_MAX = alpha_avg * 2.0
    Config.BETA_MIN = beta_avg * 0.3
    Config.BETA_MAX = min(beta_avg * 1.5, 0.9)
    
    print(f"S42: 阈值计算 - A_min={Config.A_MIN:.0f}, A_max={Config.A_MAX:.0f}")
    print(f"          alpha_min={Config.ALPHA_MIN:.2f}, alpha_max={Config.ALPHA_MAX:.2f}")
    print(f"          beta_min={Config.BETA_MIN:.2f}, beta_max={Config.BETA_MAX:.2f}")
    
    # 筛选符合条件的区域
    for i in range(len(stats)):
        region_mask = (labels == i).astype(np.uint8) * 255
        
        if np.sum(region_mask) < Config.MIN_LEAF_AREA:
            continue
            
        params = calculate_geometric_parameters(region_mask)
        
        # 应用判定条件
        if (Config.A_MIN <= params['area'] <= Config.A_MAX and
            Config.ALPHA_MIN <= params['alpha'] <= Config.ALPHA_MAX and
            Config.BETA_MIN <= params['beta'] <= Config.BETA_MAX):
            valid_indices.append(i)
    
    print(f"S42: 候选叶片筛选完成，保留{len(valid_indices)}个区域")
    return valid_indices


def split_touching_leaves(labels: np.ndarray, valid_indices: List[int], 
                         B: np.ndarray) -> List[np.ndarray]:
    """
    S43: 分割粘连叶片，生成单叶掩膜X1^i
    
    参数:
        labels: 标记图像
        valid_indices: 有效区域索引
        B: 边界响应图
        
    返回:
        leaf_masks: 单叶掩膜列表
    """
    leaf_masks = []
    
    # 确保B是二值的
    if B.max() > 1:
        B_binary = (B > 127).astype(np.uint8)
    else:
        B_binary = B.astype(np.uint8)
    
    for idx in valid_indices:
        # 获取区域掩膜
        region_mask = (labels == idx).astype(np.uint8)
        
        # 如果区域内可能包含多个叶片（基于面积判断）
        area = np.sum(region_mask)
        
        # 使用分水岭算法分割粘连叶片
        # 使用距离变换和局部最大值作为种子
        distance = ndimage.distance_transform_edt(region_mask)
        
        # 寻找局部最大值作为种子点
        from scipy import ndimage as ndi
        local_maxi = ndi.maximum_filter(distance, size=20, mode='constant')
        local_max_mask = (distance == local_maxi) & (distance > 5)
        
        # 标记种子
        markers, num_markers = ndi.label(local_max_mask)
        
        if num_markers > 1:
            # 存在多个种子，应用分水岭分割
            from skimage.segmentation import watershed
            segmented = watershed(-distance, markers, mask=region_mask)
            
            # 分离每个叶片
            for leaf_id in range(1, num_markers + 1):
                leaf_mask = (segmented == leaf_id).astype(np.uint8) * 255
                
                if np.sum(leaf_mask) > Config.MIN_LEAF_AREA:
                    leaf_masks.append(leaf_mask)
        else:
            # 单个叶片
            leaf_mask = region_mask * 255
            leaf_masks.append(leaf_mask)
    
    print(f"S43: 单叶片分割完成，生成{len(leaf_masks)}个单叶掩膜")
    return leaf_masks


def step_s4_leaf_segmentation(X1: np.ndarray, B: np.ndarray) -> List[np.ndarray]:
    """
    S4总流程: 连通域分析和单叶片分割
    
    参数:
        X1: 叶片区域二值掩膜
        B: 边界响应图
        
    返回:
        leaf_masks: 单叶掩膜列表
    """
    print("\n" + "="*50)
    print("开始执行S4: 连通域分析与单叶片分割")
    print("="*50)
    
    # S41: 连通域标记
    labels, num_labels, stats, centroids = connected_component_analysis(X1)
    
    # S42: 筛选候选叶片
    valid_indices = filter_candidates(labels, stats, centroids)
    
    # S43: 分割单叶片
    leaf_masks = split_touching_leaves(labels, valid_indices, B)
    
    return leaf_masks


# ==================== 步骤S5: 弯曲度量化 ====================
def remove_small_holes(leaf_mask: np.ndarray) -> np.ndarray:
    """
    S51: 通过形态学闭运算去除小孔
    
    参数:
        leaf_mask: 单叶掩膜
        
    返回:
        X2: 去除小孔后的掩膜
    """
    kernel = np.ones((Config.MORPH_KERNEL_SIZE, Config.MORPH_KERNEL_SIZE), np.uint8)
    X2 = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return X2


def extract_centerline(X2: np.ndarray) -> np.ndarray:
    """
    S52-S53: 提取叶片中心线
    
    参数:
        X2: 去除小孔后的掩膜
        
    返回:
        centerline: 中心线像素坐标数组，形状(N, 2)
    """
    # S52: 距离变换
    D = ndimage.distance_transform_edt(X2)
    
    # S53: 检测距离值局部最大值
    # 使用形态学操作检测局部最大值
    local_max = ndimage.maximum_filter(D, size=5)
    local_max_mask = (D == local_max) & (D > 2)
    
    # 提取中心线像素坐标
    y_coords, x_coords = np.where(local_max_mask)
    centerline = np.column_stack([x_coords, y_coords])
    
    # 按y坐标排序（从上到下）
    if len(centerline) > 0:
        centerline = centerline[np.argsort(centerline[:, 1])]
    
    return centerline


def calculate_discrete_curvature(centerline: np.ndarray) -> np.ndarray:
    """
    S54: 基于离散曲率原理计算中心线曲率
    
    公式: 
        Ki = |x'y'' - y'x''| / (x'² + y'²)^(3/2) + ε
    
    参数:
        centerline: 中心线像素坐标，形状(M, 2)
        
    返回:
        K: 离散曲率数组，形状(M,)，曲率定义为0
    """
    if len(centerline) < 5:
        return np.array([])
    
    x = centerline[:, 0].astype(np.float64)
    y = centerline[:, 1].astype(np.float64)
    
    # 计算相邻点之间的像素间距li
    dx = np.diff(x)
    dy = np.diff(y)
    l = np.sqrt(dx**2 + dy**2)
    
    # 计算弧长Δqi
    # Δqi = 0.5 * (li-1 + li)
    l_prev = np.concatenate([[l[0]], l[:-1]])
    l_next = np.concatenate([l[1:], [l[-1]]])
    delta_q = 0.5 * (l_prev + l_next)
    
    # 计算一阶导数x', y'
    # x'i = (xi+1 - xi-1) / (2Δqi)
    x_next = np.concatenate([x[1:], [x[-1]]])
    x_prev = np.concatenate([[x[0]], x[:-1]])
    x_prime = (x_next - x_prev) / (2 * delta_q + Config.EPSILON)
    
    y_next = np.concatenate([y[1:], [y[-1]]])
    y_prev = np.concatenate([[y[0]], y[:-1]])
    y_prime = (y_next - y_prev) / (2 * delta_q + Config.EPSILON)
    
    # 计算二阶导数x'', y''
    x_prime_next = np.concatenate([x_prime[1:], [x_prime[-1]]])
    x_prime_prev = np.concatenate([[x_prime[0]], x_prime[:-1]])
    x_double_prime = (x_prime_next - x_prime_prev) / (2 * delta_q + Config.EPSILON)
    
    y_prime_next = np.concatenate([y_prime[1:], [y_prime[-1]]])
    y_prime_prev = np.concatenate([[y_prime[0]], y_prime[:-1]])
    y_double_prime = (y_prime_next - y_prime_prev) / (2 * delta_q + Config.EPSILON)
    
    # 计算曲率Ki
    numerator = np.abs(x_prime * y_double_prime - y_prime * x_double_prime)
    denominator = (x_prime**2 + y_prime**2)**1.5 + Config.EPSILON
    K = numerator / denominator
    
    return K


def step_s5_curvature_quantification(leaf_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    S5总流程: 量化叶片弯曲度
    
    参数:
        leaf_mask: 单叶掩膜
        
    返回:
        K: 离散曲率数组
        centerline: 中心线坐标
    """
    print("  执行S5: 弯曲度量化...")
    
    # S51: 去除小孔
    X2 = remove_small_holes(leaf_mask)
    
    # S52-S53: 提取中心线
    centerline = extract_centerline(X2)
    
    if len(centerline) < 5:
        print("  警告: 中心线点数不足，无法计算曲率")
        return np.array([]), centerline
    
    # S54: 计算离散曲率
    K = calculate_discrete_curvature(centerline)
    
    print(f"  中心线点数: {len(centerline)}, 曲率点数: {len(K)}")
    return K, centerline


# ==================== 步骤S6: 构建特征向量 ====================
def build_feature_vector(K: np.ndarray, centerline: np.ndarray) -> Optional[np.ndarray]:
    """
    S6: 构建包含反映叶片卷曲特性的多维特征向量
    
    特征维度:
        f1: 平均绝对曲率
        f2: 曲率标准差
        f3: 最大曲率
        f4: 端点收缩比
        f5: 归一化弯曲能量
    
    参数:
        K: 离散曲率数组
        centerline: 中心线坐标
        
    返回:
        f: 5维特征向量，如果K为空则返回None
    """
    if len(K) == 0 or len(K) < 3:
        return None
    
    # S61: 平均绝对曲率
    # 排除首尾点
    tau = np.arange(1, len(K)-1)
    K_tau = K[tau]
    K_bar = np.mean(np.abs(K_tau))
    
    # S62: 曲率标准差
    sigma_K = np.std(K_tau)
    
    # S63: 最大曲率
    K_max = np.max(K)
    
    # S64: 端点收缩比C
    # d = 叶片首尾两点之间的最短物理距离
    # L = 叶片沿中轴线的真实弯曲长度
    if len(centerline) >= 2:
        d = np.sqrt(
            (centerline[-1, 0] - centerline[0, 0])**2 +
            (centerline[-1, 1] - centerline[0, 1])**2
        )
        
        # 计算沿中心线的真实长度
        dx = np.diff(centerline[:, 0])
        dy = np.diff(centerline[:, 1])
        L = np.sum(np.sqrt(dx**2 + dy**2))
        
        C = d / (L + Config.EPSILON)
    else:
        C = 0
    
    # S65: 归一化弯曲能量
    l = np.sqrt(
        np.diff(centerline[:, 0])**2 + np.diff(centerline[:, 1])**2
    )
    E_norm = np.sum(K[:-1]**2 * l) / (len(K) + Config.EPSILON)
    
    # 构建特征向量
    f = np.array([
        K_bar,      # 平均绝对曲率
        sigma_K,    # 曲率标准差
        K_max,      # 最大曲率
        C,          # 端点收缩比
        E_norm      # 归一化弯曲能量
    ])
    
    print(f"  S6: 特征向量构建完成: {f}")
    return f


# ==================== 步骤S7: MLP分类 ====================
def train_mlp_classifier(feature_vectors: List[np.ndarray], 
                        labels: List[int]) -> Tuple[MLPClassifier, StandardScaler]:
    """
    S7: 训练多层感知机模型
    
    结构: 输入层(5) -> 隐藏层1(64, ReLU) -> 隐藏层2(32, ReLU) -> 输出层(3, Softmax)
    
    参数:
        feature_vectors: 特征向量列表
        labels: 标签列表 (0:不干旱, 1:轻度干旱, 2:重度干旱)
        
    返回:
        model: 训练好的MLP模型
        scaler: 特征标准化器
    """
    print("\n" + "="*50)
    print("开始执行S7: MLP模型训练")
    print("="*50)
    
    # 准备数据
    X = np.array(feature_vectors)
    y = np.array(labels)
    
    print(f"训练样本数: {len(X)}, 特征维度: {X.shape[1]}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=Config.MLP_RANDOM_STATE, stratify=y
    )
    
    # 构建MLP模型
    model = MLPClassifier(
        hidden_layer_sizes=Config.MLP_HIDDEN_LAYERS,
        activation='relu',
        solver='adam',
        max_iter=Config.MLP_MAX_ITER,
        random_state=Config.MLP_RANDOM_STATE,
        verbose=False
    )
    
    # 训练
    model.fit(X_train, y_train)
    
    # 评估
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"训练准确率: {train_score:.4f}")
    print(f"测试准确率: {test_score:.4f}")
    
    return model, scaler


def predict_drought_level(model: MLPClassifier, 
                         scaler: StandardScaler, 
                         feature_vector: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    S7: 预测干旱等级
    
    参数:
        model: 训练好的MLP模型
        scaler: 特征标准化器
        feature_vector: 输入特征向量
        
    返回:
        (predicted_class, probabilities): 预测的类别和概率
    """
    # 标准化
    X_scaled = scaler.transform(feature_vector.reshape(1, -1))
    
    # 预测
    predicted_class = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    class_names = ['不干旱', '轻度干旱', '重度干旱']
    print(f"S7: 预测结果: {class_names[predicted_class]} (置信度: {probabilities[predicted_class]:.2%})")
    print(f"  各类别概率: {dict(zip(class_names, probabilities))}")
    
    return predicted_class, probabilities


# ==================== 主流程函数 ====================
def main_pipeline(image_path: str, 
                  model: Optional[MLPClassifier] = None, 
                  scaler: Optional[StandardScaler] = None) -> Dict:
    """
    完整的主流程：从图像输入到干旱等级预测
    
    参数:
        image_path: 红外图像路径
        model: 预训练MLP模型（可选）
        scaler: 特征标准化器（可选）
        
    返回:
        results: 包含所有中间结果和最终预测的字典
    """
    print("\n" + "="*60)
    print("开始执行基于小波边界增强的红外玉米干旱识别")
    print("="*60)
    
    results = {}
    
    # S1: 加载图像
    X = load_infrared_image(image_path)
    results['original_image'] = X
    
    # S2: 小波边界增强
    E, B = step_s2_wavelet_enhancement(X)
    results['energy_map'] = E
    results['boundary_map'] = B
    
    # S3: 叶片区域分割
    X1 = segment_leaf_region(X, B)
    results['leaf_mask'] = X1
    
    # S4: 连通域分析和单叶片分割
    leaf_masks = step_s4_leaf_segmentation(X1, B)
    results['single_leaf_masks'] = leaf_masks
    
    # S5-S6: 对每个叶片计算特征
    feature_vectors = []
    curvature_list = []
    
    for i, leaf_mask in enumerate(leaf_masks):
        print(f"\n处理第{i+1}个叶片...")
        
        # S5: 弯曲度量化
        K, centerline = step_s5_curvature_quantification(leaf_mask)
        
        if K is None or len(K) < 3:
            print(f"  跳过叶片{i+1}: 无法提取有效曲率")
            continue
        
        curvature_list.append({'K': K, 'centerline': centerline})
        
        # S6: 构建特征向量
        f = build_feature_vector(K, centerline)
        
        if f is not None:
            feature_vectors.append(f)
            results[f'leaf_{i}_features'] = f
    
    results['feature_vectors'] = feature_vectors
    results['curvature_data'] = curvature_list
    
    # S7: 预测（如果提供了模型）
    predictions = []
    if model is not None and scaler is not None and feature_vectors:
        print("\n" + "="*50)
        print("开始执行S7: 干旱等级预测")
        print("="*50)
        
        for i, f in enumerate(feature_vectors):
            pred_class, probs = predict_drought_level(model, scaler, f)
            predictions.append({
                'leaf_id': i,
                'class': pred_class,
                'probabilities': probs
            })
    
    results['predictions'] = predictions
    
    print("\n" + "="*60)
    print("识别流程完成")
    print("="*60)
    
    return results


def visualize_results(results: Dict, save_path: str = None):
    """
    可视化处理结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # 原始图像
    axes[0].imshow(results['original_image'], cmap='gray')
    axes[0].set_title('S1: 原始红外图像')
    axes[0].axis('off')
    
    # 能量图
    axes[1].imshow(results['energy_map'], cmap='hot')
    axes[1].set_title('S21: 高频能量图')
    axes[1].axis('off')
    
    # 边界响应图
    axes[2].imshow(results['boundary_map'], cmap='gray')
    axes[2].set_title('S23: 边界响应图')
    axes[2].axis('off')
    
    # 叶片掩膜
    axes[3].imshow(results['leaf_mask'], cmap='gray')
    axes[3].set_title('S3: 叶片区域掩膜')
    axes[3].axis('off')
    
    # 单叶片分割结果
    single_leaf_vis = np.zeros_like(results['original_image'])
    for i, mask in enumerate(results['single_leaf_masks'][:5]):  # 最多显示5个
        color = int(255 / (i + 1))
        single_leaf_vis[mask > 0] = color
    axes[4].imshow(single_leaf_vis, cmap='nipy_spectral')
    axes[4].set_title('S4: 单叶片分割')
    axes[4].axis('off')
    
    # 中心线和曲率示例
    if 'curvature_data' in results and len(results['curvature_data']) > 0:
        # 选择第一个叶片的中心线
        leaf_vis = results['single_leaf_masks'][0].copy()
        centerline = results['curvature_data'][0]['centerline']
        
        # 绘制中心线
        for pt in centerline:
            cv2.circle(leaf_vis, tuple(pt.astype(int)), 1, 255, -1)
        
        axes[5].imshow(leaf_vis, cmap='gray')
        axes[5].set_title('S5: 中心线提取示例')
        axes[5].axis('off')
    else:
        axes[5].imshow(np.zeros_like(results['original_image']), cmap='gray')
        axes[5].set_title('S5: 无有效叶片')
        axes[5].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")
    
    plt.show()


# ==================== 示例和测试 ====================
def generate_synthetic_data(num_samples: int = 200) -> Tuple[List[np.ndarray], List[int]]:
    """
    生成合成训练数据（用于演示）
    实际应用中应使用真实标注数据
    """
    print(f"生成{num_samples}个合成训练样本...")
    
    feature_vectors = []
    labels = []
    
    # 模拟不同干旱程度的特征分布
    for i in range(num_samples):
        # 随机生成类别
        label = np.random.choice([0, 1, 2])
        
        # 根据类别生成不同的特征分布
        if label == 0:  # 不干旱
            f = np.random.normal([0.1, 0.05, 0.2, 0.9, 0.1], 
                               [0.02, 0.01, 0.05, 0.05, 0.02])
        elif label == 1:  # 轻度干旱
            f = np.random.normal([0.3, 0.1, 0.5, 0.7, 0.3], 
                               [0.05, 0.02, 0.1, 0.1, 0.05])
        else:  # 重度干旱
            f = np.random.normal([0.6, 0.2, 0.9, 0.4, 0.6], 
                               [0.1, 0.05, 0.15, 0.15, 0.1])
        
        feature_vectors.append(np.maximum(f, 0))  # 确保非负
        labels.append(label)
    
    return feature_vectors, labels


def demo():
    """
    完整的演示流程
    """
    # 1. 生成训练数据并训练模型（仅用于演示）
    print("步骤1: 生成合成训练数据并训练MLP模型")
    train_features, train_labels = generate_synthetic_data(num_samples=200)
    model, scaler = train_mlp_classifier(train_features, train_labels)
    
    # 2. 创建一个示例红外图像（或使用真实图像）
    # 实际应用中替换为真实红外图像路径
    print("\n步骤2: 创建示例红外图像")
    # 生成一个模拟的玉米叶片红外图像
    H, W = 512, 512
    synthetic_image = np.random.randint(100, 150, (H, W), dtype=np.uint8)
    
    # 添加几个模拟叶片区域
    # 叶片1: 正常叶片
    cv2.ellipse(synthetic_image, (150, 200), (80, 30), 0, 0, 360, 180, -1)
    # 叶片2: 轻度卷曲
    cv2.ellipse(synthetic_image, (350, 180), (70, 25), -20, 0, 360, 160, -1)
    cv2.ellipse(synthetic_image, (360, 160), (30, 15), 45, 0, 360, 160, -1)
    # 叶片3: 重度卷曲
    pts = np.array([[250, 350], [280, 320], [300, 350], [290, 380], [260, 400]], np.int32)
    cv2.fillPoly(synthetic_image, [pts], 140)
    
    # 添加噪声
    noise = np.random.normal(0, 10, synthetic_image.shape)
    synthetic_image = np.clip(synthetic_image + noise, 0, 255).astype(np.uint8)
    
    # 保存临时图像
    temp_path = "temp_infrared_image.png"
    cv2.imwrite(temp_path, synthetic_image)
    print(f"示例图像已保存至: {temp_path}")
    
    # 3. 执行识别流程
    print("\n步骤3: 执行干旱识别流程")
    results = main_pipeline(temp_path, model, scaler)
    
    # 4. 可视化结果
    print("\n步骤4: 可视化结果")
    visualize_results(results, save_path="drought_detection_result.png")
    
    return results, model, scaler


if __name__ == "__main__":
    # 运行演示
    results, model, scaler = demo()
    
    print("\n" + "="*60)
    print("完整流程执行完毕")
    print(f"检测到 {len(results['single_leaf_masks'])} 个叶片")
    print(f"提取了 {len(results['feature_vectors'])} 个有效特征向量")
    print(f"生成了 {len(results['predictions'])} 个预测结果")
    print("="*60)