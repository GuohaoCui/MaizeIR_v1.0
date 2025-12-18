# 基于小波边界增强的红外玉米干旱识别

&gt; 单幅中波红外图像 → 5 维曲率特征 → MLP 三分类（不干旱 / 轻度 / 重度）

## 1. 快速开始
```bash
# 1. 克隆或下载代码
# 2. 安装依赖（建议 Python ≥3.8）
pip install -r requirements.txt

# 3. 运行演示（合成数据 + 合成红外图）
python code.py

# 4. 用自己的红外图
python code.py --image your_IR.png --model drought_mlp.pkl
```
## 2. 核心流程（7 步）

1. 读取 8-bit 中波红外图像  
2. 小波分解 → 高频能量图 → 二值边界响应  
3. 原图引导分割 → 叶片粗掩膜  
4. 连通域 + 几何阈值 → 单叶片掩膜列表  
5. 距离变换提取中心线 → 离散曲率  
6. 5 维曲率特征（均值、标准差、最大值、端点收缩比、归一化弯曲能量）  
7. 轻量级 MLP(64→32→3) 输出干旱等级与概率

## 3. 文件说明

maize_drought_detection.py   # 完整可执行脚本（含 demo）
requirements.txt             # 依赖列表
README.md                    # 本文档

## 4. 依赖列表

numpy>=1.23
opencv-python>=4.7
PyWavelets>=1.4
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.6
scikit-image>=0.20      # watershed 用

### 安装

```
pip install -r requirements.txt
```

## 7. 注意事项

1. 输入请保证为 8-bit 灰度图；彩色图会被强制转灰度
2. 几何阈值（面积、长宽比、紧致度）在首次运行时会按当前图像自动统计均值并设定上下界，如需固定阈值请修改 Config 类
3. 演示用的合成数据仅用于跑通流程，真实场景请替换为已标注的曲率-干旱等级数据再训练

## 8. 许可证

不可商用
