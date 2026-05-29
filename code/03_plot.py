# -*- coding: utf-8 -*-
"""
03 结果可视化
"""
# -*- coding: utf-8 -*-
"""
普通克里格土壤属性插值完整代码
"""

import pandas as pd
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import geopandas as gpd
import os
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局设置 =====================
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 配置路径 =====================
# 样点数据：读取预处理后的CSV
SAMPLE_PATH = "D:/219/wzh/data/processed/曾都土壤样点处理后.csv"
# 网格数据：读取SHP
GRID_PATH = "D:/219/wzh/data/raw/曾都渔网点.shp"

# 属性配置
TARGET_ATTR = "速效钾"
ATTR_CN = "速效钾"

# ===================== 4读取预处理后数据 =====================
print("正在读取预处理后土壤样点数据...")
sample_df = pd.read_csv(SAMPLE_PATH, encoding='utf-8')
print(f"✅ 有效样点数量：{len(sample_df)}")

print("正在读取插值网格数据...")
grid_gdf = gpd.read_file(GRID_PATH, encoding='gbk')
print(f"✅ 插值网格数量：{len(grid_gdf)}")

# ===================== 普通克里格插值 =====================
print("开始执行普通克里格插值...")
x = sample_df["X"].values
y = sample_df["Y"].values
z = sample_df[TARGET_ATTR].values

ok_model = OrdinaryKriging(
    x, y, z,
    variogram_model="gaussian",
    verbose=False,
    enable_plotting=False
)

grid_x = grid_gdf["X"].values
grid_y = grid_gdf["Y"].values

z_pred, z_var = ok_model.execute(
    "points", grid_x, grid_y,
    backend="loop",
    n_closest_points=10
)

grid_gdf["pred_organic"] = z_pred

# ===================== 结果可视化 =====================
print("正在生成空间分布图...")
fig, ax = plt.subplots(figsize=(12, 10))

grid_gdf.plot(
    column="pred_organic",
    cmap="YlGnBu",
    legend=True,
    legend_kwds={"label": f"{ATTR_CN} 含量"},
    ax=ax,
    markersize=1
)

ax.scatter(sample_df["X"], sample_df["Y"], c="red", s=6, label="土壤采样点")
ax.set_title(f"曾都区土壤{ATTR_CN}普通克里格插值结果", fontsize=16)
ax.set_xlabel("X 投影坐标")
ax.set_ylabel("Y 投影坐标")
ax.legend()
ax.axis("equal")

# 保存结果
os.makedirs("D:/219/wzh/data/JG2", exist_ok=True)
os.makedirs("D:/219/wzh/data/JG2", exist_ok=True)
plt.savefig(f"D:/219/wzh/data/JG2/{ATTR_CN}_map.png", bbox_inches="tight")
plt.close()

print("普通克里格插值全部完成！")
print(f"插值属性：{ATTR_CN}")
print(f"图片保存：results/figures/{ATTR_CN}_map.png")
