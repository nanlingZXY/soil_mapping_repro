# -*- coding: utf-8 -*-
"""
03 结果可视化
"""
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os

# ===================== 路径配置 =====================
# 插值结果
RESULT_PATH = "../results/output/kriging_result.csv"
# 原始网格数据
GRID_PATH = "../data/raw/曾都渔网点.shp"
# 采样点数据
SAMPLE_PATH = "../data/processed/曾都土壤样点处理后.csv"
# 图片保存路径
FIG_PATH = "../results/figures/有机质_map.png"

ATTR_CN = "有机质"

# ===================== 绘图设置 =====================
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 主程序 =====================
if __name__ == '__main__':
    print("===== 开始可视化绘图 =====")
    
    # 1. 读取原始网格shp
    grid_gdf = gpd.read_file(GRID_PATH, encoding='gbk')
    # 2. 读取插值结果
    pred_data = pd.read_csv(RESULT_PATH, encoding='utf-8')
    
    # 合并数据
    grid_gdf["pred_organic"] = pred_data["pred_organic"]
    
    # 读取采样点
    sample_df = pd.read_csv(SAMPLE_PATH, encoding='utf-8')

    # 3. 绘图
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
    ax.legend()
    ax.axis("equal")

    # 4. 保存图片
    os.makedirs("../results/figures", exist_ok=True)
    plt.savefig(FIG_PATH, bbox_inches="tight")
    plt.close()

    print(" 绘图完成！图片已保存至 results/figures/")