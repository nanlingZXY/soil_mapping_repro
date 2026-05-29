# -*- coding: utf-8 -*-
"""
02 普通克里格插值
"""
import pandas as pd
# pyrefly: ignore [missing-import]
from pykrige.ok import OrdinaryKriging
import geopandas as gpd
import os
import warnings
warnings.filterwarnings('ignore')

# ===================== 路径配置 =====================
SAMPLE_PATH = "../data/processed/曾都土壤样点处理后.csv"
GRID_PATH = "../data/raw/曾都渔网点.shp"
# 插值结果保存路径
RESULT_PATH = "../results/output/kriging_result.csv"

# 插值属性
TARGET_ATTR = "有机质"

# ===================== 主程序 =====================
if __name__ == '__main__':
    print("===== 开始克里格插值计算 =====")
    
    # 1. 读取数据
    sample_df = pd.read_csv(SAMPLE_PATH, encoding='utf-8')
    grid_gdf = gpd.read_file(GRID_PATH, encoding='gbk')

    # 2. 克里格插值
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

    z_pred, _ = ok_model.execute(
        "points", grid_x, grid_y,
        backend="loop",
        n_closest_points=10
    )
    
    # 3. 保存插值结果
    grid_gdf["pred_organic"] = z_pred
    os.makedirs("../results/output", exist_ok=True)
    grid_gdf.to_csv(RESULT_PATH, index=False, encoding='utf-8')

    print(" 插值完成！")