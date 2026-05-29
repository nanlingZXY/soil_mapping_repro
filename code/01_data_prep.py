"""
01 土壤样点数据预处理脚本
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. 配置相对路径 ======================
# 原始土壤样点Excel数据
RAW_SAMPLE_PATH = r"../data/raw/曾都历史数据.xls"
# 插值网格数据
RAW_GRID_PATH = r"../data/raw/曾都渔网点.shp"
# 处理后数据保存文件夹
PROCESSED_DIR = r"../data/processed"

# 处理后文件路径
OUTPUT_SAMPLE_SHP = os.path.join(PROCESSED_DIR, "曾都土壤样点处理后.shp")
OUTPUT_SAMPLE_CSV = os.path.join(PROCESSED_DIR, "曾都土壤样点处理后.csv")
OUTPUT_GRID_SHP = os.path.join(PROCESSED_DIR, "曾都渔网点处理后.shp")

# 目标属性列（可修改：有机质、PH值、碱解氮、有效磷、速效钾）
TARGET_ATTR = "有机质"
# 坐标列
X_COL = "X"
Y_COL = "Y"

# ====================== 2. 创建processed文件夹 ======================
os.makedirs(PROCESSED_DIR, exist_ok=True)
print(f"✅ 已创建处理后数据文件夹：{PROCESSED_DIR}")

# ====================== 3. 读取并预处理土壤样点数据 ======================
print("\n📥 正在读取原始土壤样点数据...")
try:
    # 读取Excel数据
    raw_df = pd.read_excel(RAW_SAMPLE_PATH, sheet_name="曾都历史数据")
    print(f"📊 原始样点数据：")
    print(f"   - 总记录数：{len(raw_df)}")
    print(f"   - 核心列名：{[X_COL, Y_COL, TARGET_ATTR, 'PH值', '碱解氮', '有效磷', '速效钾']}")
except Exception as e:
    print(f"❌ 读取Excel数据失败：{e}")
    exit(1)

# 3.1 数据清洗：删除缺失值
print("\n🧹 数据清洗：删除缺失值...")
processed_df = raw_df.dropna(subset=[X_COL, Y_COL, TARGET_ATTR])
print(f"   - 清洗后记录数：{len(processed_df)}")

# 3.2 去重
print("🗑️ 去除重复数据...")
initial_count = len(processed_df)
processed_df = processed_df.drop_duplicates(subset=[X_COL, Y_COL, TARGET_ATTR], keep="first")
print(f"   - 去除重复数：{initial_count - len(processed_df)}，剩余记录数：{len(processed_df)}")

# 3.3 异常值处理
print("🚫 处理异常值...")
values = processed_df[TARGET_ATTR]
mean_val = values.mean()
std_val = values.std()
lower_bound = mean_val - 3 * std_val
upper_bound = mean_val + 3 * std_val

initial_count = len(processed_df)
processed_df = processed_df[(processed_df[TARGET_ATTR] >= lower_bound) & 
                             (processed_df[TARGET_ATTR] <= upper_bound)]
print(f"   - 删除异常值：{initial_count - len(processed_df)} 条，剩余记录数：{len(processed_df)}")

# 3.4 转换为GeoDataFrame
print("\n🌐 转换为空间数据格式...")
gdf = gpd.GeoDataFrame(
    processed_df,
    geometry=gpd.points_from_xy(processed_df[X_COL], processed_df[Y_COL]),
    crs="EPSG:4547"  
)

gdf[f"pred_{TARGET_ATTR}"] = None
gdf.rename(columns={TARGET_ATTR: "organic"}, inplace=True) 

# ====================== 4. 读取并预处理渔网点数据 ======================
print("\n📥 正在读取插值网格数据...")
try:
    grid_gdf = gpd.read_file(RAW_GRID_PATH)
    print(f"📊 网格数据信息：")
    print(f"   - 总网格点数：{len(grid_gdf)}")
    print(f"   - 坐标系：{grid_gdf.crs}")
except Exception as e:
    print(f"❌ 读取网格数据失败：{e}")
    exit(1)

# 4.1 确保网格和样点坐标系一致
if grid_gdf.crs != gdf.crs:
    print("🔄 统一网格坐标系与样点数据...")
    grid_gdf = grid_gdf.to_crs(gdf.crs)

# ====================== 5. 保存处理后的数据 ======================
print("\n💾 正在保存处理后的数据...")
try:
    # 保存样点数据为Shapefile
    gdf.to_file(OUTPUT_SAMPLE_SHP, driver="ESRI Shapefile", encoding="utf-8")
    # 同时保存CSV格式
    processed_df.to_csv(OUTPUT_SAMPLE_CSV, index=False, encoding="utf-8")
    # 保存网格数据
    grid_gdf.to_file(OUTPUT_GRID_SHP, driver="ESRI Shapefile", encoding="utf-8")
    
    print(f"✅ 样点数据已保存：{OUTPUT_SAMPLE_SHP}")
    print(f"✅ CSV数据已保存：{OUTPUT_SAMPLE_CSV}")
    print(f"✅ 网格数据已保存：{OUTPUT_GRID_SHP}")
except Exception as e:
    print(f"❌ 保存数据失败：{e}")
    exit(1)

# ====================== 6. 生成预处理报告 ======================
print("\n📋 数据预处理完成报告：")
print(f"   - 最终样点数量：{len(gdf)}")
print(f"   - 目标属性：{TARGET_ATTR}")
print(f"   - 数据范围：")
print(f"     X：{gdf[X_COL].min():.2f} ~ {gdf[X_COL].max():.2f}")
print(f"     Y：{gdf[Y_COL].min():.2f} ~ {gdf[Y_COL].max():.2f}")
print(f"     {TARGET_ATTR}：{gdf['organic'].min():.2f} ~ {gdf['organic'].max():.2f}")

# 生成预处理结果图
fig, ax = plt.subplots(figsize=(10, 8))
gdf.plot(ax=ax, column="organic", cmap="YlOrRd", legend=True, legend_kwds={"label": f"{TARGET_ATTR} 含量"})
plt.title(f"曾都区土壤样点数据预处理结果（{TARGET_ATTR}）", fontsize=14)
plt.xlabel("X 坐标")
plt.ylabel("Y 坐标")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(PROCESSED_DIR, "预处理结果图.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"预处理结果图已保存至：{os.path.join(PROCESSED_DIR, '预处理结果图.png')}")
print("\n🎉 数据预处理完成！")