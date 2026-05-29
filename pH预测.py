# -*- coding: utf-8 -*-
import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import Affine
import rasterio.warp
from rasterio.enums import Resampling
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import RFECV
from scipy.stats import boxcox, boxcox_normmax, pearsonr
from scipy.special import inv_boxcox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pykrige.ok import OrdinaryKriging
import joblib
import warnings
import shutil
import statistics

warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心参数配置（保留你的数据路径）
# ==========================================
BASE_DIR = r"I:\sxzt\qd"
TRAIN_FILE = os.path.join(BASE_DIR, r"trainx.shp")
VAL_FILE = os.path.join(BASE_DIR, r"testx.shp")
GDB_PATH = os.path.join(BASE_DIR, r"属性制图.gdb")
GDB_LAYER = "空间连接14"

# 目标变量
TARGET_COL = "pH1"
# 环境特征列
FEATURE_COLS_EXPECT = [
    "TRLX_AZRXT", "TRLX_CDAZR", "TRLX_CDZZT",
    "TRLX_DXAZR", "TRLX_DXHZR", "TRLX_DXZR",
    "TRLX_HSSHT", "TRLX_HZRXT", "TRLX_HCT",
    "TRLX_SHXZS", "TRLX_SXZST", "TRLX_YYSDT",
    "TRLX_ZXZST", "TRLX_ZYSDT", "TRLX_ZRXT",
    "TRLX_ZSSHT", "TDLY_GMLD", "TDLY_GY",
    "TDLY_HD", "TDLY_CY", "TDLY_ST",
    "TDLY_QTCD", "TDLY_QTLD", "TDLY_QTYD",
    "TDLY_QMLD", "TDLY_SJD", "TDLY_TRMCD",
    "TDLY_ZZCD", "TDLY_ZLD", 
    "DX_DEM", "DX_SLOPE", 
    "ZB_NDVI", "ZB_EVI",
    "QH_PRE", "QH_TEM", "MZ_ANZ", "MZ_CN", "MZ_GZ", "MZ_HLCJW",
    "MZ_HN", "MZ_HNZ", "MZ_NZ", "MZ_SNZ", "MZ_SN", "MZ_TSYL",
    "MZ_ZTZ"
]

# 字段映射（解决命名不一致）
FIELD_MAPPING = {
    "DX_ROUGHNE": "DX_ROUGHNESS",
    "TRLX_HZR": "TRLX_HZRXT",
}

# 输出配置
OUT_DIR = os.path.join(BASE_DIR, r"output\pH2013_Final1")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[输出路径] 已创建/确认输出目录: {OUT_DIR}")

# 输出文件路径
MODEL_PATH = os.path.join(OUT_DIR, "rf_model_pH1.joblib")
PRED_RF_TIF = os.path.join(OUT_DIR, "predicted_pH1_RF.tif")
PRED_RFK_TIF = os.path.join(OUT_DIR, "predicted_pH1_RFK.tif")
FEATURE_IMPORTANCE_CSV = os.path.join(OUT_DIR, "feature_importance.csv")
PRED_CSV = os.path.join(OUT_DIR, "pH1_prediction_results.csv")

# 核心配置
RASTER_RESOLUTION = 30
NODATA_VALUE = -9999.0
BOXCOX_SHIFT = 0.01  # 避免0值影响Box-Cox变换
PRED_CHUNK_SIZE = 100000  # 分块预测大小
MAX_RETRY = 3
RETRY_DELAY = 2
FEATURE_IMPORTANCE_THRESHOLD = 0.001
COORD_TOLERANCE = 1.0  # 坐标匹配容差

# ==========================================
# 2. 工具函数（整合参考代码的核心逻辑）
# ==========================================
def is_file_locked(filepath):
    """检查文件是否被占用"""
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, 'a'):
            pass
        return False
    except PermissionError:
        return True

def release_locked_file(filepath):
    """释放被占用的文件"""
    if not os.path.exists(filepath):
        return True
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for handle in proc.open_files():
                    if filepath.lower() in handle.path.lower():
                        print(f"[文件释放] 结束占用进程 {proc.name()} (PID: {proc.pid})")
                        proc.terminate()
                        time.sleep(1)
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        if os.path.exists(filepath):
            os.chmod(filepath, 0o777)
            os.remove(filepath)
            print(f"[文件释放] 成功删除锁定文件: {filepath}")
        return True
    except Exception as e:
        print(f"[警告] 释放文件失败: {e}")
        return False

def safe_remove_file(filepath):
    """安全删除文件"""
    if not os.path.exists(filepath):
        return True
    try:
        os.chmod(filepath, 0o777)
        os.remove(filepath)
        return True
    except PermissionError:
        return release_locked_file(filepath)
    except Exception as e:
        print(f"[警告] 删除文件 {filepath} 失败: {e}")
        return False

def safe_write_raster(filepath, data, meta, max_retry=MAX_RETRY):
    """安全写入栅格（兼容ArcMap金字塔）"""
    temp_file = filepath + ".tmp"
    safe_remove_file(temp_file)
    for ext in ['', '.ovr', '.aux.xml', '.tif.ovr']:
        safe_remove_file(filepath + ext)
    
    # 关键：启用分块+float32+DEFLATE压缩（ArcMap兼容）
    meta['tiled'] = True
    meta['blockxsize'] = 256
    meta['blockysize'] = 256
    meta['compress'] = 'DEFLATE'
    meta['predictor'] = 1
    meta['dtype'] = np.float32
    meta['BIGTIFF'] = 'IF_SAFER'
    
    data = data.astype(np.float32)
    data[np.isnan(data)] = NODATA_VALUE
    
    retry_count = 0
    while retry_count < max_retry:
        try:
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                GDAL_TIFF_OVR_BLOCKSIZE=128
            ):
                with rasterio.open(temp_file, "w", **meta) as dst:
                    dst.write(data, 1)
                    # 提前构建金字塔
                    overviews = [2, 4, 8, 16, 32]
                    dst.build_overviews(overviews, Resampling.average)
            if os.path.exists(temp_file):
                shutil.move(temp_file, filepath)
                print(f"[栅格保存] 成功写入: {filepath}")
                return True
        except Exception as e:
            retry_count += 1
            print(f"[警告] 写入栅格失败 (重试 {retry_count}/{max_retry}): {e}")
            time.sleep(RETRY_DELAY)
            safe_remove_file(temp_file)
            continue
    print(f"[错误] 多次重试后仍无法写入栅格文件: {filepath}")
    return False

def remove_outliers(gdf, col):
    """3σ原则移除异常值"""
    mean_val = gdf[col].mean()
    std_val = gdf[col].std()
    lower, upper = mean_val - 3 * std_val, mean_val + 3 * std_val
    clean_gdf = gdf[(gdf[col] >= lower) & (gdf[col] <= upper)].copy()
    print(f"[数据清洗] {col} - 原始样点: {len(gdf)} -> 清洗后: {len(clean_gdf)}")
    print(f"[数据分布] {col}范围: {clean_gdf[col].min():.2f} - {clean_gdf[col].max():.2f}")
    return clean_gdf

def get_nearest_env_value(point_gdf, env_gdf, feature_cols):
    """最近邻补充缺失环境变量"""
    if point_gdf.crs != env_gdf.crs:
        point_gdf = point_gdf.to_crs(env_gdf.crs)
    
    point_coords = np.array([(p.x, p.y) for p in point_gdf.geometry])
    env_coords = np.array([(p.x, p.y) for p in env_gdf.geometry])
    
    from scipy.spatial import cKDTree
    tree = cKDTree(env_coords)
    distances, indices = tree.query(point_coords)
    
    env_vals = env_gdf.iloc[indices][feature_cols].reset_index(drop=True)
    env_vals.columns = feature_cols
    
    print(f"[最近邻匹配] 补充字段数: {len(feature_cols)} | 最大匹配距离: {distances.max():.2f} 米")
    return point_gdf.reset_index(drop=True).join(env_vals)

def match_and_supplement_fields(train_gdf, val_gdf, env_gdf, expect_cols, field_mapping):
    """字段匹配与补充"""
    train_gdf.rename(columns=field_mapping, inplace=True)
    val_gdf.rename(columns=field_mapping, inplace=True)
    
    env_cols = [col for col in expect_cols if col in env_gdf.columns]
    print(f"\n[字段匹配] GDB中存在的期望字段数: {len(env_cols)}/{len(expect_cols)}")
    
    train_have = [col for col in env_cols if col in train_gdf.columns]
    train_lack = [col for col in env_cols if col not in train_gdf.columns]
    val_have = [col for col in env_cols if col in val_gdf.columns]
    val_lack = [col for col in env_cols if col not in val_gdf.columns]
    
    print(f"[字段匹配] 训练集 - 已有: {len(train_have)} | 缺失: {train_lack}")
    print(f"[字段匹配] 验证集 - 已有: {len(val_have)} | 缺失: {val_lack}")
    
    if train_lack:
        train_gdf = get_nearest_env_value(train_gdf, env_gdf, train_lack)
    if val_lack:
        val_gdf = get_nearest_env_value(val_gdf, env_gdf, val_lack)
    
    final_feats = [col for col in env_cols if col in train_gdf.columns and col in val_gdf.columns]
    print(f"\n[字段匹配] 最终可用特征数: {len(final_feats)}")
    print(f"[字段匹配] 最终特征列表: {final_feats}")
    
    return train_gdf, val_gdf, final_feats

def feature_selection(X, y, feature_names):
    """参考代码的特征筛选逻辑（VIF+RFECV）- 调整为保留约15个特征"""
    print("\n===== 特征筛选（VIF+RFECV） =====")
    # 1. 计算VIF - 放宽阈值从10到15，保留更多特征
    VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    VIF_SELECT = [index for index, value in enumerate(VIF_list) if value < 15]  # 关键修改：VIF阈值从10→15
    
    # 2. RFECV筛选 - 保持原有逻辑，但后续合并时优先保留
    regressor = RandomForestRegressor(n_estimators=150, random_state=42)
    rfecv = RFECV(estimator=regressor, step=1, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    rfecv_model = rfecv.fit(X, y)
    
    # 3. 合并筛选结果（VIF通过 + RFECV通过）
    rfecv_selected = rfecv_model.get_support(indices=True)
    final_selected_idx = list(set(VIF_SELECT) & set(rfecv_selected))
    
    # 4. 过滤低有效值特征 - 放宽0值占比从80%到90%
    X_temp = X[:, final_selected_idx]
    feature_names_temp = [feature_names[i] for i in final_selected_idx]
    # 关键修改：0值占比阈值从0.8→0.9，更少特征被剔除
    selct_index = (X_temp == 0).sum(axis=0) < np.round(len(X_temp) * 0.9)
    final_selected_idx = [final_selected_idx[i] for i in range(len(final_selected_idx)) if selct_index[i]]
    final_features = [feature_names[i] for i in final_selected_idx]
    
    # 5. 兜底逻辑：如果筛选后仍不足15个，按特征重要性补充到15个
    if len(final_features) < 15:
        # 计算所有VIF通过特征的重要性，补充到15个
        rf_import = RandomForestRegressor(n_estimators=150, random_state=42)
        rf_import.fit(X[:, VIF_SELECT], y)
        importances = rf_import.feature_importances_
        
        # 按重要性排序VIF通过的特征
        vif_feature_names = [feature_names[i] for i in VIF_SELECT]
        vif_feature_importance = list(zip(vif_feature_names, importances))
        vif_feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 补充特征到15个（排除已选中的，避免重复）
        selected_set = set(final_features)
        for feat, _ in vif_feature_importance:
            if feat not in selected_set and len(final_features) < 15:
                final_features.append(feat)
        # 重新获取最终特征的索引
        final_selected_idx = [feature_names.index(feat) for feat in final_features]
    
    # 6. 重新训练特征重要性并保存
    rf_import = RandomForestRegressor(n_estimators=150, random_state=42)
    rf_import.fit(X[:, final_selected_idx], y)
    importances = rf_import.feature_importances_
    
    # 保存特征重要性
    importance_df = pd.DataFrame({
        'feature': final_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False, encoding='utf-8-sig')
    
    print(f"[特征筛选] 原始特征数: {len(feature_names)} -> 筛选后: {len(final_features)}")
    print(f"[特征筛选] 筛选后特征: {final_features}")
    
    return X[:, final_selected_idx], final_features

def create_raster_template_from_gdf(gdf, resolution):
    """生成栅格模板（ArcMap兼容）"""
    minx, miny, maxx, maxy = gdf.total_bounds
    
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    
    # 左上角对齐（ArcMap默认）
    transform = Affine(
        resolution, 0.0, minx,
        0.0, -resolution, maxy
    )
    
    print(f"\n[栅格模板] 基于GDB真实边界生成:")
    print(f"[栅格模板] GDB边界: {minx:.6f},{miny:.6f} -> {maxx:.6f},{maxy:.6f}")
    print(f"[栅格模板] 栅格尺寸: {width}列 x {height}行 | 分辨率: {resolution}米")
    print(f"[栅格模板] Transform参数: a={transform.a:.6f}, b={transform.b:.6f}, c={transform.c:.6f}")
    print(f"[栅格模板] Transform参数: d={transform.d:.6f}, e={transform.e:.6f}, f={transform.f:.6f}")
    
    meta = {
        'driver': 'GTiff', 'height': height, 'width': width, 'count': 1,
        'dtype': np.float32, 'crs': gdf.crs, 'transform': transform,
        'nodata': NODATA_VALUE, 'BIGTIFF': 'IF_SAFER'
    }
    return transform, width, height, meta

def vector_points_to_raster(points_gdf, value_col, transform, width, height, nodata):
    """矢量转栅格（修复异常值）"""
    raster_array = np.full((height, width), nodata, dtype=np.float32)
    
    coords = np.array([(geom.x, geom.y) for geom in points_gdf.geometry])
    values = points_gdf[value_col].values.astype(np.float32)
    
    if len(coords) == 0:
        print("[警告] 无采样点数据！")
        return raster_array
    
    # 严格边界过滤
    raster_minx = transform.c
    raster_maxx = transform.c + width * transform.a
    raster_maxy = transform.f
    raster_miny = transform.f + height * transform.e
    
    in_bounds_mask = (
        (coords[:, 0] >= raster_minx) & (coords[:, 0] < raster_maxx) &
        (coords[:, 1] > raster_miny) & (coords[:, 1] <= raster_maxy)
    )
    coords_in_bounds = coords[in_bounds_mask]
    values_in_bounds = values[in_bounds_mask]
    
    print(f"[矢量转栅格] 总GDB采样点: {len(points_gdf)} | 栅格范围内点: {len(coords_in_bounds)}")
    if len(coords_in_bounds) == 0:
        print("[警告] 没有GDB采样点落在栅格范围内！")
        return raster_array
    
    # 转换行列号并校验
    rows, cols = rasterio.transform.rowcol(transform, coords_in_bounds[:, 0], coords_in_bounds[:, 1])
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    
    valid_mask = (
        (rows >= 0) & (rows < height) &
        (cols >= 0) & (cols < width)
    )
    valid_rows = rows[valid_mask]
    valid_cols = cols[valid_mask]
    valid_values = values_in_bounds[valid_mask]
    
    # 赋值（仅有效位置）
    raster_array[valid_rows, valid_cols] = valid_values
    
    # 过滤pH异常值（4.0-9.0）
    valid_pixel_mask = raster_array != nodata
    abnormal_mask = (raster_array < 4.0) | (raster_array > 9.0)
    raster_array[abnormal_mask & valid_pixel_mask] = nodata
    
    # 统计
    valid_pixels = np.sum(raster_array != nodata)
    print(f"[矢量转栅格] 有效像素数: {valid_pixels}/{width*height}")
    if valid_pixels > 0:
        print(f"[矢量转栅格] 最终预测值范围: {np.min(raster_array[raster_array!=nodata]):.2f} - {np.max(raster_array[raster_array!=nodata]):.2f}")
    
    return raster_array

def predict_in_chunks(model, X_data, lambda_opt, chunk_size):
    """分块预测（避免内存溢出）"""
    all_preds = []
    total = len(X_data)
    print(f"[全量预测] 开始分块预测，总数据量: {total}，分块大小: {chunk_size}")
    
    for i in range(0, total, chunk_size):
        end_idx = min(i + chunk_size, total)
        X_chunk = X_data[i:end_idx]
        
        # 预测并逆变换
        pred_trans = model.predict(X_chunk)
        pred_raw = inv_boxcox(pred_trans, lambda_opt) - BOXCOX_SHIFT
        
        all_preds.append(pred_raw)
        print(f"[全量预测] 完成 {end_idx}/{total} 条数据预测")
    
    return np.hstack(all_preds)

# ==========================================
# 3. 主执行逻辑（整合参考代码的RF+RFK核心）
# ==========================================
def main():
    """主执行函数"""
    try:
        # 1. 加载数据
        print("===== 加载数据 =====")
        train_gdf = gpd.read_file(TRAIN_FILE)
        val_gdf = gpd.read_file(VAL_FILE)
        env_gdf = gpd.read_file(GDB_PATH, layer=GDB_LAYER)
        
        print(f"训练集CRS: {train_gdf.crs}")
        print(f"GDB图层CRS: {env_gdf.crs}")
        print(f"训练集样本数: {len(train_gdf)}")
        print(f"GDB图层要素数: {len(env_gdf)}")
        
        # 2. 数据预处理
        # 清洗目标变量
        train_gdf = remove_outliers(train_gdf, TARGET_COL)
        val_gdf = remove_outliers(val_gdf, TARGET_COL)
        
        # 匹配并补充字段
        train_gdf, val_gdf, final_feats = match_and_supplement_fields(
            train_gdf, val_gdf, env_gdf, FEATURE_COLS_EXPECT, FIELD_MAPPING
        )
        
        # 清理缺失值
        train_gdf = train_gdf.dropna(subset=final_feats + [TARGET_COL])
        val_gdf = val_gdf.dropna(subset=final_feats + [TARGET_COL])
        
        # 提取特征和标签
        X_train = train_gdf[final_feats].values
        y_train = train_gdf[TARGET_COL].values
        X_val = val_gdf[final_feats].values
        y_val = val_gdf[TARGET_COL].values
        
        print(f"\n===== 数据准备完成 =====")
        print(f"训练集有效样本: {len(X_train)}")
        print(f"验证集有效样本: {len(X_val)}")
        print(f"特征数: {len(final_feats)}")
        
        # 3. 特征筛选（VIF+RFECV）
        X_train_selected, selected_features = feature_selection(X_train, y_train, final_feats)
        X_val_selected = X_val[:, [final_feats.index(f) for f in selected_features]]
        
        # 4. Box-Cox变换（参考代码核心，修复解包错误）
        print("\n===== Box-Cox变换 =====")
        y_train_shift = y_train + BOXCOX_SHIFT
        y_val_shift = y_val + BOXCOX_SHIFT
        
        # 计算最优lambda
        lambda_opt = boxcox_normmax(y_train_shift)
        # 修复：指定lmbda参数时，boxcox仅返回变换后的数据
        y_train_trans = boxcox(y_train_shift, lmbda=lambda_opt)
        y_val_trans = boxcox(y_val_shift, lmbda=lambda_opt)
        
        print(f"Box-Cox最优lambda: {lambda_opt:.4f}")
        print(f"变换后训练集范围: {y_train_trans.min():.2f} - {y_train_trans.max():.2f}")
        
        # 5. 随机森林网格搜索调参（参考代码核心）
        print("\n===== 随机森林网格搜索调参 =====")
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 150, 200, 300],
            "max_depth": [2, 3, 5, 15, 20, 25, 30],
            "max_features": [2, 4, 6, 8, 10]
        }
        
        # 网格搜索+5折交叉验证
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_selected, y_train_trans)
        
        # 最优模型
        best_rf = grid_search.best_estimator_
        print(f"最优参数: {grid_search.best_params_}")
        
        # 6. 模型验证（参考代码的逆变换+评价）
        print("\n===== 模型验证 =====")
        y_train_pred_trans = best_rf.predict(X_train_selected)
        y_val_pred_trans = best_rf.predict(X_val_selected)
        
        # Box-Cox逆变换（修复：使用lambda_opt而非lmbda）
        y_train_pred = inv_boxcox(y_train_pred_trans, lambda_opt) - BOXCOX_SHIFT
        y_val_pred = inv_boxcox(y_val_pred_trans, lambda_opt) - BOXCOX_SHIFT
        
        # 计算评价指标
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        print(f"训练集 R²: {train_r2:.4f} | RMSE: {train_rmse:.4f}")
        print(f"验证集 R²: {val_r2:.4f} | RMSE: {val_rmse:.4f}")
        
        # 7. 保存最优模型
        safe_remove_file(MODEL_PATH)
        joblib.dump(best_rf, MODEL_PATH)
        print(f"\n最优模型已保存至: {MODEL_PATH}")
        
        # 8. 全量GDB预测
        print("\n===== 全量GDB数据预测 =====")
        env_gdf_clean = env_gdf.dropna(subset=final_feats)
        print(f"GDB清洗后有效数据量: {len(env_gdf_clean)}")
        
        # 提取筛选后的特征
        X_pred = env_gdf_clean[selected_features].values
        
        # 分块预测
        y_rf_pred = predict_in_chunks(best_rf, X_pred, lambda_opt, PRED_CHUNK_SIZE)
        env_gdf_clean['pred_rf'] = y_rf_pred
        
        # 值域过滤（pH合理范围4.0-9.0）
        env_gdf_clean['pred_rf'] = env_gdf_clean['pred_rf'].clip(4.0, 9.0)
        print(f"全量预测完成，预测值范围: {env_gdf_clean['pred_rf'].min():.2f} - {env_gdf_clean['pred_rf'].max():.2f}")
        
        # 9. 残差克里格（RFK，参考代码核心，修复参数错误）
        print("\n===== 随机森林克里格（RFK） =====")
        # 计算训练集残差
        train_res = y_train - y_train_pred
        
        # 训练集坐标
        train_coords = list(zip(train_gdf.geometry.x, train_gdf.geometry.y))
        
        # 构建克里格模型
        try:
            ok = OrdinaryKriging(
                np.array([c[0] for c in train_coords], dtype=np.float32),
                np.array([c[1] for c in train_coords], dtype=np.float32),
                train_res.astype(np.float32),
                variogram_model="gaussian",
                nlags=12,
                weight=True,
                verbose=False,
                enable_plotting=False
            )
            
            # 对GDB点进行残差插值（修复style参数错误）
            gdb_coords_x = np.array([geom.x for geom in env_gdf_clean.geometry], dtype=np.float32)
            gdb_coords_y = np.array([geom.y for geom in env_gdf_clean.geometry], dtype=np.float32)
            
            # 修复：移除多余的style参数，使用正确的位置参数传参
            res_pred, _ = ok.execute(
                'points',  # 插值类型（位置参数，不是关键字参数）
                gdb_coords_x,
                gdb_coords_y,
                n_closest_points=8,
                backend="loop"
            )
            
            # RFK预测结果 = RF预测 + 残差插值
            env_gdf_clean['pred_rfk'] = env_gdf_clean['pred_rf'] + res_pred
            # 值域过滤
            env_gdf_clean['pred_rfk'] = env_gdf_clean['pred_rfk'].clip(4.0, 9.0)
            print(f"RFK预测完成，预测值范围: {env_gdf_clean['pred_rfk'].min():.2f} - {env_gdf_clean['pred_rfk'].max():.2f}")
            
        except MemoryError as e:
            print(f"[警告] 克里格内存不足: {e}")
            print(f"[降级策略] RFK = RF预测结果")
            env_gdf_clean['pred_rfk'] = env_gdf_clean['pred_rf']
        except Exception as e:
            print(f"[警告] 克里格插值失败: {e}")
            env_gdf_clean['pred_rfk'] = env_gdf_clean['pred_rf']
        
        # 10. 保存CSV结果（修复X/Y列缺失问题）
        # 新增：从geometry字段提取X/Y坐标
        env_gdf_clean['X'] = env_gdf_clean.geometry.x
        env_gdf_clean['Y'] = env_gdf_clean.geometry.y
        # 保存指定列
        env_gdf_clean[['X', 'Y', 'pred_rf', 'pred_rfk']].to_csv(PRED_CSV, index=False, encoding='utf-8-sig')
        print(f"预测结果CSV已保存至: {PRED_CSV}")
        
        # 11. 矢量转栅格
        transform, width, height, meta = create_raster_template_from_gdf(env_gdf, RASTER_RESOLUTION)
        
        # RF栅格
        full_rf = vector_points_to_raster(
            env_gdf_clean, 'pred_rf', transform, width, height, NODATA_VALUE
        )
        safe_write_raster(PRED_RF_TIF, full_rf, meta)
        
        # RFK栅格
        full_rfk = vector_points_to_raster(
            env_gdf_clean, 'pred_rfk', transform, width, height, NODATA_VALUE
        )
        safe_write_raster(PRED_RFK_TIF, full_rfk, meta)
        
        # 最终统计
        print(f"\n===== 最终结果 =====")
        rf_valid = full_rf[full_rf != NODATA_VALUE]
        if len(rf_valid) > 0:
            print(f"RF预测范围: {np.min(rf_valid):.2f} - {np.max(rf_valid):.2f}")
            print(f"RF有效像素数: {len(rf_valid)}/{width*height}")
        
        rfk_valid = full_rfk[full_rfk != NODATA_VALUE]
        if len(rfk_valid) > 0:
            print(f"RFK预测范围: {np.min(rfk_valid):.2f} - {np.max(rfk_valid):.2f}")
            print(f"RFK有效像素数: {len(rfk_valid)}/{width*height}")
        
        print(f"\n===== 程序执行完成 =====")
        print(f"RF栅格文件: {PRED_RF_TIF}")
        print(f"RFK栅格文件: {PRED_RFK_TIF}")
        
    except Exception as e:
        print(f"\n[ERROR] 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print("===== 开始pH属性预测（整合参考代码逻辑版） =====")
    main()