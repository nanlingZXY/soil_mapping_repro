# -*- coding: utf-8 -*-
"""
02 普通克里格插值
"""
def ordinary_kriging_interpolation(
    sample_df,
    grid_gdf,
    target_attr="有机质",
    variogram_model="gaussian",
    n_closest_points=10
):

    # 提取样点坐标与属性
    x = sample_df["X"].values
    y = sample_df["Y"].values
    z = sample_df[target_attr].values

    # 构建普通克里格模型
    ok_model = OrdinaryKriging(
        x, y, z,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False
    )

    # 插值网格坐标
    grid_x = grid_gdf["X"].values
    grid_y = grid_gdf["Y"].values

    # 执行插值
    z_pred, z_var = ok_model.execute(
        "points",
        grid_x,
        grid_y,
        backend="loop",
        n_closest_points=n_closest_points
    )

    # 写入结果
    grid_gdf[f"pred_{target_attr}"] = z_pred

    return grid_gdf, ok_model
