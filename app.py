from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from pykrige.ok import OrdinaryKriging

app = Flask(__name__)


# =========================================================
# 讀取工程鑽孔資料
# A = 孔號
# B = 經度
# C = 緯度
# D = MCS Mean
# E = MCS SD
# =========================================================
def load_data():
    for fname in ['ALL_VS30.xlsx', 'ALL_VS30.xlsm', '8孔.xlsx', '8孔.xlsm']:
        if os.path.exists(fname):
            try:
                df = pd.read_excel(fname, header=None)

                # 孔號去重
                df = df.drop_duplicates(subset=[0], keep='first')

                # 座標去重
                df = df.drop_duplicates(subset=[1, 2], keep='first')

                print(f"成功讀取工程鑽孔：{fname}，有效資料數：{len(df)}")
                return df

            except Exception as e:
                print(f"讀取 {fname} 失敗: {e}")

    return None


# =========================================================
# 讀取 TSMIP 實測 Vs30
# A = 站名
# B = 經度
# C = 緯度
# D = 實測 Vs30
# =========================================================
def load_tsmip():

    fname = 'TSMIP_VS30.xlsx'

    if not os.path.exists(fname):
        print("找不到 TSMIP_VS30.xlsx")
        return None

    try:
        df = pd.read_excel(fname, header=None)

        valid_data = []

        for idx, row in df.iterrows():
            try:
                name = str(row[0])
                lon = float(row[1])
                lat = float(row[2])
                vs30 = float(row[3])

                if not (
                    np.isnan(lon) or
                    np.isnan(lat) or
                    np.isnan(vs30)
                ):
                    valid_data.append([
                        name,
                        lon,
                        lat,
                        vs30
                    ])

            except (ValueError, TypeError):
                continue

        tsmip_df = pd.DataFrame(
            valid_data,
            columns=['name', 'lon', 'lat', 'vs30']
        )

        # 避免完全相同座標造成 Kriging 問題
        tsmip_df = tsmip_df.drop_duplicates(
            subset=['lon', 'lat'],
            keep='first'
        )

        print(f"成功讀取 TSMIP：{len(tsmip_df)} 站")

        return tsmip_df

    except Exception as e:
