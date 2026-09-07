from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from pykrige.ok import OrdinaryKriging

app = Flask(__name__)

def load_data():
    for fname in ['ALL_VS30.xlsx', 'ALL_VS30.xlsm', '8孔.xlsx', '8孔.xlsm']:
        if os.path.exists(fname):
            try:
                # 讀取 Excel（無標頭檔，自動對應 A:孔號, B:經度, C:緯度, D:Mean, E:SD）
                df = pd.read_excel(fname, header=None)
                
                # --- 自動去除重複資料邏輯 ---
                # 1. 先依據 A 欄 (孔號) 去重，只保留第一筆
                df = df.drop_duplicates(subset=[0], keep='first')
                
                # 2. 再依據 B, C 欄 (經緯度) 去重，避免同座標造成克里金矩陣奇異化
                df = df.drop_duplicates(subset=[1, 2], keep='first')
                # ---------------------------

                print(f"✅ 成功讀取並自動去重：{fname}，有效鑽孔數：{len(df)}")
                return df
            except Exception as e:
                print(f"❌ 讀取 {fname} 失敗: {e}")
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/boreholes', methods=['GET'])
def get_boreholes():
    df = load_data()
    if df is None:
        return jsonify([])
    
    boreholes = []
    for idx, row in df.iterrows():
        try:
            lon_val = float(row[1])
            lat_val = float(row[2])
            mean_val = float(row[3])
            sd_val = float(row[4])
            hole_name = str(row[0])

            if not (np.isnan(lon_val) or np.isnan(lat_val)):
                boreholes.append({
                    'name': hole_name if hole_name else f"鑽孔 #{idx+1}",
                    'lon': lon_val,
                    'lat': lat_val,
                    'mean': mean_val,
                    'sd': sd_val
                })
        except (ValueError, TypeError):
            continue  # 自動跳過非數字或空值列
            
    print(f"🎉 成功傳送 {len(boreholes)} 個唯一鑽孔點位至前端！")
    return jsonify(boreholes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        target_lon = float(data['lon'])
        target_lat = float(data['lat'])
        
        df = load_data()
        if df is None:
            return jsonify({'error': '找不到數據檔案'}), 400

        # 清理資料：只保留合法的數字列
        valid_data = []
        for idx, row in df.iterrows():
            try:
                lon = float(row[1])
                lat = float(row[2])
                mean = float(row[3])
                sd = float(row[4])
                if not (np.isnan(lon) or np.isnan(lat) or np.isnan(mean) or np.isnan(sd)):
                    valid_data.append([lon, lat, mean, sd])
            except (ValueError, TypeError):
                continue

        clean_np = np.array(valid_data)
        x_known = clean_np[:, 0]  # B欄 經度
        y_known = clean_np[:, 1]  # C欄 緯度
        mean_known = clean_np[:, 2] # D欄 Mean
        sd_known = clean_np[:, 3]   # E欄 SD
        
        # 1. 【Vs30 Mean】對鑽孔 Mean 直接進行 Ordinary Kriging 空間推估
        OK_mean = OrdinaryKriging(
            x_known, y_known, mean_known,
            variogram_model='spherical',
            verbose=False, enable_plotting=False
        )
        pred_mean_arr, _ = OK_mean.execute('points', [target_lon], [target_lat])
        pred_mean = float(pred_mean_arr[0])

        # 2. 【Vs30 SD】對鑽孔 SD 直接進行 Ordinary Kriging 空間推估 + 幾何空間變異數
        OK_sd = OrdinaryKriging(
            x_known, y_known, sd_known,
            variogram_model='spherical',
            verbose=False, enable_plotting=False
        )
        interpolated_sd, krig_var = OK_sd.execute('points', [target_lon], [target_lat])
        
        base_sd = float(interpolated_sd[0])
        sigma_kriging_sq = max(float(krig_var[0]), 0.0)
        
        # 總標準差 = 區域土壤變異數內插 + 距離產生之空間估計誤差
        total_sd = float(np.sqrt(base_sd**2 + sigma_kriging_sq))
        
        print(f"📍 毫秒級推估成功：座標 ({target_lon}, {target_lat}) -> Vs30: {pred_mean:.2f} ± {total_sd:.2f} m/s")
        
        return jsonify({
            'vs30_mean': pred_mean,
            'vs30_sd': total_sd
        })
    except Exception as e:
        print(f"❌ 預測出錯細節: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🚀 伺服器啟動成功！請開啟瀏覽器造訪：http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)