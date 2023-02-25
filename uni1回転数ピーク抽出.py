import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

SAMPLING = 600

# CSVファイルを読み込む
df = pd.read_csv('C:/Users/81805/Desktop/Python勉強/研究関係/uni1回転数計測_蛍光サンプル_imagejdate.csv')

# 「Angle」と「Slice」列のみを取得する
df_subset = df[['Angle', 'Slice']]

# Angleの値を-180~180の範囲に収める
ans_Angle_list = []
Angle_list = df_subset["Angle"].values
time = df_subset["Slice"].values / SAMPLING
n = 0
frame = 0
for i in range(len(Angle_list)-1):
    ans_Angle_list.append(Angle_list[i] + (180 * n))
    if abs(Angle_list[i] - Angle_list[i + 1]) > 100 and frame > 35:
        n += 1
        frame = 0
    frame += 1
ans_Angle_list.append(ans_Angle_list[-1] + (180 * n))

for i in range(10, len(ans_Angle_list)-1):
    if abs(ans_Angle_list[i] - ans_Angle_list[i+1]) > 150:
        ans_Angle_list[i+1] -= 180

# 折れ線グラフを作成する
plt.plot(time[:-1], ans_Angle_list[:-1])

# ピークを抽出する
ans_Angle_list_np = np.array(ans_Angle_list)
peaks_dict = find_peaks(ans_Angle_list_np)
peaks = peaks_dict[0]

# ピークの位置に赤いxをプロットする
plt.plot(time[peaks], ans_Angle_list_np[peaks], ".", color='red')

# グラフを表示する
plt.show()

# 3番目以降のピークの間隔を計算する
peak_distances = np.diff(time[peaks])[2:]

# ピーク間の平均距離と標準偏差を出す
mean_distance = np.mean(peak_distances)
std_distance = np.std(peak_distances)

# 3番目以降のピーク間隔の平均と標準偏差を表示する
print(f"3番目以降のピーク間隔は{mean_distance}±{std_distance}")

frequency = 1/ mean_distance
print(f"鞭毛打頻度 (Hz): {frequency}")
