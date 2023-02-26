import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 動画のframe数を決める
SAMPLING = 600

# 動画を開く
cap = cv2.VideoCapture('C:/Users/81805/Desktop/20230223_uni1回転数計測/uni1回転数計測_蛍光サンプル.avi')

angle_list = []
# フレームを取得する
while cap.isOpened():
    ret, frame = cap.read()

    # フレームが正常に取得できない場合はループを抜ける
    if not ret:
        break

    # 画像をグレースケールに変換する
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 画像を二値化する
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 二値化された画像をCV_8UC1形式に変換する
    binary = cv2.convertScaleAbs(binary)

    # 輪郭を抽出する
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 検出された輪郭がある場合
    if contours:
        # 最大面積を持つ輪郭を選択する
        contour = max(contours, key=cv2.contourArea)

        # 楕円をフィッティングする
        ellipse = cv2.fitEllipse(contour)

        # 楕円を描画する
        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

        # 楕円の角度を取得する
        angle = ellipse[2]
        angle_list.append(angle)

        # 角度を表示する
        cv2.putText(frame, f'Angle: {angle:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 画像を表示する
    cv2.imshow('frame', frame)

    # qキーが押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# メモリを解放する
cap.release()
cv2.destroyAllWindows()

print(angle_list)


# グラフを右上がり/右下がりにするための処理
ans_Angle_list = []
ans_ans_Angle_list = []
n = 0
m = 0
frame = 0
for i in range(len(angle_list)-1):
    ans_Angle_list.append(angle_list[i] - (180 * n))
    if abs(angle_list[i] - angle_list[i + 1]) > 100 and frame > 35:
        n += 1
        frame = 0
    frame += 1
ans_Angle_list.append(ans_Angle_list[-1] + (180 * n))

for i in range(10, len(ans_Angle_list)-1):
    if abs(ans_Angle_list[i] - ans_Angle_list[i+1]) > 150:
        ans_Angle_list[i+1] -= 180

# for i in range(len(ans_Angle_list)-1):
#     ans_ans_Angle_list.append(ans_Angle_list[i] + (180 * m))
#     if abs(ans_Angle_list[i] - ans_Angle_list[i + 1]) > 100 and frame > 35:
#         m += 1
#         frame = 0
#     frame += 1
# ans_ans_Angle_list.append(ans_ans_Angle_list[-1] + (180 * m))

# 時間を作成する
time = np.arange(0, len(angle_list)/SAMPLING, 1/SAMPLING)

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


# ピークを抽出する
ans_Angle_list_np = np.array(ans_Angle_list)
peaks_dict = find_peaks(ans_Angle_list_np)
peaks = peaks_dict[0]

# 3番目のピークの位置を取得する
third_peak_idx = 2  # 0-indexed
third_peak_pos = peaks[third_peak_idx]

# 3番目のピークから、3600度変化がある位置を探す
angle_diff_limit = 3600
angle_diff = 0
for i in range(third_peak_pos, len(angle_list)-1):
    angle_diff += angle_list[i+1] - angle_list[i]
    if angle_diff >= angle_diff_limit:
        # 3600度変化がある位置を発見
        break

# 3番目のピークから10回転するのにかかる時間を測り、1回転あたりにかかる時間を算出
time_diff = (i - third_peak_pos) / SAMPLING
print(f"1回転するのにかかる時間: {time_diff/10:.2f}秒")