import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df = pd.read_csv('C:/Users/81805/Desktop/Python勉強/研究関係/uni1回転数計測_蛍光サンプル_imagejdate.csv')

# 「Angle」と「Slice」列のみを取得する
df_subset = df[['Angle', 'Slice']]

ans_Angle_list = []
Angle_list = df_subset["Angle"].values
n = 0
frame = 0
for i in range (len(Angle_list)-1):
    ans_Angle_list.append(Angle_list[i] + (180 * n))
    if abs (Angle_list[i] - Angle_list[i + 1])>100 and frame > 35:
        n += 1
        frame = 0
    
    frame += 1
ans_Angle_list.append(ans_Angle_list[-1] + (180 * n))


for i in range(10, len(ans_Angle_list)-1):
    if abs(ans_Angle_list[i] - ans_Angle_list[i+1]) > 150:
        ans_Angle_list[i+1] -= 180

plt.plot(df_subset['Slice'][:-1], ans_Angle_list[:-1])
plt.show()