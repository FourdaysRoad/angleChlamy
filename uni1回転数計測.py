import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df = pd.read_csv('C:/Users/81805/Desktop/Python勉強/研究関係/uni1回転数計測_蛍光サンプル_imagejdate.csv')

# 「Angle」と「Slice」列のみを取得する
df_subset = df[['Angle', 'Slice']]

ans_Angle_list = []
Angle_list = []
n = 0
for i in range (len(Angle_list)-1):
    if abs (Angle_list[i] - Angle_list[i + 1])>150:
        n += 1
    ans_Angle_list.append(Angle_list[i] + (180 * n))

# 折れ線グラフを描画する
plt.plot(df_subset['Slice'], df_subset['Angle'])
plt.xlabel('Slice')
plt.ylabel('Angle')
plt.title('Angle vs Slice')
plt.show()
