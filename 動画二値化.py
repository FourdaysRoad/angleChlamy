import cv2
import numpy as np

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