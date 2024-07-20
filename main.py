import cv2
import numpy as np
import os
from winotify import Notification

# 画像の読み込み
image_path = './3.jpg'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

# 画像をRGBに変換(色の検出)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 画像をHSVに変換（色相、彩度、明度）
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 赤色の範囲を定義する改良した閾値
thresholds = [
    {"lower": [0, 50, 50], "upper": [10, 255, 255],
     "lower2": [170, 50, 50], "upper2": [180, 255, 255]},
    {"lower": [0, 60, 60], "upper": [10, 255, 255],
     "lower2": [170, 60, 60], "upper2": [180, 255, 255]},
    {"lower": [0, 70, 70], "upper": [10, 255, 255],
     "lower2": [170, 70, 70], "upper2": [180, 255, 255]}
]

# 保存ディレクトリの作成
output_dir = './test'
os.makedirs(output_dir, exist_ok=True)

# 面積の閾値（大きな赤い部分のみを残す）
min_area = 500  # この値は調整可能です

# 赤色と判断するピクセル数の閾値
red_pixel_threshold = 1500  # この値は調整可能です

# 検出された赤い領域の情報
red_lights = []  # 検出された赤い領域の座標と状態を格納

# 合計クリック回数
sum_click = 0

def on_mouse_click(event, x, y, flags, param):
    global red_lights, sum_click, image_with_detections
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, light in enumerate(red_lights):
            x1, y1, x2, y2, clicked = light
            if x1 <= x <= x2 and y1 <= y <= y2 and not clicked:
                cv2.rectangle(image_with_detections, (x1, y1), (x2, y2), (0, 0, 255), 2)
                red_lights[i] = (x1, y1, x2, y2, True)
                sum_click += 1
                break

# 検出と結果の保存
for i, thresh in enumerate(thresholds):
    # マスクの作成
    lower_red1 = np.array(thresh["lower"])
    upper_red1 = np.array(thresh["upper"])
    mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)

    lower_red2 = np.array(thresh["lower2"])
    upper_red2 = np.array(thresh["upper2"])
    mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)

    # マスクの結合
    mask = mask1 + mask2

    # 赤色ピクセルの数をカウント
    red_pixel_count = cv2.countNonZero(mask)

    # 赤色領域のピクセル数が閾値を超えた場合のみ続行
    if red_pixel_count > red_pixel_threshold:
        print(f"Threshold {i+1}: 赤色が検出されました (赤色ピクセル数: {red_pixel_count})")
        
        # マスクを適用して赤い部分を検出
        result = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

        # 輪郭を検出
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 検出された赤色の部分に印を付ける（大きな部分のみ）
        image_with_detections = image_rgb.copy()
        red_lights = []  # 検出された赤い領域の情報
        sum_click = 0  # 各閾値ごとにクリック数をリセット
        cv2.namedWindow(f"Detected Red Lights with Threshold {i + 1}")
        cv2.setMouseCallback(f"Detected Red Lights with Threshold {i + 1}", on_mouse_click)

        for contour in contours:
            if cv2.contourArea(contour) > min_area:  # 面積閾値を適用
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image_with_detections, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 初期は緑色
                red_lights.append((x, y, x + w, y + h, False))  # 初期状態ではクリックされていない

        # トースト通知
        toast = Notification(app_id="Red Light Detection",
                            title=f"Threshold {i + 1}",
                            msg=f"赤色検出数: {len(red_lights)}",
                            duration="short")
        toast.show()

        # クリックゲームの開始
        cv2.namedWindow(f"Detected Red Lights with Threshold {i + 1}")
        cv2.setMouseCallback(f"Detected Red Lights with Threshold {i + 1}", on_mouse_click)

        while True:
            display_image = image_with_detections.copy()  # 表示用の画像をコピー
            # クリック情報を画像に表示
            cv2.putText(display_image, f"Clicked: {sum_click}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(f"Detected Red Lights with Threshold {i + 1}", display_image)
            key = cv2.waitKey(1)
            if key == 27:  # Escキーで終了
                break

        cv2.destroyAllWindows()

        # クリックされた数を表示
        clicked_count = sum(light[-1] for light in red_lights)
        print(f"Threshold {i + 1}: クリックされた数: {clicked_count}")
        sum_click += clicked_count
    else:
        print(f"Threshold {i+1}: 赤色は検出されませんでした (赤色ピクセル数: {red_pixel_count})")

# 最後に合計クリック回数を画像に表示
final_image = image_rgb.copy()
cv2.putText(final_image, f"Total Clicks: {sum_click}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("Total Clicks", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
