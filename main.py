import cv2
import numpy as np
import os
import tkinter as tk
import threading

# 画像の読み込み
image_path = './3.jpg'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

# BGR形式のまま保持（RGB変換を行わない）
image_bgr = image

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
total_clicks = 0

# グローバル変数
exit_flag = False
current_threshold_index = 0

def countdown(timer, next_thread):
    if timer <= 0:
        global exit_flag
        exit_flag = True
        next_thread.start()
    else:
        label.config(text=f"残り時間: {timer // 60}分 {timer % 60}秒")
        label.after(1000, countdown, timer - 1, next_thread)

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

def detect_red_lights(thresh):
    global exit_flag, total_clicks, red_lights, image_with_detections, sum_click
    
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
        print(f"赤色が検出されました (赤色ピクセル数: {red_pixel_count})")
        
        # マスクを適用して赤い部分を検出
        result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

        # 輪郭を検出
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 検出された赤色の部分に印を付ける（大きな部分のみ）
        image_with_detections = image_bgr.copy()
        red_lights = []  # 検出された赤い領域の情報
        sum_click = 0  # 各閾値ごとにクリック数をリセット
        cv2.namedWindow("Detected Red Lights")
        cv2.setMouseCallback("Detected Red Lights", on_mouse_click)

        for contour in contours:
            if cv2.contourArea(contour) > min_area:  # 面積閾値を適用
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image_with_detections, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 初期は緑色
                red_lights.append((x, y, x + w, y + h, False))  # 初期状態ではクリックされていない

        # クリックゲームの開始
        while not exit_flag:
            display_image = image_with_detections.copy()  # 表示用の画像をコピー
            # クリック情報を画像に表示
            cv2.putText(display_image, f"Clicked: {sum_click}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Detected Red Lights", display_image)
            key = cv2.waitKey(1)
            if key == 27:  # Escキーで終了
                exit_flag = True
                break

        cv2.destroyAllWindows()

        # クリックされた数を表示
        clicked_count = sum(light[-1] for light in red_lights)
        print(f"クリックされた数: {clicked_count}")
        total_clicks += sum_click
    else:
        print(f"赤色は検出されませんでした (赤色ピクセル数: {red_pixel_count})")

# Tkinterウィンドウの設定
root = tk.Tk()
root.title("カウントダウンタイマー")

label = tk.Label(root, font=('Helvetica', 48), text="")
label.pack()

# スレッドの作成
threads = []
for i, thresh in enumerate(thresholds):
    threads.append(threading.Thread(target=detect_red_lights, args=(thresh,)))

# カウントダウンと検出スレッドの開始
def start_threads(index):
    global exit_flag
    if index < len(threads):
        exit_flag = False
        threads[index].start()
        countdown(10, threading.Thread(target=start_threads, args=(index + 1,)))

start_threads(0)

root.mainloop()

# スレッドの終了待ち
for thread in threads:
    thread.join()

# 最後に合計クリック回数を画像に表示
final_image = image_bgr.copy()
cv2.putText(final_image, f"Total Clicks: {total_clicks}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("Total Clicks", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()