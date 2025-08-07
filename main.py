import cv2
import mediapipe as mp
import csv
import math

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
myDraw = mp.solutions.drawing_utils

# 節點樣式
handLmsStyle = myDraw.DrawingSpec(color=(0, 0, 255), thickness=10)
handConStyle = myDraw.DrawingSpec(color=(0, 255, 0), thickness=10)

# 像素到毫米的轉換比例（需要實測或估算，例如：每像素0.264毫米）
pixel_to_mm_ratio = 0.264  # 這是範例值，請根據實際需求調整

def calculate_degree(p1, origin, p2):
    # 向量 v1 和 v2
    v1 = (p1[0] - origin[0], p1[1] - origin[1])
    v2 = (p2[0] - origin[0], p2[1] - origin[1])

    # 點積和向量長度
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = (v1[0]**2 + v1[1]**2) ** 0.5
    magnitude_v2 = (v2[0]**2 + v2[1]**2) ** 0.5
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    
    angle_rad = math.acos(dot_product / (magnitude_v1 * magnitude_v2))
    return round(math.degrees(angle_rad), 2)

# 輸出 CSV
output_file = 'hand_selected_landmarks.csv'
with open(output_file, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    header = [f'landmark_{i}_{axis}' for i in [0, 1, 9, 17] for axis in ["x", "y", "z"]]
    csv_writer.writerow(header)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        imgHeight, imgWidth, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(imgRGB)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                # 提取手腕、大拇指（節點1）、中指（節點9）、小拇指（節點17）
                important_points = [0, 4, 12, 20]
                landmarks = []
                points = []

                wrist = handLms.landmark[important_points[0]]
                wrist_x, wrist_y = int(wrist.x * imgWidth), int(wrist.y * imgHeight)

                thumb = handLms.landmark[important_points[1]]
                thumb_x, thumb_y = int(thumb.x * imgWidth), int(thumb.y * imgHeight)

                middle = handLms.landmark[important_points[2]]
                middle_x, middle_y = int(middle.x * imgWidth), int(middle.y * imgHeight)

                pinky = handLms.landmark[important_points[3]]
                pinky_x, pinky_y = int(pinky.x * imgWidth), int(pinky.y * imgHeight)

                # 計算長度並轉換為毫米，保留小數點後兩位
                thumb_length = round(((thumb_x - wrist_x) ** 2 + (thumb_y - wrist_y) ** 2) ** 0.5 * pixel_to_mm_ratio, 2)
                middle_length = round(((middle_x - wrist_x) ** 2 + (middle_y - wrist_y) ** 2) ** 0.5 * pixel_to_mm_ratio, 2)
                pinky_length = round(((pinky_x - wrist_x) ** 2 + (pinky_y - wrist_y) ** 2) ** 0.5 * pixel_to_mm_ratio, 2)

                cv2.putText(img, f'Thumb length: {thumb_length} mm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, f'Middle length: {middle_length} mm', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, f'Pinky length: {pinky_length} mm', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                angle = calculate_degree((thumb_x, thumb_y), (wrist_x, wrist_y), (pinky_x, pinky_y))
                cv2.putText(img, f'Angle: {angle} deg', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                for i in important_points:
                    lm = handLms.landmark[i]
                    cx, cy = int(lm.x * imgWidth), int(lm.y * imgHeight)
                    points.append((cx, cy))
                    landmarks.extend([lm.x, lm.y, lm.z])

                    # 繪製節點
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

                # 連接手腕到其他手指
                for i in range(1, len(points)):
                    cv2.line(img, points[0], points[i], (0, 255, 0), 5)

                # 寫入 CSV
                csv_writer.writerow(landmarks)

        cv2.imshow('Hand Tracking', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
hands.close()