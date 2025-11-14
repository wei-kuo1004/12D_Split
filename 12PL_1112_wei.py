#20250818補上疊合順序功能BY奕翔

import av
import os
import cv2
import time
import threading
from ultralytics import YOLO
import requests
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import deque

# Force detection to use CPU
#device = "cpu"
#print(f"Using device: {device}")

import torch
print("torch =", torch.__version__, "cuda =", torch.version.cuda, "available =", torch.cuda.is_available())
import torchvision
print("torchvision =", torchvision.__version__)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================
#   LINEGPT：Token 取得 + 訊息/影像傳送（企業級）
# ============================================
# ===== 全域變數 =====
_token_cache = {"token": None, "expire_time": 0}
_token_lock = threading.Lock()   # 防止多 Thread 同時刷新 Token

# ===== 基本設定 =====
LINE_LOGIN_URL = "https://lineapi.pcbut.com.tw:888/api/account/login"
LINE_NOTIFY_URL = "https://lineapi.pcbut.com.tw:888/api/notify-with-img"
LINE_USERNAME = "utbot"
LINE_PASSWORD = "mi2@admin5566"
DEFAULT_CHAT_ID = "2F0177B1-2AB0-471B-9001-E40B134F4D0F"   # 測試用

# Alert settings
alert_threshold = 0.9  # Confidence threshold for alerts
alert_cooldown = 300  # Cooldown time in seconds
alert_records = []  # Store detection events for batch alerts
#last_alert_times = {}  # Store the last alert time for each camera
alert_interval = 300  # Batch alert interval (5 minutes)


# 類別對應的冷卻時間
class_alert_cooldowns = {
        0: 300,  # 火光
        1: 300,  # 煙霧
        2: 300,  # 人員倒臥
        3: 300,  # 未戴手套
        4: 300,   # 未戴護目鏡
        5: 300,   #紅色板子
        6: 300,   #版面不一致
        7: 300,   #白色板子
        8: 300,   #疊合物料數量不正確
        9: 300,
        10: 300,
        11: 300   #推車
    }

# Mutex lock to ensure thread safety
mutex = threading.Lock()

# Load YOLOv8 model (ensure this model is trained with the 7 specified classes)
# model_fire_smoke = YOLO('model/fire_smoke/best_0312.pt').to(0)  # Replace 'best.pt' with your model file
# model_fall = YOLO('model/fall/best_0318.pt').to(0)
# model_no_gloves = YOLO('model/gloves_goggles/best_0328.pt').to(0)
# #model_cellphone = YOLO('D:/AI/Demo_eden/edenTest/model_cellphone_label_studio_reduce_v2/model_collections/weights/best.pt').to(0)
# model_red_board = YOLO('model/red_board/best_0307.pt').to(0)


# Mapping from class index to event name (English)
class_event_mapping_en = {
    0: "fire",
    1: "smoke",
    2: "fall",
    3: "no gloves",
    4: "without goggles",
    5: "red board",
    6: "not_aligned",
    7: "white paper",
    8: "number_isnot_correct",
    9: "blue_gloves",
    10: "black_board",
    11: "cart"

}

# Mapping from class index to event name (Chinese)
class_event_mapping_cn = {
    0: "火光",
    1: "煙霧",
    2: "人員倒臥",
    3: "未戴手套",
    4: "未戴護目鏡",
    5: "紅色板料",
    6: "版面不一致",
    7: "白色板子",
    8: "疊合物料數量不正確",
    9: "藍色手套",
    10: "黑色板子",
    11: "推車"

}


color_dict ={
    0: (255,0,0), #紅
    1: (0,255,0), #綠
    2: (0,0,255), #藍
    3: (255,255,0), #淡藍
    4: (255,0,255),
    5: (0,255,255),
    6: (255,255,255),
    7: (255,125,0),
    8: (125,125,125),
    9: (125,75,125),
    10: (75,125,75),
    11: (200,100,50) # 推車 - 棕色
}

last_alert_times = {}

camera_urls = {
                "501001":"rtsp://hikvision:Unitech0815!@10.80.21.130","501002":"rtsp://hikvision:Unitech0815!@10.80.21.131",
                "501005":"rtsp://hikvision:Unitech0815!@10.80.21.134","501006":"rtsp://hikvision:Unitech0815!@10.80.21.135",
                "501009":"rtsp://hikvision:Unitech0815!@10.80.21.138","501010":"rtsp://hikvision:Unitech0815!@10.80.21.139",
                "501013":"rtsp://hikvision:Unitech0815!@10.80.21.142","501014":"rtsp://hikvision:Unitech0815!@10.80.21.143"
                }

# camera_urls = {
#                 "501006":"C:/Users/vincent-shiu/Web/RecordFiles/2025-05-05/0505_v1_part3.mp4"
#                 }

model_specific_rois = {
    #手套辨識範圍，只有501006這隻會進行預疊合作業
    "501006": {
        "model_no_gloves": (50, 30, 1000, 680)
        },  
}

# 7新增推車檢測的排除區域設定
cart_exclusion_rois = {
    "501005": (50, 30, 1000, 680)  # 501005攝影機需要排除的區域
}

for i in camera_urls:
    if i not in last_alert_times:
        last_alert_times[i] = {}


# camera_rois = {"501006":(650,50,2050,1650)}
camera_rois = {
    "501006":{
                "white_paper_rois": [
                    (0, 90, 750, 1450),
                    (1950, 0, 3200, 1450),
                    (650, 50, 2050, 1650)

                ],
                "red_board_rois": [
                    (650, 50, 2050, 1650),
                    (1950, 0, 3200, 1450)
                ],
                "blue_gloves_rois":[
                    (650,25,2050,400),
                    (650,1000,2050,1600)
                ]
            } 
    }


####### 中間白色紙張計時器
# 定義 ROI 區域
ROI_COORDINATES = [(650, 50, 2050, 1650)]  # (x1, y1, x2, y2)
MIN_DETECTION_SECONDS = 3  # 最少連續偵測時間（秒）

# 初始化計時器
roi_detection_times = {roi: 0 for roi in ROI_COORDINATES}
last_detection_times = {roi: 0 for roi in ROI_COORDINATES}

double_Flag = False


# 每支攝影機的最新幀記憶區
frame_deques = {cam: deque(maxlen=1) for cam in camera_urls.keys()}

# 模型初始化（GPU ID 視需要調整）
model_map = {
    "model_no_gloves": YOLO('model/gloves_goggles/best.pt').to(0),
    "model_fire_smoke": YOLO('model/fire_smoke/best.pt').to(0),
    "model_fall": YOLO('model/fall/best.pt').to(0),
    "model_red_board": YOLO('model/red_board/best.pt').to(0),
    "model_white_paper": YOLO('model/white_paper/best.pt').to(0),
    "model_stick_hand": YOLO('model/stick_hand/best.pt').to(0),
    "model_cart": YOLO('model/cart/best.pt').to(0)  # 新增推車模型
}

# 每個攝影機分別推論哪些模型
camera_models = { #"model_no_gloves", "model_fire_smoke","model_fall"
    "501001": [ "model_fire_smoke","model_fall"],
    "501005": [ "model_fire_smoke","model_fall","model_cart"],
    "501009": [ "model_fire_smoke","model_fall"],
    "501013": [ "model_fire_smoke","model_fall"],
    "501002": [ "model_red_board"],
    "501006": [ "model_no_gloves","model_red_board","model_white_paper"],
    "501010": [ "model_red_board"],
    "501014": [ "model_red_board"]
}


def receive_frames(cam_id, url):
    print(f"Starting receiver for {cam_id}")
    input_container = av.open(url, options={"hwaccel": "cuda"})

    for frame in input_container.decode(video=0):
        if stop_event.is_set():
            break
        img = frame.to_ndarray(format="bgr24")
        frame_deques[cam_id].append(img)
    input_container.close()
    print(f"Receiver stopped for {cam_id}")

def run_model(model, frame, detections,model_name,camera_index=None):

    if model_name == "model_red_board":
        results = model.predict(frame, conf=0.9,imgsz=640,verbose=False)
    else:
        # 使用模型進行推論
        results = model.predict(frame, conf=0.9,imgsz=640,verbose=False)

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            cls = int(box.cls[0])  # Get class index
            
            x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])

            if model_name == "model_no_gloves":
                if confidence < alert_threshold:
                    continue
                cam_key = str(camera_index) if camera_index is not None else None
                x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])

                roi_gloves = None
                if cam_key and cam_key in model_specific_rois:
                    roi_gloves = model_specific_rois[cam_key].get("model_no_gloves")

                if roi_gloves is not None:
                    rx1, ry1, rx2, ry2 = roi_gloves
                    if (x1_box >= rx1) and (y1_box >= ry1) and (x2_box <= rx2) and (y2_box <= ry2):
                        adjusted_box = {
                            "model": model_name,  # 添加模型名稱
                            'xyxy': [x1_box, y1_box, x2_box, y2_box],
                            'conf': box.conf,
                            'cls': box.cls
                            }

                        detections.append(adjusted_box)
            
             # 在現有的 if model_name == "model_no_gloves": 判斷後新增：
            elif model_name == "model_cart":
                if confidence < alert_threshold:
                    continue
                
                cam_key = str(camera_index) if camera_index is not None else None
                x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])
                
                # 推車排除區域檢查
                if cam_key == "501005":
                    exclude_roi = (50, 30, 1000, 680)  # 排除區域座標
                    ex1, ey1, ex2, ey2 = exclude_roi
                    
                    # 計算檢測框的中心點
                    center_x = (x1_box + x2_box) // 2
                    center_y = (y1_box + y2_box) // 2
                    
                    # 如果中心點在排除區域內，跳過此檢測
                    if ex1 <= center_x <= ex2 and ey1 <= center_y <= ey2:
                        continue
                
                # 如果不在排除區域內，加入檢測結果
                adjusted_box = {
                    "model": model_name,
                    'xyxy': [x1_box, y1_box, x2_box, y2_box],
                    'conf': box.conf,
                    'cls': box.cls
                }
                detections.append(adjusted_box)
            else:
                if confidence > alert_threshold:
                    # Get bounding box coordinates
                    x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])

                    adjusted_box = {
                            "model": model_name,  # 添加模型名稱
                            'xyxy': [x1_box, y1_box, x2_box, y2_box],
                            'conf': box.conf,
                            'cls': box.cls
                            }

                    detections.append(adjusted_box)

def run_model_with_cart_exclusion(model, frame, detections, model_name, camera_index=None):
    """
    修改版的 run_model 函數，專門處理推車模型的區域排除邏輯
    """
    if model_name == "model_cart":
        results = model.predict(frame, conf=0.9, imgsz=640, verbose=False)
    else:
        results = model.predict(frame, conf=0.9, imgsz=640, verbose=False)

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            
            x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])

            # 推車模型的特殊處理 - 檢查是否在排除區域外
            if model_name == "model_cart":
                if confidence < alert_threshold:
                    continue
                
                cam_key = str(camera_index) if camera_index is not None else None
                
                # 檢查是否有排除區域設定
                if cam_key and cam_key in cart_exclusion_rois:
                    exclude_roi = cart_exclusion_rois[cam_key]
                    ex1, ey1, ex2, ey2 = exclude_roi
                    
                    # 計算檢測框的中心點
                    center_x = (x1_box + x2_box) // 2
                    center_y = (y1_box + y2_box) // 2
                    
                    # 如果檢測框的中心點在排除區域內，則跳過此檢測
                    if ex1 <= center_x <= ex2 and ey1 <= center_y <= ey2:
                        continue
                
                # 如果不在排除區域內，則加入檢測結果
                adjusted_box = {
                    "model": model_name,
                    'xyxy': [x1_box, y1_box, x2_box, y2_box],
                    'conf': box.conf,
                    'cls': box.cls
                }
                detections.append(adjusted_box)
            
            # 其他模型的原有邏輯保持不變
            else:
                if confidence > alert_threshold:
                    adjusted_box = {
                        "model": model_name,
                        'xyxy': [x1_box, y1_box, x2_box, y2_box],
                        'conf': box.conf,
                        'cls': box.cls
                    }
                    detections.append(adjusted_box)

def run_model_red_board(model, frame, detections,model_name,camera_index=None):
    #detections = []

    # if model_name == "model_stick_hand":
    #     results = model(frame, conf=0.3 ,verbose=False)
    # results = model(frame, conf=0.8 ,verbose=False)

    #results = model.predict(frame, conf=0.9,imgsz=640,verbose=False)
    red_board_detected = False
    if model_name == "model_red_board":
        results = model.predict(frame, conf=0.9,imgsz=640,verbose=False)
    
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            cls = int(box.cls[0])  # Get class index
            
            x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])
            
            if confidence > alert_threshold:
                red_board_detected = True

    return red_board_detected

def inference_worker(camera_index):
    models = camera_models.get(camera_index, [])

    red_board_count = 0 # 看被判定not aligned的圖，上線後可省略此變數
    red_board_frame = [] # 連續偵測到red board的buffer
    red_board_flag_first = True # 紀錄每一批次的標準答案
    red_board_cooldown = 0 #  紀錄中段作業區沒有新偵測的紅色板子物件的時間
    red_board_ori_frame = []
    roi_frame_location_buffer = []

    standard_frame = None
    number_count_process = None
    red_board_start_time = None

    code_start_time = time.time()  # 重新計時

    # 初始化手套位置與時間緩存
    glove_positions = []  # 手套座標清單
    glove_stable_time = []  # 手套穩定時間
    stability_threshold = 5 # 穩定時間閾值（秒）
    position_tolerance = 100  # 位置誤差（像素）
    distance_threshold = 400  # 手套之間距離閾值（像素）
    detection_success = False
    double_PP_start = False
    double_PP_end = time.time()
    red_board_right_time = 0
    double_PP_process_time= 0
    

    while not stop_event.is_set():
        if frame_deques[camera_index]:
            raw_frame = frame_deques[camera_index][-1]

            # ============================================================
            # 🔥 Frame 安全檢查 + 防 Crash + 清楚 LOG（這一段是新加的）
            # ============================================================
            if raw_frame is None:
                print(f"❌ [Camera {camera_index}] 收到無效畫面（raw_frame=None），跳過此次推論")
                continue

            if not hasattr(raw_frame, "shape"):
                print(f"❌ [Camera {camera_index}] raw_frame 無 shape，可能解碼失敗，跳過此次推論")
                continue

            fh, fw = raw_frame.shape[:2]
            if fh == 0 or fw == 0:
                print(f"❌ [Camera {camera_index}] raw_frame 尺寸異常：{fw}x{fh}，跳過此次推論")
                continue

            # 在這裡先記錄好 frame 寬高，後面所有 ROI 驗證都可以安全使用
            frame_height = fh
            frame_width = fw
            # ============================================================
            # 🔥 新增區塊結束，下面接你原本的程式
            # ============================================================



            frame = raw_frame.copy()
            preview_frame = raw_frame.copy()
            detections = []

            red_board_flag = False #紀錄每次的frame有沒有偵測到red board，預設為False
            white_paper_flag = False
            if frame is not None:
                if camera_index == "501005":
                    # 繪製推車排除區域框線
                    exclude_roi = (50, 30, 1000, 680)
                    ex1, ey1, ex2, ey2 = exclude_roi
                    
                    # 驗證座標有效性
                    ex1 = max(0, min(ex1, frame_width - 1))
                    ex2 = max(0, min(ex2, frame_width))
                    ey1 = max(0, min(ey1, frame_height - 1)) 
                    ey2 = max(0, min(ey2, frame_height))
                    
                    if ex1 < ex2 and ey1 < ey2:
                        # 繪製橘色框線
                        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 165, 255), 5)
                        # 加上標籤
                        cv2.putText(frame, "CART EXCLUSION ZONE", (ex1, ey1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

                # 打印影像尺寸以確認 ROI 是否有效
                frame_height, frame_width = frame.shape[:2]
                # print(f"Camera {camera_index} - Frame size: {frame_width}x{frame_height}")
                
                # Retrieve ROI for this camera
                if str(camera_index) in camera_rois:
                    if "red_board_rois" in camera_rois[str(camera_index)]:

                        roi_frame = np.full_like(frame, (0, 0, 0))  #綠色背景
                        roi_red_board_frame_right = np.full_like(frame, (0, 0, 0))  #綠色背景

                        roi_sets = camera_rois[str(camera_index)]["red_board_rois"]
                        for roi in roi_sets:
                            x1, y1, x2, y2 = roi
                            # Validate ROI coordinates against frame size
                            x1 = max(0, min(x1, frame_width - 1))
                            x2 = max(0, min(x2, frame_width))
                            y1 = max(0, min(y1, frame_height - 1))
                            y2 = max(0, min(y2, frame_height))
                        

                            if x1 >= x2 or y1 >= y2:
                                print(f"Invalid white_paper_rois ROI for camera {camera_index}: {roi}. Skipping ROI processing.")
                                stop_event.set()
                                break
                                # roi = None
                                # roi_frame = frame
                            else:
                                # Draw ROI rectangle on the frame
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                                if x1 == 650:
                                    # 將對應 ROI 區塊複製到 roi_frame（其餘仍為黑）
                                    roi_frame[y1:y2, x1:x2] = preview_frame[y1:y2, x1:x2]
                                elif x1 == 1950:
                                    roi_red_board_frame_right[y1:y2, x1:x2] = preview_frame[y1:y2, x1:x2]


                    else:
                        print(f"{str(camera_index)} 沒有 red_board_rois 的 ROI...")
                        stop_event.set()
                        break

                    if "white_paper_rois" in camera_rois[str(camera_index)]:
                        # 建立全黑畫面
                        #roi_white_paper_frame = np.zeros_like(frame)
                        roi_white_paper_frame = np.full_like(frame, (0, 255, 0))  #綠色背景

                        roi_sets = camera_rois[str(camera_index)]["white_paper_rois"]
                        for roi in roi_sets:
                            x1, y1, x2, y2 = roi
                            # Validate ROI coordinates against frame size
                            x1 = max(0, min(x1, frame_width - 1))
                            x2 = max(0, min(x2, frame_width))
                            y1 = max(0, min(y1, frame_height - 1))
                            y2 = max(0, min(y2, frame_height))
                        

                            if x1 >= x2 or y1 >= y2:
                                print(f"Invalid white_paper_rois ROI for camera {camera_index}: {roi}. Skipping ROI processing.")
                                stop_event.set()
                                break
                                # roi = None
                                # roi_frame = frame
                            else:
                                # Draw ROI rectangle on the frame
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
                                
                                # 將對應 ROI 區塊複製到 roi_frame（其餘仍為黑）
                                roi_white_paper_frame[y1:y2, x1:x2] = preview_frame[y1:y2, x1:x2]

                                # resized_roi_white_paper_frame = cv2.resize(roi_white_paper_frame, (640, 400))

                                # cv2.imshow(f"review_{camera_index}",resized_roi_white_paper_frame)

                                # if cv2.waitKey(1) & 0xFF == ord('q'):
                                #     stop_event.set()



                    else:
                        print(f"{str(camera_index)} 沒有 white_paper_rois 的 ROI...")
                        stop_event.set()
                        break
                    
                    if "blue_gloves_rois" in camera_rois[str(camera_index)]:
                        roi_blue_gloves_frame = np.full_like(frame, (0, 0, 0))  #綠色背景
                        roi_sets = camera_rois[str(camera_index)]["blue_gloves_rois"]
                        for roi in roi_sets:
                            x1, y1, x2, y2 = roi
                            # Validate ROI coordinates against frame size
                            x1 = max(0, min(x1, frame_width - 1))
                            x2 = max(0, min(x2, frame_width))
                            y1 = max(0, min(y1, frame_height - 1))
                            y2 = max(0, min(y2, frame_height))
                        

                            if x1 >= x2 or y1 >= y2:
                                print(f"Invalid blue_gloves_rois ROI for camera {camera_index}: {roi}. Skipping ROI processing.")
                                stop_event.set()
                                break
                                # roi = None
                                # roi_frame = frame
                            else:
                                # Draw ROI rectangle on the frame
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
                                
                                # 將對應 ROI 區塊複製到 roi_frame（其餘仍為黑）
                                roi_blue_gloves_frame[y1:y2, x1:x2] = preview_frame[y1:y2, x1:x2]


                    else:
                        print(f"{str(camera_index)} 沒有 blue_gloves 的 ROI...")
                        stop_event.set()
                        break

                else:
                    roi = None
                    roi_frame = frame  # If no ROI defined, use the whole frame
                detections = []  # Store detections for alerts
                detections_v2 = []
                detected_gloves = [] #藍色手套
                diff_Flag = False
                
                number_right_flag = False
                number_left_flag = False
                for m_name in models:

                    model = model_map[m_name]
                    
                    if m_name == "model_red_board":
                        if camera_index == "501006":
                            run_model(model, roi_frame, detections,m_name)
                            red_board_right_status = run_model_red_board(model, roi_red_board_frame_right, detections,m_name)
                            #detections.extend(run_model(model, roi_frame, detections,m_name))
                    elif m_name == "model_white_paper":
                        if camera_index == "501006":
                            run_model(model, roi_white_paper_frame, detections,m_name,camera_index)
                    elif m_name == "model_no_gloves":
                        if camera_index == "501006":
                            if red_board_flag_first == False:
                                run_model(model, preview_frame, detections,m_name)
                    elif m_name == "model_cart":
                        if camera_index == "501005":
                            run_model_with_cart_exclusion(model, preview_frame, detections, m_name, camera_index)
            
                    else:
                        run_model(model, preview_frame, detections,m_name)
                        #detections.extend(run_model(model, frame, detections,m_name))

                #wrong_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                #white_paper_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                for box in detections:

                    model_name = box["model"]

                    confidence = float(box['conf'][0])
                    cls = int(box['cls'][0])
                    
                    if model_name == "model_no_gloves":
                        if cls == 0:
                            #cv2.imwrite(f"original/{model_name}/{camera_index}_{wrong_time}_ori.jpg",preview_frame)
                            cls = 3
                            event_name_en = class_event_mapping_en.get(3, "Unknown Event")
                        elif cls == 1:
                            cls = 4
                            event_name_en = class_event_mapping_en.get(4, "Unknown Event")
              
                    if model_name == "model_fall":
                        #cv2.imwrite(f"original/{model_name}/{camera_index}_{wrong_time}_ori.jpg",preview_frame)

                        cls = 2
                        event_name_en = class_event_mapping_en.get(2, "Unknown Event")

                    if model_name == "model_cart":
                        cls = 11
                        event_name_en = class_event_mapping_en.get(11, "Unknown Event")    

                    if model_name == "model_fire_smoke":
                        #cv2.imwrite(f"original/{model_name}/{camera_index}_{wrong_time}_ori.jpg",preview_frame)
                        if cls == 0:
                            cls = 0
                            event_name_en = class_event_mapping_en.get(0, "Unknown Event")
                        else:
                            cls = 1
                            event_name_en = class_event_mapping_en.get(1, "Unknown Event")

                    if model_name == "model_red_board":

                        #red_board_detection = True
                        red_board_flag = True #有偵測到紅色板子
                        cls = 5
                        event_name_en = class_event_mapping_en.get(5, "Unknown Event")
                        red_board_frame.append(roi_frame) #把要比較的圖存到buffer裡面
                        red_board_ori_frame.append(preview_frame)
                        #print(len(red_board_frame))

                        x1_box, y1_box, x2_box, y2_box = box['xyxy']
                        roi_frame_location_buffer.append((x1_box, y1_box, x2_box, y2_box))

                        # print("red_board_frame :",len(red_board_frame))
                        
                        #當連續偵測到板子20個frame後，且第20個frame它是此一批次第一個物件或是它的紅色板子冷卻時間已到，代表中段已持續超過cooldown的時間沒有偵測到紅色板子
                        # if (len(red_board_frame) == 20) and ((red_board_flag_first == True) or (red_board_cooldown == 0)):
                        if (len(red_board_frame) == 20) and (red_board_flag_first == True) :
                            #print("standard_frame :",standard_frame.shape)
                            x1_box, y1_box, x2_box, y2_box = box['xyxy']
                            standard_frame = preview_frame[y1_box:y2_box, x1_box:x2_box]

                            if standard_frame.shape[0] <= 1000 or standard_frame.shape[1] <=1000:
                                print("標準答案的圖片尺寸有誤...")
                                break
                            standard_frame = standard_frame[300:600, 300:600]

                            # cv2.imwrite("test/ori.jpg",preview_frame)
                            # cv2.imwrite("test/sample.jpg",standard_frame)

                            #standard_frame = cv2.imread("test/sample.jpg")
                            red_board_flag_first = False

                            number_count_process = False

                            # detection_success = None

                        #當連續偵測到100個frames後，開始比對
                        # if (len(red_board_frame) == 60) and (red_board_flag_first == False):
                        #     # 比對三個區域
                        #     roi_frame = red_board_frame[-30]
                        #     ori_frame = red_board_ori_frame[-30]
                        if (len(red_board_frame) == 40) and (red_board_flag_first == False):
                            # 比對三個區域
                            roi_frame = red_board_frame[-20]
                            ori_frame = red_board_ori_frame[-20]

                            # x1_box, y1_box, x2_box, y2_box = box['xyxy']
                            x1_box, y1_box, x2_box, y2_box = roi_frame_location_buffer[-20]
                            roi_frame = roi_frame[y1_box:y2_box, x1_box:x2_box]

                            if roi_frame.shape[0] <= 1000 or roi_frame.shape[1] <= 1000:
                                print("被比對的圖片尺寸有誤...")
                                break

                            
                            roi_frame = roi_frame[300:600, 300:600]

                            gray1 = cv2.cvtColor(standard_frame, cv2.COLOR_BGR2GRAY)
                            gray2 = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

                            # gray1 = gray1[250:950, 300:850]
                            # gray2 = gray2[250:950, 300:850]

                            score = local_shifted_ssim(gray1, gray2)
                            #print(f"score : {score}")

                            if score < 0.2:
                                diff_Flag = True
                                #cv2.imwrite(f"test/not_aligned_{red_board_count}.jpg",roi_frame)

                            # if diff_Flag:

                            #     cv2.putText(frame, "Not Aligned", (50, 50),
                            #                 cv2.FONT_HERSHEY_SIMPLEX, 2, color_dict[cls], 10)

                            #     cv2.imwrite(f"test/not_aligned_{red_board_count}.jpg",roi_frame)

                            # red_board_frame.pop(0)
                            # red_board_ori_frame.pop(0)
                            red_board_frame = []
                            red_board_ori_frame = []
                            roi_frame_location_buffer = []
                        
                        # red_board_count = red_board_count + 1
                            
                    if model_name == "model_white_paper":
                        # cls 1 > white , cls 0 > black
                        detected = False
                        x1_box, y1_box, x2_box, y2_box = box['xyxy']

                        roi_sets = camera_rois[str(camera_index)]["white_paper_rois"]
                        for roi in roi_sets:
                            x1, y1, x2, y2 = roi
                            # Validate ROI coordinates against frame size
                            x1 = max(0, min(x1, frame_width - 1))
                            x2 = max(0, min(x2, frame_width))
                            y1 = max(0, min(y1, frame_height - 1))
                            y2 = max(0, min(y2, frame_height))
                        

                            if x1 >= x2 or y1 >= y2:
                                print(f"Invalid white_paper_rois ROI for camera {camera_index}: {roi}. Skipping ROI processing.")
                                stop_event.set()
                                break
                            else:
                                if not (x1 <= x1_box <= x2 and y1 <= y1_box <= y2 and x1 <= x2_box <= x2 and y1 <= y2_box <= y2):
                                    continue
                                detected = True
                                if x1 == 650:
                                    if not double_Flag:
                                        continue
                                    else:
                                        if cls == 1:
                                            white_paper_flag = True
                                            
                                            white_mid_time = time.time()
                                            if last_detection_times[roi] == 0:
                                                last_detection_times[roi] = white_mid_time
                                            
                                            roi_detection_times[roi] = white_mid_time - last_detection_times[roi]
                                            red_board_right_final_time = time.time() - red_board_right_time

                                            if (roi_detection_times[roi] >= MIN_DETECTION_SECONDS) and (red_board_right_final_time >= MIN_DETECTION_SECONDS):
                                                if standard_frame is None and double_PP_start == False and not red_board_right_status:
                                                    print(f"物件在 ROI {roi} 連續偵測超過 {MIN_DETECTION_SECONDS} 秒")
                                                double_PP_start = True



                                elif x1 == 0 or x1 == 1950:
                                    if cls == 1:
                                    
                                    
                                        if standard_frame is not None:
                                            x1_box, y1_box, x2_box, y2_box = box['xyxy']
                                            if x1_box < 1500:
                                                number_left_flag = True
                                            elif x1_box >= 1500:
                                                number_right_flag = True
                                    elif cls == 0:

                                        if standard_frame is not None:
                                            x1_box, y1_box, x2_box, y2_box = box['xyxy']
                                            if x1_box >= 1500:
                                                number_right_flag = True
                        
                        if detected:
                            if cls == 1:
                                cls = 7
                                event_name_en = class_event_mapping_en.get(7, "Unknown Event")
                            elif cls == 0:
                                cls = 10
                                event_name_en = class_event_mapping_en.get(10, "Unknown Event")
                        else:
                            continue



                        # if cls == 1:
                        #     cls = 7
                        #     event_name_en = class_event_mapping_en.get(7, "Unknown Event")
                        
                        
                        #     if standard_frame is not None:
                        #         x1_box, y1_box, x2_box, y2_box = box['xyxy']
                        #         if x1_box < 1500:
                        #             number_left_flag = True
                        #         elif x1_box >= 1500:
                        #             number_right_flag = True
                        # elif cls == 0:
                        #     cls = 10
                        #     event_name_en = class_event_mapping_en.get(10, "Unknown Event")
                        #     if standard_frame is not None:
                        #         x1_box, y1_box, x2_box, y2_box = box['xyxy']
                        #         if x1_box >= 1500:
                        #             number_right_flag = True
                    
            

                    if cls != 4:
                        # 排除未戴眼鏡
                        # 畫bounding box在preview frame上
                        x1_box, y1_box, x2_box, y2_box = box['xyxy']

                        cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), color_dict[cls], 10)
                        label = f'{event_name_en} {confidence:.2f}'

                        cv2.putText(frame, label, (x1_box, y1_box - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color_dict[cls], 10)
                    # try:
                    #     if cls == 7:
                    #         cv2.imwrite(f"result/{model_name}/{camera_index}_{white_paper_time}_result.jpg",frame)
                    # except Exception as e:
                    #     pass
                    
                    # try:
                    #     if cls == 3 or cls == 2 or cls == 0 or cls == 1:
                    #         cv2.imwrite(f"result/{model_name}/{camera_index}_{wrong_time}_result.jpg",frame)
                    # except Exception as e:
                    #     pass    

                if camera_index == "501006":
                    if double_Flag:
                        if not white_paper_flag:
                            roi_detection_times[roi] = 0
                            last_detection_times[roi] = 0

                        if red_board_right_status:
                            red_board_right_time = time.time()

                    ### 當個frame若發生版面不一致，存結果
                    if diff_Flag:
                            adjusted_box = {
                                            "model": "red_board_compare",  # 添加模型名稱
                                            'cls': 6,
                                            "detected_frame": ori_frame

                                            }
                            detections.append(adjusted_box)

                    ### 疊合物料是否正確        
                    if number_right_flag and number_left_flag:
                        if number_count_process is not None:
                            number_count_process = True
                    

                    #每次的frame都會去看這次有沒有偵測到red board
                    #如果沒有則清空red board buffer，且會開始倒數cooldown時間，且如果cooldown已倒數完畢，則會把red_board_flag_first變數重新reset
                    if not red_board_flag:
                        red_board_frame = []
                        red_board_ori_frame = []
                        roi_frame_location_buffer = []
                        # if red_board_cooldown > 0 :
                        #     red_board_cooldown = red_board_cooldown - 1

                        # 判斷是否已經開始倒數計時
                        if red_board_start_time is None:
                            red_board_start_time = time.time()  # 開始計時（以當前時間記錄）
                        # 判斷是否超過 1 分鐘
                        elapsed_time = time.time() - red_board_start_time
                        #elif red_board_cooldown == 0:

                        if elapsed_time >= 60:  # 60 秒
                            red_board_flag_first = True
                            # if os.path.exists("test/sample.jpg"):
                            #     os.remove("test/sample.jpg")
                            standard_frame = None

                            if number_count_process is not None:
                                if not number_count_process:
                                    process_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                                    #print(f"疊合物料數量不正確... , time is {process_time}")
                                    #cv2.imwrite(f"photo/{process_time}.jpg",preview_frame)
                                    adjusted_box = {
                                            "model": "number_count_incorrect",  # 添加模型名稱
                                            'cls': 8,
                                            "detected_frame": preview_frame

                                            }
                                    detections.append(adjusted_box)

                                #else:
                                    # 檢測數量正確，將變數reset至None
                                    #print("疊合物料數量正確...")

                                number_count_process = None


                    else:
                        #如果frame有偵測到紅色版子，則重新倒數
                        # red_board_cooldown = 3000
                        #red_board_cooldown = 500
                        red_board_start_time = time.time()  # 重新計時


                if detections:
                    # 將警報處理放入獨立的線程，以避免阻塞
                    alert_thread = threading.Thread(target=send_alert, args=(preview_frame.copy(), camera_index, detections), daemon=True)
                    alert_thread.start()

                    resized_frame = cv2.resize(frame, (640, 400))
                    cv2.imshow(f"{camera_index}",resized_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()

        else:
            time.sleep(0.01)

        # if camera_index == "UT5-1F-06":
        #     end_time = time.time()
        #     elapsed_time = end_time - start_time  # 單位為秒
        #     print(f"迴圈執行時間：{elapsed_time:.4f} 秒")


def local_shifted_ssim(img1, img2, window_size=(25, 25), stride=30, max_shift=5):
    """
    在滑動視窗時容忍微小位移 (max_shift)，找到最佳對齊後的 SSIM。
    """
    img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
    img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)

    h, w = img1.shape[:2]
    win_w, win_h = window_size
    ssim_scores = []

    for y in range(0, h - win_h, stride):

        for x in range(0, w - win_w, stride):
            
            roi1 = img1[y:y+win_h, x:x+win_w]

            max_score = -1
            # 對應範圍內做微調 (x_shift, y_shift)
            for dx in range(-max_shift, max_shift + 1):
                for dy in range(-max_shift, max_shift + 1):
                    xx = x + dx
                    yy = y + dy

                    if 0 <= xx < w - win_w and 0 <= yy < h - win_h:
                        roi2 = img2[yy:yy+win_h, xx:xx+win_w]
                        score = ssim(roi1, roi2)
                        max_score = max(max_score, score)
            
            if max_score != -1:
                ssim_scores.append(max_score)
    
    return np.mean(ssim_scores) if ssim_scores else 0


# =============================================================
# 取得 LINEGPT Token（融合：Cache + Refresh + Thread-safe）
# =============================================================
def get_line_token(force_refresh=False):
    global _token_cache

    now = time.time()

    with _token_lock:

        # ---- Token 尚未過期 → 使用 Cache ----
        if (not force_refresh
            and _token_cache["token"]
            and now < _token_cache["expire_time"]):
            return _token_cache["token"]

        # ---- Token 過期 → 重新登入 ----
        try:
            resp = requests.post(
                LINE_LOGIN_URL,
                json={"username": LINE_USERNAME, "password": LINE_PASSWORD},
                verify=False,
                timeout=(8, 10)
            )

            if resp.status_code == 200:
                data = resp.json()
                token = data.get("token")

                if not token:
                    print("❌ Token 回應缺少 token 欄位")
                    return None

                # 更新 Cache
                _token_cache["token"] = token
                _token_cache["expire_time"] = now + 3600 * 24 * 365

                print("✔ Token 已更新（Cache 生效）")
                return token

            print(f"❌ Login Token 失敗: {resp.status_code}")
            return None

        except Exception as e:
            print(f"❌ Token API 例外：{e}")
            return None


# =============================================================
# 傳送 LINEGPT 訊息（含圖片 + Token Refresh）
# =============================================================
def send_line_message(message, file_path=None, retries=3):

    token = get_line_token()
    if not token:
        print("❌ 無法取得 Token，放棄傳送")
        return False

    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message, "chatId": DEFAULT_CHAT_ID}

    for attempt in range(1, retries + 1):
        try:
            files = None
            if file_path and os.path.exists(file_path):
                files = {
                    "file": (
                        os.path.basename(file_path),
                        open(file_path, "rb"),
                        "image/jpeg"
                    )
                }

            resp = requests.post(
                LINE_NOTIFY_URL,
                headers=headers,
                data=data,
                files=files,
                verify=False,
                timeout=(10, 15)
            )

            # 成功
            if resp.status_code in (200, 201):
                print(f"✔ LINEGPT 已送出（attempt {attempt}）")
                return True

            # Token 過期 → 刷新後重送
            if resp.status_code == 401:
                print("⚠ Token 過期 → 正在強制刷新")
                token = get_line_token(force_refresh=True)
                headers = {"Authorization": f"Bearer {token}"}
                continue

            print(f"❌ LINEGPT 回傳錯誤：{resp.status_code}")
            return False

        except Exception as e:
            print(f"❌ LINEGPT 例外：{e}")

        finally:
            if file_path and os.path.exists(file_path) and files:
                try:
                    files["file"].close()
                except:
                    pass

    return False


# =============================================================
# 產生企業級【影像辨識通知】格式
# =============================================================
def format_alert_message(location, feature, content):
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    msg = f"""
【影像辨識通知】
系統已偵測到疑似違規行為或潛在安全風險：

📍 地點：{location}
🕒 時間：{now_str}
🧠 特徵項目：{feature}
📄 內容：{content}

請儘速處理此事件並依據公司規定採取適當行動。
若需詳細影像，請聯繫資訊處系統一課。
問題回報：https://forms.gle/rFZXVRP1aUxqQNG97
""".strip()

    return msg


# =============================================================
# 🔥 send_alert：整合 Screenshot + LINEGPT + EIP
# =============================================================
def send_alert(send_frame, camera_index, detections):
    """
    企業級整合：
    - Screenshot
    - 冷卻控管
    - LINEGPT + 圖片
    - EIP UploadYoloImage
    """

    current_time = time.time()
    cls_buffer = []

    # 依你的 class_alert_cooldowns / last_alert_times 運行（不改動你的邏輯）
    for box in detections:
        model_name = box["model"]

        if model_name in ["model_red_board", "model_white_paper", "model_stick_hand"]:
            continue

        if model_name in ["red_board_compare", "number_count_incorrect"]:
            cls = int(box["cls"])
        else:
            cls = int(box["cls"])
            confidence = float(box["conf"][0])

        # 類別歸一（依你原本 3/2/0/1/11）
        if model_name == "model_no_gloves":
            cls = 3
        elif model_name == "model_fall":
            cls = 2
        elif model_name == "model_fire_smoke":
            cls = 0 if cls == 0 else 1
        elif model_name == "model_cart":
            cls = 11

        cooldown_time = class_alert_cooldowns.get(cls, 300)
        if cls not in last_alert_times[str(camera_index)]:
            last_alert_times[str(camera_index)][cls] = 0

        if current_time - last_alert_times[str(camera_index)][cls] <= cooldown_time:
            continue

        last_alert_times[str(camera_index)][cls] = current_time
        cls_buffer.append({**box, "cls": cls})

    if not cls_buffer:
        return

    # ========== Screenshot ==========
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{camera_index}-{timestamp}.jpg"
    full_path = f"images/{filename}"

    for box in cls_buffer:
        x1, y1, x2, y2 = box['xyxy']
        cls = box["cls"]
        cv2.rectangle(send_frame, (x1, y1), (x2, y2), color_dict[cls], 10)

    success = cv2.imwrite(full_path, send_frame)
    if success:
        print(f"🟩 [Camera {camera_index}] Screenshot 成功：{filename} ({frame_width}x{frame_height})")
    else:
        print(f"🟥 [Camera {camera_index}] Screenshot 失敗：{filename}")
        print(f"    ➤ send_frame 類型：{type(send_frame)}")
        try:
            print(f"    ➤ send_frame 尺寸：{send_frame.shape}")
        except:
            print("    ➤ send_frame 無法取得 shape（可能為 None 或 decode 失敗）")

    # ========== 事件資訊 ==========
    location = "十二課預疊合室"
    event_names = [
        class_event_mapping_cn.get(int(box["cls"]), "未知事件")
        for box in cls_buffer
    ]
    formatted_event = "；".join(event_names)

    # ========== LINEGPT 通知 ==========
    try:
        alert_msg = format_alert_message(
            location=location,
            feature=formatted_event,
            content=f"攝影機 {camera_index} 偵測到：{formatted_event}，請現場確認。"
        )

        send_line_message(
            alert_msg,
            file_path=full_path
        )

        print(f"✔ LINEGPT 已通知：{filename}")

    except Exception as e:
        print(f"❌ LINEGPT 發送例外：{e}")

    # ========== EIP 上傳 ==========
    try:
        api_url = "https://eip.pcbut.com.tw/File/UploadYoloImage"

        camera_model = {
            "cameraId": camera_index,
            "location": location,
            "eventName": formatted_event,
            "eventDate": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": f"{len(cls_buffer)} events detected.",
            "fileName": filename,
            "result": f"疑似發生 {formatted_event}, 請同仁儘速查看"
        }

        with open(full_path, "rb") as img_file:
            files = {"files": (filename, img_file, "image/jpeg")}
            r = requests.post(api_url, data=camera_model, files=files, verify=False)

        if r.status_code == 200:
            print(f"✔ EIP 上傳成功：{filename}")
        else:
            print(f"❌ EIP 回傳：{r.status_code}, {r.text}")

    except Exception as e:
        print(f"❌ EIP 同步例外：{e}")


    #print(last_alert_times)

# Global stop event for all threads
stop_event = threading.Event()

def batch_alert():
    """
    處理批次警報的函數，每隔一段時間檢查一次 alert_records 並發送報告。
    """
    while not stop_event.is_set():
        time.sleep(alert_interval)
        with mutex:
            if alert_records:
                alert_message = f"Alert Report: {len(alert_records)} events detected."
                print(alert_message)
                # 在此添加批次警報郵件發送邏輯
                alert_records.clear()


if __name__ == '__main__':
    # Start batch alert processing thread #daemon: 主程式結束時強制結束thread
    alert_thread = threading.Thread(target=batch_alert, daemon=True)
    alert_thread.start()

    # Start threads for all cameras
    camera_threads = []
    # for index, url in enumerate(camera_urls):
    #     threads = process_camera(index, url)
    #     camera_threads.extend(threads)

    # global video
    # # 讀取影片
    # #cap = cv2.VideoCapture("D:/AI/Demo_eden/edenTest/datasets/fall/test_fall/queda.mp4")
    # cap = cv2.VideoCapture("demo_12_preLam_add_data.mp4")
    # output_video = "0303.mp4"
    # # 確保影片成功讀取
    # if not cap.isOpened():
    #     print("無法讀取影片！")
    #     exit()
    # # 取得影片資訊
    # fps = int(cap.get(cv2.CAP_PROP_FPS))  # 幀率
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 影片寬度
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 影片高度
    # # 設定影片編碼格式與輸出物件
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 影片格式（mp4）
    # video = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))


    # 閾值設定
    similar_compare_threshold = 18  # 可根據實際情況調整

    for index, (key,value) in enumerate(camera_urls.items()):
        camera_threads.append(threading.Thread(target=receive_frames, args=(key, value), daemon=True))
        camera_threads.append(threading.Thread(target=inference_worker, args=(key,), daemon=True))
        # threads = process_camera(key, value)
        # camera_threads.extend(threads)
    for t in camera_threads:
        t.start()

    try:
        # Wait for all threads to complete
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        print("Interrupt received, stopping all threads...")
    finally:
        for thread in camera_threads:
            thread.join(timeout=2)
        alert_thread.join(timeout=2)
        cv2.destroyAllWindows()
        print("All resources released, exiting.")