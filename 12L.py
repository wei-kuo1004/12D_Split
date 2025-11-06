import os
import sys

# Temporarily comment out to allow error messages for debugging
# Suppress FFmpeg error messages
# sys.stderr = open(os.devnull, 'w')
# Set FFmpeg log level to quiet
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"
import av
import cv2
import queue
import time
import threading
from ultralytics import YOLO
import requests
import numpy as np
from collections import deque


# Force detection to use CPU
device = "cpu"
# print(f"Using device: {device}")

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
        4: 300,   # 使用手機
        5: 300,
        6: 300,
        7: 300, # 黏手
        8: 300  # 黏手作業錯誤
    }

# Mutex lock to ensure thread safety
mutex = threading.Lock()

# Load YOLOv8 model (ensure this model is trained with the 7 specified classes)
# model_fire_smoke = YOLO('model/fire_smoke/best_0312.pt').to(0)  # Replace 'best.pt' with your model file
# model_fall = YOLO('model/fall/best_0318.pt').to(0)
# model_no_gloves = YOLO('model/gloves_goggles/best_0328.pt').to(0)
# #model_cellphone = YOLO('D:/AI/Demo_eden/edenTest/model_cellphone_label_studio_reduce_v2/model_collections/weights/best.pt').to(0)
# model_pose = YOLO("yolo11n-pose.pt").to(0)
# model_foreign_objects = YOLO('model/foreign_objects/best_0312.pt').to(0)
# model_stick_hand = YOLO('model/stick_hand/best.pt').to(0)

# Mapping from class index to event name (English)
class_event_mapping_en = {
    0: "fire",
    1: "smoke",
    2: "fall",
    3: "no gloves",
    4: "without goggles",
    5: "imfrared not aligned",
    6: "foreign objects",
    7: "stick hand",
    8: "stick hand process wrong"

}

# Mapping from class index to event name (Chinese)
class_event_mapping_cn = {
    0: "火光",
    1: "煙霧",
    2: "人員倒臥",
    3: "未戴手套",
    4: "未戴護目鏡",
    5: "未對齊紅外線",
    6: "異物",
    7: "黏手",
    8: "黏手作業錯誤"

}


color_dict ={
    0: (255,0,0), #紅
    1: (0,255,0), #綠
    2: (0,0,255), #藍
    3: (255,255,0), #淡藍
    4: (255,0,255),
    5: (0,255,255),
    6: (255,255,255),
    7: (126,126,126),
    8: (126,0,126)
}


# ]
last_alert_times = {}
camera_urls = {"501012":"rtsp://hikvision:Unitech0815!@10.80.21.141",
               "501016":"rtsp://hikvision:Unitech0815!@10.80.21.145",
               "501008":"rtsp://hikvision:Unitech0815!@10.80.21.137",
               "501004":"rtsp://hikvision:Unitech0815!@10.80.21.133",
               "501003":"rtsp://hikvision:Unitech0815!@10.80.21.132",
               "501007":"rtsp://hikvision:Unitech0815!@10.80.21.136",
               "501011":"rtsp://hikvision:Unitech0815!@10.80.21.140",
               "501015":"rtsp://hikvision:Unitech0815!@10.80.21.144"
               }

# camera_urls = {
#                "UT5-1F-12":"rtsp://hikvision:Unitech0815!@10.80.21.141",
#                "UT5-1F-16":"rtsp://hikvision:Unitech0815!@10.80.21.145",
#                "UT5-1F-08":"rtsp://hikvision:Unitech0815!@10.80.21.137",
#                "UT5-1F-04":"rtsp://hikvision:Unitech0815!@10.80.21.133",
#                }


for i in camera_urls:
    if i not in last_alert_times:
        last_alert_times[i] = {}
#print(last_alert_times)


camera_rois = {"501015":(900, 450,2550, 1500) , "501011":(600, 350,2100, 1350) , "501007":(850, 625,2500, 1615) , "501003":(550, 525,2150, 1475)} #144


##肢體判斷
landmark_parameters = {"501015":(2775,600,1300,1000,400,450) , "501011":(2350,600,1100,900,350,415) ,
                        "501007":(2650,600,1350,1100,400,600) , "501003":(2150,300,1000,1100,280,480)} #144

#l1是 take
#l2 左手跟右手的距離
#l3 右手的邊界
#l4是 縮小landmark的邊界，降低軀幹偵測的誤報
#l5是 更優化take flag的判定，多了y軸的判斷
#l6是 排除作業員在清潔疊合台時會產生的誤報，用檢測頭的位置來達成


stick_hand_parameters = {"501012":{"A_ROI": (1160, 1500, 1460, 1780),"B_ROI": (950, 300, 1900, 1300)},
                        "501016":{"A_ROI": (1100, 1400, 1430, 1700),"B_ROI": (350, 400, 1600, 950)},
                        "501008":{"A_ROI": (1100, 1550, 1400, 1780),"B_ROI": (1000, 400, 1650, 1300)},
                        "501004":{"A_ROI": (1350, 1650, 1730, 1780),"B_ROI": (950, 450, 1350, 1450)}}

# 每支攝影機的最新幀記憶區
frame_deques = {cam: deque(maxlen=1) for cam in camera_urls.keys()}

# 模型初始化（GPU ID 視需要調整）
model_map = {
    "model_no_gloves": YOLO('model/gloves_goggles/best_0428.pt').to(0),
    "model_fire_smoke": YOLO('model/fire_smoke/best_0312.pt').to(0),
    "model_fall": YOLO('model/fall/best_0430.pt').to(0),
    "model_pose": YOLO('yolo11n-pose.pt').to(0),
    "model_foreign_objects": YOLO('model/foreign_objects/best_0312.pt').to(0),
    "model_stick_hand": YOLO('model/stick_hand/best.pt').to(0)
}

# 每個攝影機分別推論哪些模型
camera_models = {
    "501012": [ "model_fire_smoke","model_stick_hand","model_fall"],
    "501016": [ "model_fire_smoke","model_stick_hand","model_fall"],
    "501008": [ "model_fire_smoke","model_stick_hand","model_fall"],
    "501004": [ "model_fire_smoke","model_stick_hand","model_fall"],
    "501003": [ "model_pose", "model_foreign_objects"],
    "501007": [ "model_pose", "model_foreign_objects"],
    "501011": [ "model_pose", "model_foreign_objects"],
    "501015": [ "model_pose", "model_foreign_objects"]
}

def remove_key(d, key):
    if key in d:
        del d[key]
    return d

# def receive_frames(cam_id, url):
#     print(f"📡 Starting receiver for {cam_id}")
#     cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#     while not stop_event.is_set():
#         ret, frame = cap.read()
#         if ret:
#             frame_deques[cam_id].append(frame)
#         else:
#             print(f"⚠️ {cam_id} disconnected, retrying...")
#             time.sleep(1)
#     cap.release()
#     print(f"🛑 Receiver stopped for {cam_id}")

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

def run_model(model, frame, detections,model_name,roi=None):
    #detections = []

    # if model_name == "model_stick_hand":
    #     results = model(frame, conf=0.3 ,verbose=False)
    # results = model(frame, conf=0.8 ,verbose=False)

    if model_name == "model_stick_hand":
        results = model.predict(frame, conf=0.5,imgsz=640,verbose=False)
    else:
        results = model.predict(frame, conf=0.9,imgsz=640,verbose=False)

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            cls = int(box.cls[0])  # Get class index
            
            x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])

            if model_name == "model_foreign_objects" and roi:
                if confidence > alert_threshold:
                    #x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])

                    # 檢查該類別是否受 ROI 限制
                    # 假設類別 1 (without_goggles) 只能在 ROI 偵測
                    #print(roi)
                    x1, y1, x2, y2 = roi
                    if not (x1 <= x1_box <= x2 and y1 <= y1_box <= y2 and 
                            x1 <= x2_box <= x2 and y1 <= y2_box <= y2):
                        continue  # 如果 bbox 不在 ROI 內，跳過
                    
                    adjusted_box = {
                            "model": model_name,  # 添加模型名稱
                            'xyxy': [x1_box, y1_box, x2_box, y2_box],
                            'conf': box.conf,
                            'cls': box.cls
                            }
                    detections.append(adjusted_box)
            elif model_name == "model_stick_hand":
                if confidence > 0.5:
                    # Get bounding box coordinates
                    #x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])
                    adjusted_box = {
                        "model": model_name,  # 添加模型名稱
                        'xyxy': [x1_box, y1_box, x2_box, y2_box],
                        'conf': box.conf,
                        'cls': box.cls
                        }
                    detections.append(adjusted_box)
            else:
                if confidence > alert_threshold:
                    #x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])

                    adjusted_box = {
                            "model": model_name,  # 添加模型名稱
                            'xyxy': [x1_box, y1_box, x2_box, y2_box],
                            'conf': box.conf,
                            'cls': box.cls
                            }
                    detections.append(adjusted_box)


def inference_worker(camera_index):
    models = camera_models.get(camera_index, [])

    landmark_bounding = None
    landmark_box = None
    print(f"Starting to display camera {camera_index}")
    #imfrared_flag_first = True
    threshold = 42

    #state = "WAITING_FOR_A"

    # # 讀取影片
    # output_video = f"{camera_index}.mp4"

    # # 設定影片編碼格式與輸出物件
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 影片格式（mp4）
    # video = cv2.VideoWriter(output_video, fourcc, 20, (3200, 1800))

    camera_states = {}
    #imfrared_flag_first = True

    # 每個 camera 啟動時初始化
    camera_states[camera_index] = {
        "disappear_count": 2000,
        "take_flag": False,
        "put_flag": False,
        "take_flag_count": 0,
        "put_flag_count": 0,
        "consecutive_frames": 0,
        "foreign_objects_flag": False,
        "imfrared_flag_first": True,
        "state": "WAITING_FOR_A",
        "hands_last_location" : None 
    }
    camera_index_state = camera_states[camera_index]

    while not stop_event.is_set():
        # if camera_index == "UT5-1F-07":
        #     start_time = time.time()
        

        if frame_deques[camera_index]:

            #frame = frame_deques[camera_index][-1]
            detections = []

            raw_frame = frame_deques[camera_index][-1]

            #preview_frame = frame.copy()
            frame = raw_frame.copy()  # 明確切乾淨
            preview_frame = raw_frame.copy()

            roi_box = None
            roi_flag = False

            if frame is not None:
                imfrared_flag_first = camera_index_state["imfrared_flag_first"]
                state = camera_index_state["state"]
                hands_last_location = camera_index_state["hands_last_location"]

                frame_height, frame_width = frame.shape[:2]
                # print(f"Camera {camera_index} - Frame size: {frame_width}x{frame_height}")
                
                camera_states[camera_index]["foreign_objects_flag"] = False

                # Retrieve ROI for this camera
                if str(camera_index) in camera_rois:
                    roi = camera_rois[str(camera_index)]
                    x1, y1, x2, y2 = roi
                    # Validate ROI coordinates against frame size
                    x1 = max(0, min(x1, frame_width - 1))
                    x2 = max(0, min(x2, frame_width))
                    y1 = max(0, min(y1, frame_height - 1))
                    y2 = max(0, min(y2, frame_height))


                    if x1 >= x2 or y1 >= y2:
                        print(f"Invalid ROI for camera {camera_index}: {roi}. Skipping ROI processing.")
                        roi = None
                        roi_frame = frame
                    else:
                        top_left = (x1, y1)  # 左上角座標
                        bottom_right = (x2, y2)  # 右下角座標

                        # Draw ROI rectangle on the frame
                        roi_flag = True
                        #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        #roi_frame = frame
                        roi_box = (x1,y1,x2,y2)
                        # # Optionally, mask out the area outside ROI for detection
                        # mask = frame.copy()
                        # mask[:y1, :] = 0
                        # mask[y2:, :] = 0
                        # mask[:, :x1] = 0
                        # mask[:, x2:] = 0
                        # roi_frame = mask
                else:
                    roi = None
                    roi_frame = frame  # If no ROI defined, use the whole frame
                    #preview_frame = frame.copy()

                if camera_index == "501004" or camera_index == "501008" or camera_index == "501012" or camera_index == "501016":
                    glove_A = False
                    glove_B = False
                    A_ROI = stick_hand_parameters[str(camera_index)]["A_ROI"]
                    B_ROI = stick_hand_parameters[str(camera_index)]["B_ROI"]
                
                    # ROI 及畫面顯示
                    cv2.rectangle(frame, (A_ROI[0], A_ROI[1]), (A_ROI[2], A_ROI[3]), (0, 0, 255), 10)
                    cv2.rectangle(frame, (B_ROI[0], B_ROI[1]), (B_ROI[2], B_ROI[3]), (0, 255, 0), 10)
                if camera_index == "501003" or camera_index == "501007" or camera_index == "501011" or camera_index == "501015":
                    if str(camera_index) in landmark_parameters:
                        roi = landmark_parameters[str(camera_index)]
                        l1 , l2 , l3 , l4 , l5 , l6 = roi
                        #l4是 縮小landmark的邊界，降低軀幹偵測的誤報
                        #l5是 更優化take flag的判定，多了y軸的判斷
                        #l6是 排除作業員在清潔疊合台時會產生的誤報，用檢測頭的位置來達成
                        landmark_box = (l1,l2,l3,l5,l6)
                        landmark_bounding = l4
                    else:
                        print("沒有對應的肢體參數...")
                        sys.exit()

                if roi_flag == True:
                    if camera_index == "501003" or camera_index == "501007" or camera_index == "501011" or camera_index == "501015":
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 10)
                for m_name in models:
                    model = model_map[m_name]

                    if camera_index == "501003" or camera_index == "501007" or camera_index == "501011" or camera_index == "501015":
                        if m_name == "model_pose":
                            pose_model(model, frame, detections, m_name,landmark_box,landmark_bounding,camera_index,camera_states)
                        else:
                            #detections.extend(run_model(model, frame, detections,m_name,roi_box))
                            run_model(model, preview_frame, detections,m_name,roi_box)
                    else:
                        #detections.extend(run_model(model, frame, detections,m_name,roi_box))
                        run_model(model, preview_frame, detections,m_name,roi_box)
                
                #wrong_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                for box in detections:
                    

                    model_name = box["model"]
                    if (model_name != "model_pose") & (model_name != "imfrared_not_aligned") & (model_name != "stick_hand_process_wrong"):
                        confidence = float(box['conf'][0])
                        cls = int(box['cls'][0])
                    if model_name == "model_no_gloves":
                        if cls == 0:
                            
                            #cv2.imwrite(f"original/{model_name}/{camera_index}_{wrong_time}_ori.jpg",preview_frame)

                            cls = 3
                            event_name_en = class_event_mapping_en.get(3, "Unknown Event")
                        elif cls == 1:
                            cls = 4
                    # if model_name == "cellphone":
                    #     print("good")
                    #     event_name_en = class_event_mapping_en.get(4, "Unknown Event")
                    if model_name == "model_fall":
                        #cv2.imwrite(f"original/{model_name}/{camera_index}_{wrong_time}_ori.jpg",preview_frame)

                        cls = 2
                        event_name_en = class_event_mapping_en.get(2, "Unknown Event")
                    if model_name == "model_fire_smoke":
                        #cv2.imwrite(f"original/{model_name}/{camera_index}_{wrong_time}_ori.jpg",preview_frame)

                        if cls == 0:
                            cls = 0
                            event_name_en = class_event_mapping_en.get(0, "Unknown Event")
                        else:
                            cls = 1
                            event_name_en = class_event_mapping_en.get(1, "Unknown Event")
                    if model_name == "model_foreign_objects":
                        

                        cls = 6
                        event_name_en = class_event_mapping_en.get(6, "Unknown Event")

                        foreign_objects_flag = camera_states[camera_index]["foreign_objects_flag"]

                        if foreign_objects_flag == True:
                            cleaned_data = remove_key(detections, "model_foreign_objects")
                            continue

                        #cv2.imwrite(f"original/{model_name}/{camera_index}_{wrong_time}_ori.jpg",preview_frame)


                    if model_name == "model_stick_hand":

                        cls = 7
                        event_name_en = class_event_mapping_en.get(7, "Unknown Event")
                    
                    if (model_name != "model_pose") & (model_name != "imfrared_not_aligned") & (model_name != "stick_hand_process_wrong"):
                        if cls != 4:
                            x1_box, y1_box, x2_box, y2_box = box['xyxy']
                            cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), color_dict[cls], 10) # 2   10
                            label = f'{event_name_en} {confidence:.2f}'
                            cv2.putText(frame, label, (x1_box, y1_box - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, color_dict[cls], 10)
                    # try:
                    #     if cls == 3 or cls == 2 or cls == 0 or cls == 1 or cls == 6:
                    #         cv2.imwrite(f"result/{model_name}/{camera_index}_{wrong_time}_result.jpg",frame)
                    # except Exception as e:
                    #     pass
                    
                    #imfrared compare
                    if model_name == "model_pose":

                        if imfrared_flag_first == True:

                            standard_frame = box["compare_frame"]
                            
                            standard_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                            cv2.imwrite(f"test/{camera_index}_{standard_time}_standard.jpg",standard_frame)

                            imfrared_flag_first = False
                        else:
                            # cv2.line(preview_frame, (0,l6), (3200,l6), (255, 255, 0), 10)
                            # cv2.line(preview_frame, (0,l5), (3200,l5), (0, 0, 255), 10)
                            compare_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                            cv2.imwrite(f"test/{camera_index}_{compare_time}_compare.jpg",preview_frame)

                            be_compared_frame = box["compare_frame"]
                            
                            if camera_index == "501015":
                                # 定義 ROI 區域 #144

                                # 定義 3 條比對線（相對於 ROI）  # 144
                                line_1 = ((400, 52), (750, 50))
                                line_2 = ((857, 350), (860, 650))
                                line_3 = ((870, 50), (1300, 73))

                                line_4 = ((400, 350), (750, 345))
                                line_5 = ((950, 350), (1300, 364))
                            elif camera_index == "501011": #140
                                # 定義 ROI 區域 #140

                                # 定義 3 條比對線（相對於 ROI）  # 140
                                line_1 = ((95, 80), (550, 45))
                                line_2 = ((710, 350), (705, 650))
                                line_3 = ((850, 45), (1300, 60))

                                line_4 = ((95, 380), (550, 345))
                                line_5 = ((850, 340), (1300, 355))
                            elif camera_index == "501007": #136

                                # 定義 3 條比對線（相對於 ROI）  # 136
                                line_1 = ((125, 88), (590, 70))
                                line_2 = ((785, 350), (790, 650))
                                line_3 = ((850, 60), (1300, 55))

                                line_4 = ((125, 388), (590, 370))
                                line_5 = ((850, 360), (1300, 355))
                            elif camera_index == "501003": #132

                                line_1 = ((125, 130), (590, 80))
                                line_2 = ((700, 350), (705, 650))
                                line_3 = ((850, 65), (1300, 45))

                                line_4 = ((125, 430), (590, 380))
                                line_5 = ((850, 365), (1300, 345))


                            # 轉換為灰階
                            image1 = cv2.cvtColor(standard_frame, cv2.COLOR_BGR2GRAY)
                            image2 = cv2.cvtColor(be_compared_frame, cv2.COLOR_BGR2GRAY)
                            image3 = be_compared_frame.copy()


                            # 確保圖片尺寸一致
                            if image1.shape != image2.shape:
                                print(f"圖片尺寸不一致: {image1.shape} vs {image2.shape}")
                                continue
                                
                            # 裁剪 ROI
                            roi1 = image1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                            roi2 = image2[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                            roi3 = image3[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                            # 比對三條線
                            aligned_1, diff_1  = compare_line_pixels(roi1, roi2, *line_1, threshold, "Line 1",roi3,preview_frame,camera_index)
                            aligned_2, diff_2  = compare_line_pixels(roi1, roi2, *line_2, threshold, "Line 2",roi3,preview_frame,camera_index)
                            aligned_3, diff_3  = compare_line_pixels(roi1, roi2, *line_3, threshold, "Line 3",roi3,preview_frame,camera_index)
                            aligned_4, diff_4  = compare_line_pixels(roi1, roi2, *line_4, threshold, "Line 3",roi3,preview_frame,camera_index)
                            aligned_5, diff_5  = compare_line_pixels(roi1, roi2, *line_5, threshold, "Line 3",roi3,preview_frame,camera_index)


                            if diff_1 >5:
                                aligned_1 = False
                                #not_aligned_lines.append("Line 1")
                            else:
                                aligned_1 = True
                                
                            if diff_2 >5:
                                aligned_2 = False
                                #not_aligned_lines.append("Line 2")
                            else:
                                aligned_2 = True
                                
                            if diff_3 >5:
                                aligned_3 = False
                                    #not_aligned_lines.append("Line 3")
                            else:
                                aligned_3 = True
                            
                            if diff_4 >6:
                                aligned_4 = False
                            else:
                                aligned_4 = True
                            
                            if diff_5 >6:
                                aligned_5 = False
                            else:
                                aligned_5 = True

                            #is_aligned = aligned_1 and aligned_2 and aligned_3

                            # 判斷整體對齊情況
                            if (aligned_1 or aligned_4) and (aligned_3 or aligned_5) and (aligned_2 or aligned_4) and (aligned_2 or aligned_5):
                                #alignment_status = f"not aligned ({', '.join(not_aligned_lines)})"
                                alignment_status = "aligned"                                        
                            else:
                                alignment_status = "not aligned"
                                adjusted_box = {
                                "model": "imfrared_not_aligned",  # 添加模型名稱
                                "detected_frame": preview_frame,
                                "cls": 5
                                        }
                                detections.append(adjusted_box)

                                # # 要顯示的文字
                                # text = "紅外線未對齊"

                                # # 設定位置 (x, y)
                                # position = (10, 10)  # 距離左上角 (10 px, 30 px)，你可以微調

                                # # 字型、大小、顏色、粗細
                                # font = cv2.FONT_HERSHEY_SIMPLEX
                                # font_scale = 5
                                # color = (0, 255, 0)  # 綠色 (BGR)
                                # thickness = 10

                                # cv2.putText(process_frame, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
                                # cv2.imwrite(f"test/{camera_index}_wrong.jpg",process_frame)
                            
                            #print(alignment_status)
                    if model_name == "model_stick_hand":
                        #x1, y1, x2, y2 = map(int, result.xyxy[0])
                        x1_box, y1_box, x2_box, y2_box = box['xyxy']
                        if is_inside_roi((x1_box, y1_box, x2_box, y2_box), A_ROI):
                            glove_A = True
                        if is_inside_roi((x1_box, y1_box, x2_box, y2_box), B_ROI):
                            glove_B = True

                if camera_index == "501004" or camera_index == "501008" or camera_index == "501012" or camera_index == "501016":
                    if state == "WAITING_FOR_A":
                        if glove_A:
                            glove_timer = time.time()
                            state = "WAITING_FOR_B"
                            #print(f"🟠 {camera_index} 偵測到進入 A 區，等待 B 區")
                    elif state == "WAITING_FOR_B":
                        if glove_B:
                            #print(f"🟢 {camera_index} 清潔完成")
                            state = "FLOW_COMPLETE"
                        elif time.time() - glove_timer > 10:
                            #print(f"🔴 {camera_index} 清潔超時")
                            #take_screenshot(frame, camera_id, location)
                            state = "FLOW_COMPLETE"

                            adjusted_box = {
                            "model": "stick_hand_process_wrong",  # 添加模型名稱
                            "process_wrong_frame": preview_frame,
                            "cls": 8
                                    }
                            detections.append(adjusted_box)

                    elif state == "FLOW_COMPLETE":
                        if not glove_A and not glove_B:
                            state = "WAITING_FOR_A"
                            #print(f"🔁 {camera_index} 等待下一輪清潔流程")

                if detections:
                    # 將警報處理放入獨立的線程，以避免阻塞
                    alert_thread = threading.Thread(target=send_alert, args=(preview_frame.copy(), camera_index, detections), daemon=True)
                    alert_thread.start()
                if camera_index == "501003" or camera_index == "501007" or camera_index == "501011" or camera_index == "501015":
                    disappear_count = camera_states[camera_index]['disappear_count']
                    #if disappear_count == 0: #修正
                    if disappear_count == 1700: #修正
                        standard_frame = None
                        imfrared_flag_first = True
                        hands_last_location = None


                camera_index_state["imfrared_flag_first"] = imfrared_flag_first
                camera_index_state["state"] = state

                # if camera_index == "501003" or camera_index == "501007" or camera_index == "501011" or camera_index == "501015":
                #     # if camera_index == "501015":
                #     #     cv2.line(frame, (0,l6), (3200,l6), (255, 255, 0), 10)
                #     #     cv2.line(frame, (0,l5-100), (3200,l5-100), (0, 0, 255), 10)
                #     resized_frame = cv2.resize(frame, (640, 400))

                #     cv2.imshow(f"{camera_index}",resized_frame)

                # resized_frame = cv2.resize(frame, (640, 400))

                # cv2.imshow(f"{camera_index}",resized_frame)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()

        else:
            time.sleep(0.01)
        # if camera_index == "501016":
        #     end_time = time.time()
        #     elapsed_time = end_time - inference_time  # 單位為秒
        #     print(f"迴圈執行時間：{elapsed_time:.4f} 秒")



def is_inside_roi(box, roi):
    x1, y1, x2, y2 = box
    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    # 檢查是否有任何交集
    overlap_x = not (x2 < roi_x1 or x1 > roi_x2)
    overlap_y = not (y2 < roi_y1 or y1 > roi_y2)

    result = overlap_x and overlap_y
    # print(f"🎯 ROI碰撞檢查 Box:({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) → 接觸: {result}")
    return result

def compare_line_pixels(image1, image2, start_point, end_point, threshold, line_name,image3,preview_frame,camera_index):
    """
    針對一條直線上的每個像素點，比較 RGB 通道的像素差異
    """

    aligned = True
    diff_count = 0  # 記錄不匹配的像素數量

    # 計算直線上的所有像素點
    line_points = list(zip(
        np.linspace(start_point[0], end_point[0], num=10).astype(int),
        np.linspace(start_point[1], end_point[1], num=10).astype(int)
    ))


    for x, y in line_points:
        if y >= image1.shape[0] or x >= image1.shape[1]:  # 確保點在 ROI 範圍內
            continue

        # 取得 RGB 像素值
        pixel_value_1 = image1[y, x]
        pixel_value_2 = image2[y, x]
        pixel_value_3 = image3[y, x]

        # 計算 RGB 通道的差異
        #diff = np.linalg.norm(pixel_value_1.astype(int) - pixel_value_2.astype(int))  # L2 範數
        diff = abs(int(pixel_value_1) - int(pixel_value_2))  # 直接使用絕對差異

        if diff > threshold:
            aligned = False
            diff_count += 1
            #cv2.circle(image2, (x, y), 3, (0, 0, 255), -1)  # 紅色標記未對齊的點
            cv2.circle(image3, (x, y), 3, (0, 0, 255), -1)
        else:
            #cv2.circle(image2, (x, y), 3, (0, 255, 0), -1)  # 綠色標記對齊的點
            cv2.circle(image3, (x, y), 3, (0, 255, 0), -1)

    # 標示比對線
    #cv2.line(image2, start_point, end_point, (255, 255, 0), 2)

    cv2.line(image3, start_point, end_point, (255, 255, 0), 2)
    #cv2.imshow("red windows",image3)
    # check_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    # cv2.imwrite(f"compare/{camera_index}_ORI_{check_time}.jpg",preview_frame)

    # cv2.imwrite(f"compare/{camera_index}_result_{check_time}.jpg",image3)

    # compare_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    # cv2.imwrite(f"compare/{camera_index}_{compare_time}.jpg",preview_frame)

    # process_frame = preview_frame.copy()

    return aligned, diff_count

def pose_model(model, frame, detections, model_name,landmark_box,landmark_bounding,camera_index,camera_states):
    total_keypoints  = {}
    shoulder_keypoints = {}
    hand_keypoints = {}

    ori_frame = frame.copy()

    frame = frame[0:landmark_bounding, 0:3200]


    results = model(frame,imgsz=640,verbose=False,conf=0.3)

    state = camera_states[camera_index]  # 獲取此 camera 的變數狀態

    disappear_count = state["disappear_count"]
    take_flag = state["take_flag"]
    put_flag = state["put_flag"]
    take_flag_count = state["take_flag_count"]
    put_flag_count = state["put_flag_count"]
    consecutive_frames = state["consecutive_frames"]
    foreign_objects_flag = state["foreign_objects_flag"]
    hands_last_location = state["hands_last_location"]

    detected = False

    # 定義骨架連線關係
    skeleton_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 頭部
        (1, 5), (2, 6), (5, 6),  # 肩膀
        (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
        (5, 11), (6, 12), (11, 12),  # 胯部
        (11, 13), (13, 15), (12, 14), (14, 16)  # 腿部
    ]

    l1 , l2 ,l3  ,l5 , l6 = landmark_box
    hand_determination = int((l6+l5)/2)
    if camera_index == "501011":
        hand_determination = l5 - 100 #修正
    elif camera_index == "501015":
        hand_determination = l5 - 100 #修正



    # 設定顏色
    hand_side = [9,10]  # 右邊關鍵點（紅色）
    other_side = [1, 2,3,4, 5,6, 7,8, 11,12, 13,14, 15,16,17]  # 左邊關鍵點（藍色）


    for result in results:
        disappear_count = disappear_count - 1
        take_flag_count = take_flag_count + 1
        put_flag_count = put_flag_count + 1
        


        if result.keypoints is None or len(result.keypoints.xy) == 0:
            #print("⚠️ 沒有偵測到人體，跳過該影格")
            #continue  # 沒偵測到人體，則跳過

            continue
        
        keypoints = result.keypoints.xy.cpu().numpy()  # 轉換成 NumPy 陣列

        # if len(keypoints) > 0:
            
        #     torso_keypoints = keypoints[0][:17]  # 取出與軀幹相關的關鍵點 (head, shoulders, hips)
        #     #print(torso_keypoints)
        #     zero_count = np.sum(torso_keypoints == 0)
        #     if len(torso_keypoints) == 0:  
        #         pass
        #     else:
        #         detected = True
        #         if consecutive_frames > 5:  #因為landmark detect也會誤報，所以要確信它有連續偵測到軀幹才算
        #             disappear_count = 2000  # 如果 torso_keypoints 有值，則將 disappear_count 設為 2000
            # if all(kp is not None and not np.isnan(kp).any() and kp.any() for kp in torso_keypoints):  # 確保所有關鍵點都有值
            #     #detected = True
            #     pass
            # else:
                
            #     disappear_count = 2000
        
        for person_kp in keypoints:
            if person_kp.shape[0] < 17:
                continue  # 確保關鍵點足夠

            # 畫關鍵點
            for kp_id, (x, y) in enumerate(person_kp):
                if np.isnan(x) or np.isnan(y) or (x == 0 and y == 0):

                    continue  # 跳過無效的關鍵點
                x, y = int(x), int(y)
                # if consecutive_frames >= 2:
                #     disappear_count = 2000

                # 根據關鍵點位置上顏色
                if kp_id in hand_side:
                    color = (0, 0, 255)  # 紅色
                elif kp_id in other_side:
                    color = (255, 0, 0)  # 藍色
                else:
                    color = (0, 255, 0)  # 綠色

                cv2.circle(frame, (x, y), 5, color, -1)  # 標記關鍵點
                cv2.putText(frame, str(kp_id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # if np.all(keypoints[0] == 0):
                #     print("✅ 陣列中的所有值都是 0")
                # else:
                #     print("❌ 陣列中至少有一個值不是 0")

                if kp_id == 9:
                    #左手腕
                    left_hand_x, left_hand_y = int(x), int(y)
                    total_keypoints[kp_id] = [left_hand_x,left_hand_y]
                    hand_keypoints[kp_id] = [left_hand_x,left_hand_y]

                    if left_hand_y > hand_determination:
                        foreign_objects_flag = True

                elif kp_id == 10:
                    #右手腕
                    right_hand_x, right_hand_y = int(x), int(y)
                    total_keypoints[kp_id] = [right_hand_x,right_hand_y]
                    hand_keypoints[kp_id] = [right_hand_x,right_hand_y]

                    if right_hand_y > hand_determination:
                        foreign_objects_flag = True


                elif kp_id == 11:
                    #左腳
                    left_feet_x, left_feet_y = int(x), int(y)
                    total_keypoints[kp_id] = [left_feet_x,left_feet_y]
                elif kp_id == 12:
                    #右腳
                    right_feet_x, right_feet_y = int(x), int(y)
                    total_keypoints[kp_id] = [right_feet_x,right_feet_y]
                elif kp_id == 5:
                    left_shoulder_x , left_shoulder_y = int(x) , int(y)
                    shoulder_keypoints[kp_id] = [left_shoulder_x,left_shoulder_y]
                elif kp_id == 6:
                    right_shoulder_x , right_shoulder_y = int(x) , int(y)
                    shoulder_keypoints[kp_id] = [right_shoulder_x,right_shoulder_y]
                elif kp_id == 0:
                    head_x , head_y = int(x) , int(y)
                    if head_y > hand_determination:
                        take_flag = False
                        put_flag = False
                        


            if len(hand_keypoints) >= 1:
                if (9 in hand_keypoints) & (10 in hand_keypoints):
                    hand_y = max(right_hand_y, left_hand_y)
                elif 9 in hand_keypoints :
                    hand_y = left_hand_y
                elif 10 in hand_keypoints :
                    hand_y = right_hand_y

                hands_last_location = hand_y

                detected = True
                if consecutive_frames > 1:
                    disappear_count = 2000

            if  len(total_keypoints) == 4:
                left_hand_x , left_hand_y = total_keypoints[9]
                right_hand_x , right_hand_y = total_keypoints[10]
                left_feet_x , left_feet_y = total_keypoints[11]
                right_feet_x , right_feet_y = total_keypoints[12]

                pointA = (left_hand_x, left_hand_y)
                pointB = (right_feet_x, right_feet_y)
                long = np.linalg.norm(np.array(pointA) - np.array(pointB))
                #print(long)
                #print(right_hand_x)
                #if (left_hand_x > left_feet_x > right_hand_x > right_feet_x) & (left_hand_y > right_hand_y > (left_feet_y and right_feet_y)) & (left_hand_x > 2850) :
                #if (left_hand_x > left_feet_x > right_hand_x > right_feet_x)  & (left_hand_x > 2850) :
                #if (left_hand_x  > right_hand_x )  & (left_hand_x > l1) & ((left_hand_x - right_feet_x) >l2 ):
                if (left_hand_x  > right_hand_x )  & (left_hand_x > l1) & (long > l2 ) & (left_hand_y < l5):
                    take_flag = True
                    take_flag_count = 0

                if right_hand_x < l3:
                    #print("hello")
                    put_flag = True
                    put_flag_count = 0

                  

            # 畫骨架連線
            for (start, end) in skeleton_pairs:
                if start >= len(person_kp) or end >= len(person_kp):
                    continue  # 避免索引超出範圍

                x1, y1 = int(person_kp[start][0]), int(person_kp[start][1])
                x2, y2 = int(person_kp[end][0]), int(person_kp[end][1])

                # **新增條件，確保兩點座標有效，且不為 (0,0)**
                if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                    continue
                if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                    continue

                cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # 橘色線段

     # 判斷是否連續兩個 frame 偵測到軀幹
    if detected:
        consecutive_frames += 1
    else:
        consecutive_frames = 0  # 如果當前 frame 沒偵測到，重置計數

    if take_flag_count > 13:
        take_flag = False
    if put_flag_count > 5:
        put_flag = False

    # if camera_index == "501011":
    #         print(f"take_flag_count : {take_flag_count} , put_flag_count : {put_flag_count} , disappear_count : {disappear_count} , take_flag : {take_flag} , put_flag : {put_flag}")
    #         #pass
    # if  hands_last_location:
    #     print(f"{camera_index} : {hands_last_location}")

    if hands_last_location:
        if (disappear_count == 1998) & take_flag & put_flag & (hands_last_location < hand_determination):
            adjusted_box = {
                        "model": model_name,  # 添加模型名稱
                        "compare_frame": ori_frame,
                                }
            detections.append(adjusted_box)
    
    # if take_flag_count > 150:
    #     take_flag = False
    # if put_flag_count > 50:
    #     put_flag = False
    # hand_determination = int((l6+l5)/2)
    # if hands_last_location:
    #     if (disappear_count == 1985) & take_flag & put_flag & (hands_last_location < hand_determination):

    #         adjusted_box = {
    #                     "model": model_name,  # 添加模型名稱
    #                     "compare_frame": ori_frame,
    #                             }
    #         detections.append(adjusted_box)
    
    # 將變數更新回 camera_states
    state["disappear_count"] = disappear_count
    state["take_flag"] = take_flag
    state["put_flag"] = put_flag
    state["take_flag_count"] = take_flag_count
    state["put_flag_count"] = put_flag_count
    state["consecutive_frames"] = consecutive_frames
    state["foreign_objects_flag"] = foreign_objects_flag
    state["hands_last_location"] = hands_last_location  #是否完成板料擺放其一的判斷依據


def send_alert(send_frame, camera_index, detections):
    """
    發送警報的函數，處理偵測到的事件並與 API 進行互動。
    detections 現在是一個包含字典的列表，每個字典包含 'xyxy', 'conf', 'cls'。
    """
    cooldown_updated_classes = set()  # 延後更新的冷卻清單

    current_time = time.time()
    cls_buffer_cooldown_dections = []
    #cls_buffer_cooldown = {}
    for box in detections:
        model_name = box["model"]
        if model_name == "imfrared_not_aligned" or model_name == "stick_hand_process_wrong":
            
            cls = int(box['cls'])  # 取得類別索引
        elif model_name == "model_pose" or model_name == "model_stick_hand":
            continue
        else:
            confidence = float(box['conf'][0])
            cls = int(box['cls'])  # 取得類別索引
        
        if model_name == "model_no_gloves":
            if cls == 0:
                cls = 3
            else:
                cls = 4
        # if model_name == "cellphone":
        #     cls = 4
        if model_name == "model_fall":
            cls = 2
        if model_name == "model_fire_smoke":
            if cls == 0:
                cls = 0
            else:
                cls = 1
        if model_name == "model_foreign_objects":
            cls = 6

        # 取得該類別的冷卻時間
        cooldown_time = class_alert_cooldowns.get(cls, 300)
        # 初始化該類別的警報時間
        if cls not in last_alert_times[str(camera_index)]:
            last_alert_times[str(camera_index)][cls] = 0

        if (current_time - last_alert_times[str(camera_index)][cls]) <= cooldown_time:
            continue

        if current_time - last_alert_times[str(camera_index)][cls] > cooldown_time:
            # last_alert_times[str(camera_index)][cls] = current_time
            # print(f"Sending alert for camera {camera_index} to API!")

            if model_name == "model_no_gloves":
                if cls == 3:
                    box["cls"] = 3
                    cls_buffer_cooldown_dections.append(box)
                    event_name_en = class_event_mapping_en.get(3, "Unknown Event")
            # if model_name == "cellphone":
            #     event_name_en = class_event_mapping_en.get(4, "Unknown Event")
            if model_name == "model_fall":
                box["cls"] = 2
                cls_buffer_cooldown_dections.append(box)
                event_name_en = class_event_mapping_en.get(2, "Unknown Event")
            if model_name == "model_fire_smoke":
                if cls == 0:
                    box["cls"] = 0
                    cls_buffer_cooldown_dections.append(box)
                    event_name_en = class_event_mapping_en.get(0, "Unknown Event")
                elif cls == 1:
                    box["cls"] = 1
                    cls_buffer_cooldown_dections.append(box)
                    event_name_en = class_event_mapping_en.get(1, "Unknown Event")
            if model_name == "imfrared_not_aligned":
                box["cls"] = 5
                cls_buffer_cooldown_dections.append(box)
            if model_name == "model_foreign_objects":
                box["cls"] = 6
                cls_buffer_cooldown_dections.append(box)
                event_name_en = class_event_mapping_en.get(6, "Unknown Event")
            if model_name == "stick_hand_process_wrong":
                box["cls"] = 8
                cls_buffer_cooldown_dections.append(box)

        if (model_name != "model_pose") and ( model_name != "imfrared_not_aligned") and ( model_name != "stick_hand_process_wrong"):
            if cls != 4:
                x1, y1, x2, y2 = box['xyxy']
                cv2.rectangle(send_frame, (x1, y1), (x2, y2), color_dict[cls], 10)
                label = f'{event_name_en} {confidence:.2f}'
                cv2.putText(send_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color_dict[cls], 10)
        # ✅ 延遲更新冷卻時間
        cooldown_updated_classes.add(cls)

    # ✅ 統一寫入冷卻時間
    for cls in cooldown_updated_classes:
        last_alert_times[str(camera_index)][cls] = current_time
    
    if cls_buffer_cooldown_dections:    
        for box in cls_buffer_cooldown_dections:
            model_name = box["model"]
            if model_name == "imfrared_not_aligned" or model_name == "stick_hand_process_wrong":
                
                cls = int(box['cls'])  # 取得類別索引
            elif model_name == "model_pose":
                continue
            else:
                confidence = float(box['conf'][0])
                cls = int(box['cls'])  # 取得類別索引
    
            if camera_index == "501003" or camera_index == "501004" or camera_index == "501007" or camera_index == "501008" or camera_index == "501011" or camera_index == "501012" or camera_index == "501015" or camera_index == "501016":
                location = "十二課疊合室"
            else:
                location = f"未知位置"

            # 格式化檔名為 "1-2024-12-12_17-53-11.jpg"
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            formatted_filename = f"{camera_index}-{timestamp}.jpg"
            
            if model_name == "imfrared_not_aligned":

                red_frame = box["detected_frame"]
                success = cv2.imwrite(f"images/{formatted_filename}", red_frame)
            elif model_name == "stick_hand_process_wrong":

                stick_hand_process_frame = box["process_wrong_frame"]
                success = cv2.imwrite(f"images/{formatted_filename}", stick_hand_process_frame)
            elif model_name == "model_pose":
                continue
            # 保存警報截圖
            else:
                success = cv2.imwrite(f"images/{formatted_filename}", send_frame)
            
            if success:
                print(f"Saved screenshot: {formatted_filename}")
            else:
                print(f"Failed to save screenshot: {formatted_filename}")

            # 準備 API 請求的數據（使用中文事件名稱）
            api_url = "https://eip.pcbut.com.tw/File/UploadYoloImage"
                    
            #join_cls = int(box['cls'][0])
            event_names = []


            if model_name == "model_no_gloves":
                event_name_cn = class_event_mapping_cn.get(3, "Unknown Event")
            # if model_name == "cellphone":
            #     event_name_cn = class_event_mapping_cn.get(4, "Unknown Event")
            if model_name == "model_fall":
                event_name_cn = class_event_mapping_cn.get(2, "Unknown Event")
            if model_name == "model_fire_smoke":
                if cls == 0:
                    event_name_cn = class_event_mapping_cn.get(0, "Unknown Event")
                else:
                    event_name_cn = class_event_mapping_cn.get(1, "Unknown Event")
                    #event_name = class_event_mapping_cn.get(int(box['cls'][0]), "未知事件")
            if model_name == "imfrared_not_aligned":
                event_name_cn = class_event_mapping_cn.get(5, "Unknown Event")
            if model_name == "model_foreign_objects":
                event_name_cn = class_event_mapping_cn.get(6, "Unknown Event")
            if model_name == "stick_hand_process_wrong":
                event_name_cn = class_event_mapping_cn.get(8, "Unknown Event")


            event_names.append(event_name_cn)
            print(event_names)
            formatted_event_name = "；".join(event_names)

            
            camera_model = {
                "cameraId": camera_index,
                "location": location,
                "eventName": formatted_event_name,
                "eventDate": time.strftime("%Y-%m-%d %H:%M:%S"),
                "notes": f"{len(detections)} events detected with confidence > {alert_threshold}",
                "fileName": formatted_filename,
                "result": f"疑似發生 {formatted_event_name}, 請同仁儘速查看"
            }

            # 發送包含影像和攝影機數據的 POST 請求
            # "D:/My Documents/vincent-shiu/桌面/ENIG/images/"+

            try:
                with open(f"images/{formatted_filename}", 'rb') as img_file:
                    files = {'files': (formatted_filename, img_file, 'image/jpeg')}
                    response = requests.post(api_url, files=files, data=camera_model, verify=False)

                if response.status_code == 200:
                    print(f"Successfully sent alert for camera {camera_index}. Response: {response.text}")

                else:
                    print(f"Failed to send alert for camera {camera_index}. Status Code: {response.status_code}, Response: {response.text}")
                            

            except Exception as e:
                print(f"Error sending alert for camera {camera_index}: {e}")
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
    # cap = cv2.VideoCapture("C:/Users/vincent-shiu/Web/RecordFiles/2025-03-12/10.80.21.136_01_20250312164150554_2.mp4")
    # output_video = "12_lam_demo.mp4"
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

    # take_flag = False
    # put_flag = False
    # take_flag_count = 0
    # put_flag_count = 0
    # disappear_count = 2000

    # consecutive_frames = 0

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