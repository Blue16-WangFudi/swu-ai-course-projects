from ultralytics import YOLO
import cv2
import numpy as np

# 关闭冗余日志，避免终端刷屏
model = YOLO("yolo11m.pt", verbose=False)
pose_model = YOLO("yolo11m-pose.pt", verbose=False)

source = "./game_area.jpg"

# 最小图片大小
MIN_CROP_SIZE = 32
# 最小平均亮度，用于过滤影子
MIN_AVG_V = 35
# 最小平均饱和度，作用相同
MIN_AVG_S = 20
# 最小关键点置信度
MIN_POSE_CONF = 0.3


# 得到球员的边界框
def get_person_boxes(img,source_img):
    results = model(img)
    #用于存储人物的边界框坐标
    person_boxes = []
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    for result in results:
        # result.save(filename="result-person.jpg")
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                box_hsv = hsv[y1:y2,x1:x2]
                avg_v = np.mean(box_hsv[:,:,2])
                avg_s = np.mean(box_hsv[:,:,1])
                if avg_v >= MIN_AVG_V and avg_s >= MIN_AVG_S:
                    person_boxes.append([x1,y1,x2,y2])
                    cv2.rectangle(source_img,[x1,y1],[x2,y2],(255,0,0),1)
                    if (x2-x1)<=MIN_CROP_SIZE or (y2-y1)<=MIN_CROP_SIZE:
                        print("pic too small!")
    # cv2.imwrite("result-person.jpg",source_img)

    return person_boxes

# 得到每个边界框里面的球员的关键点
def get_person_keypoints(img,source_img,write_dir, person_boxes=None, save_result=True):
    image = img.copy()
    # 对每个人物进行关键点检测
    if person_boxes is None:
        person_boxes = get_person_boxes(img,source_img)
    all_keypoints = []

    for i,bbox in enumerate(person_boxes):
        x1,y1,x2,y2 = bbox
        # 每个人在原图中的区域
        person_crop = image[y1:y2,x1:x2]

        pose_results = pose_model(person_crop)# 置信度设置为0.5

        player_keypoints = {} # 储存当前球员的关键点

        # 遍历每一个得到的关键点
        for pose in pose_results:

            if pose.keypoints is None or pose.keypoints.data.numel() == 0:
                # 未检测到关键点，跳过该人
                continue

            keypoints = pose.keypoints.data.cpu().numpy()
            if len(keypoints) >0 :
                keypoint_names = [
                    "鼻子", "左眼", "右眼", "左耳", "右耳",
                    "左肩", "右肩", "左肘", "右肘", "左手腕", "右手腕",
                    "左髋", "右髋", "左膝", "右膝", "左脚踝", "右脚踝"
                ]
                for idx,name in enumerate(keypoint_names):
                    if name in ["鼻子","左肩","右肩","左髋","右髋","左膝","右膝","左脚踝", "右脚踝"]:
                        local_x = keypoints[0][idx][0]
                        local_y = keypoints[0][idx][1]
                        conf = keypoints[0][idx][2]

                        if conf >= MIN_POSE_CONF and 0<=local_x<(x2-x1) and 0<= local_y<(y2-y1):
                            # 计算在原图的坐标
                            global_x = int(x1 + local_x)
                            global_y = int(y1 + local_y)
                            player_keypoints[name] = (global_x,global_y,conf)
                            cv2.circle(source_img,(global_x,global_y),3,(0,255,0),-1)


        all_keypoints.append({
            "player_id": i,
            "bbox": bbox,
            "keypoints": player_keypoints,
            "keypoint_count":len(player_keypoints)
        })

    if save_result:
        cv2.imwrite(write_dir+"/result_allperson_kp.jpg",source_img)
        print(f"共检测到 {len(all_keypoints)} 名球员，图片已保存")
    else:
        print(f"共检测到 {len(all_keypoints)} 名球员")

    return all_keypoints,image


