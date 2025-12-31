import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def extract_jersey_feature(player_all_keypoints,img):
    keypoints = player_all_keypoints["keypoints"]
    bbox = player_all_keypoints["bbox"]
    x1,y1,x2,y2 = bbox

    key_combinations = [
        ["左肩","右肩","左髋","右髋"],
        ["边界框"],
    ]

    selected_comb = None
    valid_kps = {}

    for comb in key_combinations:
        if comb[0]=="边界框":
            selected_comb =comb
            break
        has_all = all(kp_name in keypoints for kp_name in comb)
        if has_all:
            selected_comb = comb
            valid_kps = {kp_name:keypoints[kp_name][:2] for kp_name in comb}
            break

    if selected_comb[0] == "边界框":
        jersey_x1 = x1
        jersey_x2 = x2
        jersey_y1 = y1 + int((y2-y1)*0.1) #取上10%
        jersey_y2 = y1 + int((y2-y1)*0.4) #取上40%
    elif selected_comb == ["左肩", "右肩", "左髋", "右髋"]:
        #
        left_shoulder = valid_kps["左肩"]
        right_shoulder = valid_kps["右肩"]
        left_hip = valid_kps["左髋"]
        right_hip = valid_kps["右髋"]
        jersey_x1 = min(left_shoulder[0], right_shoulder[0]) - 5
        jersey_x2 = max(left_shoulder[0], right_shoulder[0]) + 5
        jersey_y1 = min(left_shoulder[1], right_shoulder[1]) - 1
        jersey_y2 = max(left_hip[1], right_hip[1]) + 1
    # 避免越界
    jersey_x1 = max(0, jersey_x1)
    jersey_x2 = min(img.shape[1] - 1, jersey_x2)
    jersey_y1 = max(0, jersey_y1)
    jersey_y2 = min(img.shape[0] - 1, jersey_y2)


    jersey_roi = img[jersey_y1:jersey_y2, jersey_x1:jersey_x2].copy()

    color_feat = [
        np.mean(jersey_roi[:, :, 0]),
        np.mean(jersey_roi[:, :, 1]),
        np.mean(jersey_roi[:, :, 2])
    ]
    return color_feat


def cluster_players_by_jersey(all_player_data,img,write_dir,save_result=True,reference_centers=None):
    """
    队伍聚类，支持跨帧一致性跟踪
    Args:
        all_player_data: 球员数据列表
        img: 图像
        write_dir: 输出目录
        save_result: 是否保存结果
        reference_centers: 参考帧的聚类中心（用于跨帧一致性），格式：{label: [h_mean, s_mean, v_mean]}
    Returns:
        labeled_players: 标注后的球员列表
        vis_img: 可视化图像
        cluster_centers: 当前帧的聚类中心（用于下一帧参考）
    """
    player_features = []
    valid_players = []
    invalid_player_ids = []

    for player in all_player_data:
        feat = extract_jersey_feature(player,img)
        if feat is not None:
            try:
                # 将列表转换为numpy数组以便检查
                feat_array = np.array(feat)
                # 检查是否包含NaN或inf
                if not (np.isnan(feat_array).any() or np.isinf(feat_array).any()):
                    # 检查特征是否在合理范围内（RGB值0-255）
                    if all(0 <= val <= 255 for val in feat):
                        player_features.append(feat)
                        valid_players.append(player)
                        continue
            except:
                pass
        else:
            invalid_player_ids.append(player["player_id"])

    if len(valid_players) < 2:
        print(f"有效球员数量不足（仅{len(valid_players)}名），无法聚类")
        return all_player_data, img.copy(), None

    # 进行标准化
    scaler = StandardScaler()
    player_features_scaled = scaler.fit_transform(player_features)

    # DBSCAN聚类
    dbscan = DBSCAN(eps=1.0, min_samples=2)
    team_labels = dbscan.fit_predict(player_features_scaled)
    # DBSCAN中label=-1表示噪声

    # 计算每个聚类的中心
    cluster_centers = {}
    unique_labels = set(team_labels)
    for label in unique_labels:
        if label == -1:  # 跳过噪声点
            continue
        label_indices = [i for i, l in enumerate(team_labels) if l == label]
        if len(label_indices) > 0:
            cluster_features = [player_features[i] for i in label_indices]
            cluster_centers[label] = np.mean(cluster_features, axis=0).tolist()

    # 如果有参考中心，进行标签映射以保持一致性
    if reference_centers is not None and len(cluster_centers) >= 2:
        # 计算当前聚类中心与参考中心的相似度
        label_mapping = {}
        used_ref_labels = set()
        
        # 对每个当前聚类，找到最相似的参考聚类
        for curr_label, curr_center in cluster_centers.items():
            best_match = None
            best_distance = float('inf')
            
            for ref_label, ref_center in reference_centers.items():
                if ref_label in used_ref_labels:
                    continue
                # 计算特征的距离
                distance = np.linalg.norm(np.array(curr_center) - np.array(ref_center))
                if distance < best_distance:
                    best_distance = distance
                    best_match = ref_label
            
            if best_match is not None and best_distance < 50:  # 阈值
                label_mapping[curr_label] = best_match
                used_ref_labels.add(best_match)
            else:
                # 如果没有找到匹配，保持原标签
                label_mapping[curr_label] = curr_label
        
        # 应用标签映射
        remapped_labels = []
        for label in team_labels:
            if label == -1:  # 噪声保持为-1
                remapped_labels.append(-1)
            elif label in label_mapping:
                remapped_labels.append(label_mapping[label])
            else:
                remapped_labels.append(label)
        team_labels = remapped_labels
        
        # 更新聚类中心标签
        new_cluster_centers = {}
        for old_label, new_label in label_mapping.items():
            new_cluster_centers[new_label] = cluster_centers[old_label]
        cluster_centers = new_cluster_centers
    else:
        # 第一帧或没有参考，将DBSCAN的标签映射到0,1
        # DBSCAN可能产生0,1,2...等标签，我们需要映射到0和1
        unique_valid_labels = [l for l in unique_labels if l != -1]
        if len(unique_valid_labels) >= 2:
            # 按聚类大小排序，最大的两个映射到0和1
            label_sizes = {l: sum(1 for x in team_labels if x == l) for l in unique_valid_labels}
            sorted_labels = sorted(label_sizes.items(), key=lambda x: x[1], reverse=True)
            
            label_mapping = {}
            for idx, (old_label, _) in enumerate(sorted_labels[:2]):
                label_mapping[old_label] = idx
            
            # 其他标签保持原样或映射为-2
            remapped_labels = []
            for label in team_labels:
                if label == -1:
                    remapped_labels.append(-1)
                elif label in label_mapping:
                    remapped_labels.append(label_mapping[label])
                else:
                    remapped_labels.append(-2)  # 未分类
            team_labels = remapped_labels
            
            # 更新聚类中心
            new_cluster_centers = {}
            for old_label, new_label in label_mapping.items():
                if old_label in cluster_centers:
                    new_cluster_centers[new_label] = cluster_centers[old_label]
            cluster_centers = new_cluster_centers

    labeled_players = []
    for idx,(player,label) in enumerate(zip(valid_players,team_labels)):
        player["team_label"] = int(label)
        player["cluster_confidence"] = 1
        labeled_players.append(player)

    # 处理分类失败的球员
    for player in all_player_data:
        if player["player_id"] in invalid_player_ids:
            player["team_label"] = -2  # -2表示未分类
            player["cluster_confidence"] = 0.0
            labeled_players.append(player)

    vis_img = img.copy()

    team_colors = [
        (255, 0, 0),    # 队伍0：蓝色 (BGR格式)
        (0, 0, 255),    # 队伍1：红色 (BGR格式)
        (0, 255, 0),    # DBSCAN噪声/裁判：绿色
        (128, 128, 128) # 未分类：灰色
    ]

    for player in labeled_players:
        x1, y1, x2, y2 = player["bbox"]
        label = player["team_label"]
        player_id = player["player_id"]
        confidence = player["cluster_confidence"]

        # 选择颜色
        if label == -2:
            color = team_colors[3]
            label_text = f"Unclassified"
        elif label == -1:  # DBSCAN噪声
            color = team_colors[2]
            label_text = f"Ref/Staff"
        else:
            color = team_colors[label % len(team_colors)]
            label_text = f"Team {label} (conf:{confidence:.2f})"

        # 绘制边界框和标签
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        # 标签背景（避免文字与图像重叠）
        cv2.rectangle(vis_img, (x1, y1-20), (x1 + len(label_text)*8, y1), color, -1)
        cv2.putText(vis_img, label_text, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(vis_img, (x1, y2), (x1 + 8 * 8, y2+20), color, -1)
        cv2.putText(vis_img, f"player:{player_id}", (x1,y2+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
    if save_result:
        cv2.imwrite(write_dir+'/team_classification_result.jpg', vis_img)

    return labeled_players, vis_img, cluster_centers