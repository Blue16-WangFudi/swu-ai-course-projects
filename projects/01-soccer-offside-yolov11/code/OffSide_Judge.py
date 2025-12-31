from Vanishing_Points import get_angle

def isOffSide(labeled_players_with_angles,offend_team_label,defend_team_label):

    ANGLE_THRESHOLD = 0.01  # 角度阈值，只有当进攻方角度明显大于防守方时才判定为越位
    
    offend_biggest_angle = 0
    defend_biggest_angle = 0
    

    offend_count = 0
    defend_count = 0
    
    for player in labeled_players_with_angles:
        angle = player.get('angle', 0)
        team_label = player.get('team_label', -999)
        
        if team_label == offend_team_label:
            offend_count += 1
            if angle > offend_biggest_angle:
                offend_biggest_angle = angle
        elif team_label == defend_team_label:
            defend_count += 1
            if angle > defend_biggest_angle:
                defend_biggest_angle = angle

    # 如果某队没有球员，不判定为越位
    if offend_count == 0 or defend_count == 0:
        return 0
    

    if defend_biggest_angle < offend_biggest_angle - ANGLE_THRESHOLD:
        return 1
    else:
        return 0


def Caculate_angle(labeled_players,ver_vanishing_points,side):
    vec_ref_vanishing_points = [ver_vanishing_points[0]-1,ver_vanishing_points[1]] if side=='right' \
        else [ver_vanishing_points[0]+1,ver_vanishing_points[1]]

    for player in labeled_players:

        if not player.get('keypoints', {}):  # 优先用 get() 避免 KeyError，空字典/None 都判定为 False
            if side=='right':
                player['most_point'] = [player['bbox'][2],player['bbox'][3]]
                player['angle'] = get_angle(vec_ref_vanishing_points,ver_vanishing_points,player['most_point'])
            else:
                player['most_point'] = [player['bbox'][0], player['bbox'][3]]
                player['angle'] = get_angle(vec_ref_vanishing_points, ver_vanishing_points,player['most_point'])
            continue

        right_most_angle = 0
        angle = None
        most_point = None
        if side == 'right':
            # max_x_item = max(player['keypoints'].items(), key=lambda item: item[1][0])# 取出x最大的关键点，即最右端关键点
            # rightmost_points = max_x_item[1][:2]
            # player['most_point'] = rightmost_points
            # angle = get_angle(vec_ref_vanishing_points,ver_vanishing_points,rightmost_points)

            for point in player['keypoints'].values():
                point = point[:2]
                angle = get_angle(vec_ref_vanishing_points,ver_vanishing_points,point)
                if angle>right_most_angle:
                    right_most_angle = angle
                    most_point = point
        else:
            # min_x_item = min(player['keypoints'].items(), key=lambda item: item[1][0])  # 取出x最小的关键点，即最左端关键点
            # leftmost_points = min_x_item[1][:2]
            # player['most_point'] = leftmost_points
            # angle = get_angle(vec_ref_vanishing_points,ver_vanishing_points,leftmost_points)

            for point in player['keypoints'].values():
                point = point[:2]
                angle = get_angle(vec_ref_vanishing_points,ver_vanishing_points,point)
                if angle>right_most_angle:
                    right_most_angle = angle
                    most_point = point

        player['angle'] = angle
        player['most_point'] = most_point


    for player in labeled_players:
        print(f"球员：{player['player_id']}，角度:{player['angle']}")
# if __name__ == '__main__':

def draw_off_side():
    return