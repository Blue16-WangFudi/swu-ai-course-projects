import cv2
from pathlib import Path
from ultralytics import YOLO
from GetField import *
from PersonAndPose import *
from Vanishing_Points import *
from Play_Team_Cluster import cluster_players_by_jersey
from OffSide_Judge import *
source = "./images/test4.jpg"
source_img = cv2.imread(source)

# 提取无后缀的文件名
pic_name = Path(source).stem  # stem 属性直接返回无后缀的文件名
target_dir = "./result_img"
new_path = Path(target_dir) / pic_name  # 用 / 拼接路径（自动适配系统分隔符）
new_path.mkdir(parents=True, exist_ok=True)
print(f"目录已创建/存在：{new_path.absolute()}")  # 输出绝对路径，验证创建结果
write_direction = str(new_path)
print(write_direction+'\\team_classification_result.jpg')

# 获取比赛区域
game_area = get_field(source_img,write_direction)
# 获取人物边界框与关键点
person_boxes = get_person_boxes(game_area,source_img)
person_keypoints,image = get_person_keypoints(game_area,source_img,write_direction, person_boxes=person_boxes)

# 进行队伍聚类
labeled_players,team_vis_img = cluster_players_by_jersey(person_keypoints,image,write_direction)

#计算水平消失点
try:
    hor_vanishing_points = get_horizontal_vanishing_point(game_area)
    side = 'right' if hor_vanishing_points[0]>0 else 'left'
    print('朝向计算成功')

except TimeoutError as e:
    print('消失点计算错误')
    side = input('请手动输入朝向:')

print('镜头朝向是: '+ side)
# 计算垂直消失点
ver_vanishing_points = get_vertical_vanishing_point(game_area,side)


# 进行越位判断
Caculate_angle(labeled_players,ver_vanishing_points,side)
# 画出消失点
draw_vanishing_points(source_img,ver_vanishing_points,labeled_players,write_direction)


off_team_label = int(input('请输入进攻方队伍的队伍编号：'))
def_team_label = int(input('请输入防守方队伍的队伍编号：'))
if isOffSide(labeled_players,off_team_label,def_team_label):
    print('检测到越位')
else:
    print('未检测到越位')

