import cv2
import numpy as np
import itertools
import random
from itertools import starmap
import math
import time


def get_vertical_line(image, side):  # 获取当前球员所在垂直位置线
    img = image.copy()
    selectedLines = []
    selectedLinesParams = []
    linesFound = False
    BlueRedMask = 100

    while linesFound == False:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 把图像转换为hsv图像
        mask = cv2.inRange(hsv, (35, BlueRedMask, BlueRedMask), (70, 255, 255))  # 从HSV颜色空间的图像中筛选出绿色区域
        # BlueRedMask 为色彩明度的下界 ，inRange输出一个二值化的图像，符合绿色的区域像素为255（白色),不符合则为0（黑色）
        imask = mask > 0  # 将mask中转化为同等维度的布尔数组，绿色区域位置为true
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]  # 选出图中绿色区域，滤除背景
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        v = np.median(green)  # 计算出green中所有元素的中位数
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(max(255, (1.0 + sigma) * v))
        cv2.imwrite('green.jpg', green)
        edges = cv2.Canny(green, 150, 250, apertureSize=3)
        minLineLenth = 1
        maxLineGap = 1250
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        """edges：输入图像，通常是经过 Canny 边缘检测得到的二值化边缘图像（边缘为白色，背景为黑色）。
            1：距离分辨率（以像素为单位），表示霍夫空间中极坐标 ρ（距离原点的距离）的步长。
            np.pi/180：角度分辨率（以弧度为单位），表示极坐标 θ（角度）的步长（此处为 1° 对应的弧度）。
            200：阈值，只有当累加器（记录直线候选点数量的容器）的值超过该阈值时，才被认为是一条有效的直线。
        """
        if lines.any():  # lines.any()检查数组中是否存在至少一个非零元素，即是否检测到直线
            if len(lines) > 2:
                linesFound = True
            else:
                BlueRedMask -= 10

    linesFound = False

    if side == 'left':
        angleMaxLimit = 70
        angleMinLimit = 20
    else:
        angleMaxLimit = 150
        angleMinLimit = 105

    print("vertical_lines_founded")

    rLimit = 300
    while linesFound == False:
        for line in lines:
            for r, theta in line:
                isLineValid = True
                cos = np.cos(theta)
                sin = np.sin(theta)

                if (theta * 180*7/22) > angleMinLimit and (theta * 180 *7/22) < angleMaxLimit:
                    x0 = cos * r
                    y0 = sin * r
                    x1 = int(x0 + 1000 * (-sin))    #其实不太理解为什么要转换为int型
                    y1 = int(y0 + 1000 * (cos))
                    x2 = int(x0 - 1000 * (-sin))
                    y2 = int(y0 - 1000 * (cos))

                    if len(selectedLines) > 0:
                        for lineParams in selectedLinesParams:
                            if abs(lineParams[0] - r) < rLimit: #过滤距离过近的直线
                                isLineValid = False
                        for selectedLine in selectedLines:
                            if not line_intersection(selectedLine, [[x1, y1], [x2, y2]]):  # 如果不与其他直线相交，就算不出交点
                                isLineValid = False
                        if [[x1, y1], [x2, y2]] in selectedLines or [[x2, y2], [x1, y1]] in selectedLines:  # 如果已经有了
                            isLineValid = False
                    if isLineValid:
                        selectedLines.append([[x1, y1], [x2, y2]])
                        print(x1, y1, x2, y2)
                        selectedLinesParams.append([r, theta])
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        cv2.putText(img, str((theta * 180 * 7 / 22)), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (200, 255, 155), 2, cv2.LINE_AA)
                        cv2.putText(img, str((theta * 180 * 7 / 22)), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (200, 255, 155), 2, cv2.LINE_AA)

            if len(selectedLines) < 2:  # 如果选择到的直线数量小于2，就要调整参数
                if rLimit >= 75:
                    rLimit -= 10
                else:
                    angleMinLimit -= 1
                    angleMaxLimit += 1
                    rLimit = 100
            else:
                linesFound = True
    cv2.imwrite("line_ver.jpg",img)
    print("inter_vertical_lines_founded")
    return selectedLines


def get_horizontal_lines(image):
    img = image.copy()
    height, width = img.shape[:2]
    selectedLines = []
    selectedLinesParams = []
    linesFound = False
    BlueRedMask = 100
    timeout = 8  # 超时时间（秒）
    start_time = time.time()  # 记录循环启动时间

    while linesFound == False:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(f"循环超时！已运行 {elapsed_time:.2f} 秒，未找到直线")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (36, BlueRedMask, BlueRedMask), (100, 255, 255))
        imask = mask > 0;
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]
        gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)  # gray似乎不会用到
        edges = cv2.Canny(gray, 150, 250, apertureSize=3)
        minLineLength = 1
        maxLineGap = 1250
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines.any():
            if len(lines) > 2:
                linesFound = True
            else:
                BlueRedMask -= 10

    linesFound = False
    max_angle_btw = 15# 直线与画面x轴的夹角
    min_angle_btw = 3  # 直线与画面x轴的夹角
    rLimit = 100
    print("horizontal_lines_founded")
    start_time = time.time()  # 记录循环启动时间
    while linesFound == False:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(f"循环超时！已运行 {elapsed_time:.2f} 秒，未找到符合要求的直线，请手动输入朝向")
        for line in lines:
            for r, theta in line:
                isLineValid = True
                cos = np.cos(theta)
                sin = np.sin(theta)
                angle_btw = abs(theta - np.pi / 2)*180*7/22
                # print('============直线的夹角是=================')
                # print(angle_btw)
                if min_angle_btw<angle_btw<max_angle_btw:
                    x0 = cos * r
                    y0 = sin * r
                    x1 = int(x0 - 1000 * sin)
                    y1 = int(y0 + 1000 * cos)
                    x2 = int(x0 + 1000 * sin)
                    y2 = int(y0 - 1000 * cos)
                    edge_points = get_line_boundary_intersection(r,theta,[height,width])
                    x1_e = edge_points[0][0]
                    y1_e = edge_points[0][1]
                    x2_e = edge_points[1][0]
                    y2_e = edge_points[1][1]
                    if len(selectedLines) > 0:
                        for lineParams in selectedLinesParams:
                            if abs(lineParams[0] - r) < rLimit:
                                isLineValid = False
                            for selectedLine in selectedLines:
                                if not line_intersection(selectedLine, [[x1, y1], [x2, y2]]):  # 判断是否相交
                                    isLineValid = False
                            if [[x1, y1], [x2, y2]] in selectedLines or [[x2, y2], [x1, y1]] in selectedLines:#判断是否重复
                                isLineValid = False
                    if isLineValid:
                        print([[x1, y1], [x2, y2]])
                        selectedLines.append([[x1_e, y1_e], [x2_e, y2_e]])
                        selectedLinesParams.append([r, theta])
                        cv2.line(img, (x1_e, y1_e), (x2_e, y2_e), (0, 0, 255), 1)
                        cv2.putText(img, str((theta * 180 * 7 / 22)), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (200, 255, 155), 2, cv2.LINE_AA)
                        cv2.putText(img, str((theta * 180 * 7 / 22)), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (200, 255, 155), 2, cv2.LINE_AA)
        if len(selectedLines) < 2:  # 至少找到两条线
            if rLimit >= 75 and rLimit>=5: #减少距离的要求
                rLimit -= 10
            else:
                if max_angle_btw<=15:
                    max_angle_btw += 1
        else:
            linesFound = True

    cv2.imwrite("line_hor.jpg", img)
    print("inter_horizontal_lines_founded")
    print("最大夹角"+ str(max_angle_btw))
    return selectedLines





def get_line_boundary_intersection(rho, theta, img_shape):
    """
    计算极坐标直线(ρ,θ)与图像边界的真实交点
    """
    height, width = img_shape
    intersections = []

    # 步骤1：将极坐标直线转为笛卡尔坐标方程
    # 极坐标公式：x*cosθ + y*sinθ = ρ
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # 情况1：直线垂直（sinθ≈0，θ≈0或π）→ x = ρ/cosθ
    if np.isclose(sin_theta, 0):
        x = rho / cos_theta
        if 0 <= x <= width - 1:
            # 与上边界(y=0)、下边界(y=height-1)的交点
            intersections.append((x, 0))
            intersections.append((x, height - 1))
    # 情况2：直线水平（cosθ≈0，θ≈π/2）→ y = ρ/sinθ
    elif np.isclose(cos_theta, 0):
        y = rho / sin_theta
        if 0 <= y <= height - 1:
            # 与左边界(x=0)、右边界(x=width-1)的交点
            intersections.append((0, y))
            intersections.append((width - 1, y))
    # 情况3：斜直线（y = kx + b）
    else:
        k = -cos_theta / sin_theta  # 斜率
        b = rho / sin_theta  # 截距

        # 求与左边界(x=0)的交点
        y_left = k * 0 + b
        if 0 <= y_left <= height - 1:
            intersections.append((0, y_left))
        # 求与右边界(x=width-1)的交点
        y_right = k * (width - 1) + b
        if 0 <= y_right <= height - 1:
            intersections.append((width - 1, y_right))
        # 求与上边界(y=0)的交点
        x_top = (0 - b) / k
        if 0 <= x_top <= width - 1:
            intersections.append((x_top, 0))
        # 求与下边界(y=height-1)的交点
        x_bottom = (height - 1 - b) / k
        if 0 <= x_bottom <= width - 1:
            intersections.append((x_bottom, height - 1))

    # 步骤2：去重并筛选有效交点（保留2个端点）
    # 浮点坐标转整数（像素坐标为整数）
    intersections = [(int(round(x)), int(round(y))) for x, y in intersections]
    # 去重（避免重复交点）
    intersections = list(set(intersections))

    # 步骤3：取两端点（最多2个有效交点）
    if len(intersections) >= 2:
        # 按x坐标排序（或y坐标，确保是两端点）
        intersections.sort()
        return intersections[0], intersections[-1]
    elif len(intersections) == 1:
        # 仅1个交点（直线与图像相切），补充一个延长点（沿直线方向）
        x0, y0 = intersections[0]
        # 沿直线方向延长100像素
        dx = -sin_theta * 100
        dy = cos_theta * 100
        x1 = max(0, min(int(x0 + dx), width - 1))
        y1 = max(0, min(int(y0 + dy), height - 1))
        return (x0, y0), (x1, y1)
    else:
        # 无有效交点（直线在图像外）
        return None


def det(a, b):  # 计算行列式
    return a[0] * b[1] - a[1] * b[0]
def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(x_diff, y_diff)
    if div == 0:
        return None
    # 使用矩阵求解交点
    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return [x, y]

# def line_intersection(line1, line2, parallel_threshold=1e-6):
#     """
#     求解两条线段的交点（兼容角度相近的近乎平行直线）
#     :param line1: 线段1，格式[(x1,y1), (x2,y2)]
#     :param line2: 线段2，格式[(x1,y1), (x2,y2)]
#     :param parallel_threshold: 平行判定阈值（越小越严格，建议1e-6~1e-4）
#     :return: 交点坐标[x,y]（落在两条线段范围内），无则返回None
#     """
#     # 提取直线端点
#     (x1, y1), (x2, y2) = line1
#     (x3, y3), (x4, y4) = line2
#
#     # 计算方向向量的差分
#     x_diff = (x1 - x2, x3 - x4)
#     y_diff = (y1 - y2, y3 - y4)
#
#     # 计算行列式（叉积），处理浮点精度
#     div = det(x_diff, y_diff)
#     # 修正1：用阈值判定平行（而非绝对等于0）
#     if abs(div) < parallel_threshold:
#         # 角度极近（近乎平行），尝试计算“近似交点”（可选）
#         # 若不需要近似交点，直接返回None；若需要，可计算投影最近点
#         img = cv2.imread('./images/test4.jpg')
#         inter = near_parallel_intersection(line1,line2)
#         cv2.line(img, line1[0], line1[1], (0, 0, 255), 1)
#         cv2.line(img,line2[0],line2[1],(0, 0, 255), 1)
#         cv2.namedWindow("Near_parell", cv2.WINDOW_NORMAL)
#         cv2.imshow("Near_parell", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         print('角度极近（近乎平行），尝试计算“近似交点”')
#         print(inter)
#         return inter
#
#     # 克莱姆法则求解无限长直线的交点
#     d1 = x1 * y2 - y1 * x2  # det(line1)
#     d2 = x3 * y4 - y3 * x4  # det(line2)
#     x = det((d1, d2), x_diff) / div
#     y = det((d1, d2), y_diff) / div
#
#
#     return [round(x, 4), round(y, 4)]  # 四舍五入减少浮点噪声


# 扩展：近乎平行直线的“近似交点”求解（可选）
def near_parallel_intersection(line1, line2, angle_threshold=1):
    """
    求解角度相近（≤angle_threshold度）直线的近似交点（投影最近点）
    :param angle_threshold: 角度阈值（度），超过则返回None
    :return: 近似交点坐标
    """

    # 计算两条直线的夹角
    def line_angle(line):
        """计算直线与X轴的夹角（弧度）"""
        (x1, y1), (x2, y2) = line
        return np.arctan2(y2 - y1, x2 - x1)

    angle1 = line_angle(line1)
    angle2 = line_angle(line2)
    angle_diff = abs(np.degrees(angle1 - angle2))
    angle_diff = min(angle_diff, 180 - angle_diff)  # 归一化到0~90度

    if angle_diff > angle_threshold:
        return None  # 角度差超过阈值，用普通交点函数

    # 计算两条线段的中点连线的中点，作为近似交点
    mid1 = [(line1[0][0] + line1[1][0]) / 2, (line1[0][1] + line1[1][1]) / 2]
    mid2 = [(line2[0][0] + line2[1][0]) / 2, (line2[0][1] + line2[1][1]) / 2]
    near_x = (mid1[0] + mid2[0]) / 2
    near_y = (mid1[1] + mid2[1]) / 2
    return [near_x, near_y]


def find_intersections(lines):  # 找到一组直线两两之间的交点
    intersections = []
    for i, line_1 in enumerate(lines):
        for line_2 in lines[i + 1:]:
            if not line_1 == line_2:
                intersection = line_intersection(line_1, line_2)
                if intersection:
                    intersections.append(intersection)
    return intersections


def get_vertical_vanishing_point(img, side):
    selectedLines = get_vertical_line(img, side)
    intersectionPoints = find_intersections(selectedLines)
    vanishingPointX = 0.0
    vanishingPointY = 0.0
    for point in intersectionPoints:
        vanishingPointX += point[0]
        vanishingPointY += point[1]
    # 求平均值计算
    return (round(vanishingPointX / len(intersectionPoints)), round(vanishingPointY / len(intersectionPoints)))


def get_horizontal_vanishing_point(img):
    selectedLines = get_horizontal_lines(img)
    intersectionPoints = find_intersections(selectedLines)
    vanishingPointX = 0.0
    vanishingPointY = 0.0
    for point in intersectionPoints:
        vanishingPointX += point[0]
        vanishingPointY += point[1]
    return (round(vanishingPointX / len(intersectionPoints)), round(vanishingPointY / len(intersectionPoints)))


def get_angle_x(vanishing_point, test_point, goalDirection):  # 获得消失点向量与x轴的夹角
    reference_point = 0.0, vanishing_point[1]
    ref_vec = np.array(reference_point)
    vanish_vec = np.array(vanishing_point)
    test_vec = np.array(test_point)  # 目标点与消失点连线向量
    x_axis_vec = ref_vec - vanish_vec
    angle_vec = test_vec - vanish_vec
    cos_angle = np.dot(x_axis_vec, angle_vec) / (np.linalg.norm(x_axis_vec) * np.linalg.norm(angle_vec))
    angle = np.degrees(np.arccos(cos_angle))
    if goalDirection == ('left'):
        if reference_point[0] > vanishing_point[0]:
            angle = -angle
    if goalDirection == 'right':
        if reference_point[0] < vanishing_point[0]:
            angle = -angle

    return angle


def get_angle(a, b, c):  # 获取角度
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cos_angle))

    return angle


def get_round_plane(img, vertical_vanishing_point, horizontal_vanishing_point, side):
    allVerticalLines = get_vertical_line(img, side)
    allHorizontalLines = get_horizontal_lines(img)
    # 找出地面的一个交点
    for i in range(len(allVerticalLines)):
        for j in range(len(allHorizontalLines)):
            selectedLines = [allVerticalLines[i], allHorizontalLines[j]]
            if find_intersections(selectedLines):
                break
        if find_intersections(selectedLines):
            break
    print(selectedLines)
    intersections = find_intersections(selectedLines)
    thirdPoint = intersections[0]
    x1 = vertical_vanishing_point[0]
    y1 = vertical_vanishing_point[1]
    z1 = 0
    x2 = horizontal_vanishing_point[0]
    y2 = horizontal_vanishing_point[1]
    z2 = 0
    x3 = thirdPoint[0]
    y3 = thirdPoint[1]
    z3 = 0
    print("A", x2, "B", x1, "C")
    # 计算地面的平面表达式
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (-a * x1 - b * y1 - c * z1)

    return [a, b, c, d]


# 画出消失点,在计算完对员与消失点的参考角度之后才会使用到
def draw_vanishing_points(img,vanishing_points,labeled_players,write_dir):
    h,w = img.shape[:2]
    point_radius = 20
    point_color = (0,0,255)

    ext_left = 0 if vanishing_points[0] > 0 else -vanishing_points[0] + point_radius
    ext_right = 0 if vanishing_points[0] + point_radius < w else vanishing_points[0] + point_radius - w
    ext_top = 0 if vanishing_points[1]>0 else - vanishing_points[1] + point_radius
    ext_bottom = 0 if vanishing_points[1]+ point_radius < h else vanishing_points[1] + point_radius - h

    new_h = h + ext_top + ext_bottom
    new_w = w + ext_left + ext_right

    ext_img = np.ones((new_h,new_w,3),dtype=np.uint8)*255
    ext_img[ext_top:ext_top+h,ext_left:ext_left+w,:] = img

    vx = ext_left + vanishing_points[0]
    vy = ext_top + vanishing_points[1]

    cv2.circle(ext_img,(vx,vy),point_radius,point_color,-1)
    cv2.putText(ext_img,f"Vanishing Points:{vanishing_points[0],vanishing_points[1]}",(vx+10,vy-10),cv2.FONT_HERSHEY_SIMPLEX, 1, point_color, 2)

    for player in labeled_players:
        point = player['most_point']
        print(point)
        player_most_x = point[0]+ext_left
        player_most_y = point[1]+ext_top
        if player['team_label'] == 0:
            color = (0,0,255)
        elif player['team_label'] == 1:
            color = (255,0,0)
        else:
            color = (0,255,0)
        cv2.line(ext_img, [vx, vy], [player_most_x, player_most_y], thickness=1, lineType=cv2.LINE_AA,color=color)

    if cv2.imwrite(write_dir+"\\extended_image.jpg",ext_img):
        print('连线图已保存')
    else:
        print('保存失败')

    print(write_dir+"\\extended_image.jpg")
    return ext_img


def project_point_on_plane(plane, point, pose, ratio, image):
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]
    if 'rightShoulder' in pose[2].keys() and 'rightHip' in pose[2].keys() and 'leftShoulder' in pose[
        2].keys() and 'leftHip' in pose[2].keys():
        rightUpperBody = (math.sqrt((pose[2]['rightShoulder']['y'] - pose[2]['rightHip']['y']) ** 2 + (
                pose[2]['rightShoulder']['x'] - pose[2]['rightHip']['x']) ** 2))
        leftUpperBody = (math.sqrt((pose[2]['leftShoulder']['y'] - pose[2]['leftHip']['y']) ** 2 + (
                pose[2]['leftShoulder']['x'] - pose[2]['leftHip']['x']) ** 2))
        approx_z = ratio * ((rightUpperBody + leftUpperBody) / 2)
        temp_pt = [pose[2]['leftHip']['y'], 0.0]
        angle = get_angle([pose[2]['leftShoulder']['y'], pose[2]['leftShoulder']['x']],
                          [pose[2]['leftHip']['y'], pose[2]['leftHip']['x']], temp_pt)
    elif 'rightShoulder' in pose[2].keys() and 'rightHip' in pose[2].keys():
        if 'leftShoulder' not in pose[2].keys() or 'leftHip' not in pose[2].keys():
            approx_z = ratio * (math.sqrt((pose[2]['rightShoulder']['y'] - pose[2]['rightHip']['y']) ** 2 + (
                    pose[2]['rightShoulder']['x'] - pose[2]['rightHip']['x']) ** 2))
            temp_pt = [pose[2]['rightHip']['y'], 0.0]
            angle = get_angle([pose[2]['rightShoulder']['y'], pose[2]['rightShoulder']['x']],
                              [pose[2]['rightHip']['y'], pose[2]['rightHip']['x']], temp_pt)
    elif 'leftShoulder' in pose[2].keys() and 'leftHip' in pose[2].keys():
        if 'rightShoulder' not in pose[2].keys() or 'rightHip' not in pose[2].keys():
            approx_z = ratio * (math.sqrt((pose[2]['leftShoulder']['y'] - pose[2]['leftHip']['y']) ** 2 + (
                    pose[2]['leftShoulder']['x'] - pose[2]['leftHip']['x']) ** 2))
            temp_pt = [pose[2]['leftHip']['y'], 0.0]
            angle = get_angle([pose[2]['leftShoulder']['y'], pose[2]['leftShoulder']['x']],
                              [pose[2]['leftHip']['y'], pose[2]['leftHip']['x']], temp_pt)

    angle = (angle * 22) / (7 * 180)
    final_z = np.cos(angle) * approx_z
    x1 = point[0]
    y1 = point[1]
    z1 = final_z
    d = abs((a * x1 + b * y1 + c * z1 + d))
    e = (math.sqrt(a * a + b * b + c * c))
    perpendicular_dist = d / e
    y_new = y1 + perpendicular_dist

    return [int(y_new), int(point[0])]


if __name__ == '__main__':
    game_area = cv2.imread('game_area.jpg')
    source_img = cv2.imread('images/OG.jpg')
    #get_vertical_line(game_area,'right')
    #get_horizontal_lines(game_area,'right')

    #print(f"horizontal VanishingPoint is{get_horizontal_vanishing_point(game_area,'right')} ")
    #print(f"vertical VanishingPoint is{get_vertical_vanishing_point(game_area, 'right')} ")
    draw_vanishing_points(source_img, get_vertical_vanishing_point(game_area, 'right'))