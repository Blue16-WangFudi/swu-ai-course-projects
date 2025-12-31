import cv2
import numpy as np
import matplotlib.pyplot as plt

#转化为hsv图像

#hsv图像中绿色的范围
lower_green = np.array([36,0,0])
upper_green = np.array([86,255,255])
#蓝色范围
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
#红色范围
lower_red = np.array([0,31,255])
upper_red = np.array([176,255,255])
#白色范围
lower_white = np.array([0,0,0])
upper_white = np.array([0,0,255])
#形态学操作核，用于后续图像处理中去除噪声或连接前景区域
stride_kernel = (25,25)
stride_aud = (30,30)
stride_close = (150,150)

#非极大值抑制
def non_max(boxes,scores,iou_num):

    scores_sort = scores.argsort().tolist() #对分数从小到大排序，得到索引列表
    keep = []
    while(len(scores_sort)):
        index = scores_sort.pop()
        keep.append(index)
        if(len(scores_sort)==0):
            break
        iou_res = []
        for i in scores_sort:
            iou_res.append(iou(boxes[index],boxes[i]))
        iou_res = np.array(iou_res)
        filtered_indexes = set((iou_res>iou_num).nonzero()[0]) #nonzero() 用于获取布尔数组中值为 True 的元素的索引位
        scores_sort = [v for (i,v) in enumerate(scores_sort) if i not in filtered_indexes] #去除掉与box[index]重叠的边框，留下不重叠的
    final = []
    for i in keep:
        final.append(boxes[i])
    return np.array(final)



def iou(box1,box2):#计算两个矩形框的交并比

    x1 = max(box1[0],box2[0])
    x2 = min(box1[2],box2[2])
    y1 = max(box1[1],box2[1])
    y2 = min(box1[3],box2[3])
    if x1>x2 or y1>y2:
        return -2
    inter = (x2-x1)*(y2-y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    fin_area = area1 + area2 - inter
    iou = inter/fin_area

    return iou


def get_contours(image):

    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lower_green,upper_green)
    res = cv2.bitwise_and(image,image,mask=mask) #过滤掉非绿色区域，绿色区域变为白色
    res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
    res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    mask = 255 - mask
    # 定义形态学操作的卷积核（用于去除噪声和增强轮廓）
    kernel = np.ones(stride_kernel,np.uint8)
    kernel_aud = np.ones(stride_aud,np.uint8)
    kernel_close = np.ones(stride_close,np.uint8)
    # 对掩码进行形态学闭操作（填充小空洞，连接断开的区域）
    thresh_aud = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_aud)
    # 对掩码进行另一轮闭操作（针对球员区域的精细处理）
    thresh_players = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 对thresh_aud进行开操作（去除小噪声区域，保留大面积区域）
    thresh_aud = cv2.morphologyEx(thresh_aud, cv2.MORPH_OPEN, kernel_close)

    thresh_aud = 255- thresh_aud
    thresh = cv2.bitwise_and(thresh_aud, thresh_players)
    thresh_aud = cv2.bitwise_and(thresh_aud,thresh_players)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours,thresh_aud

def get_boxes(image,contours):

    im1 = image.shape[0] # 获取图像高度
    im = image.shape[1]  # 获取图像宽度
    maxx = 0# 记录最大轮廓的面积
    needed = contours[0]
    boxes = []#存储筛选后的边界框
    scores = []#存储每个边界框对应的分数
    for c in contours: #
        temp = 1
        clr = (255, 0, 0) #蓝色
        x, y, w, h = cv2.boundingRect(c)#计算轮廓c的最小外接矩形
        arr_cont = w * h # 计算矩形面积
        if arr_cont > maxx:
            maxx = arr_cont # 更新最大面积
            needed = c  # 更新最大轮廓
        if h < image.shape[0] * 0.01 or w < image.shape[1] * 0.01:#过滤面积过小的轮廓，通常是噪声
            clr = (0, 255, 0) #标记为绿色
        if w > h: # 球员在图像中通常是 “高> 宽” 的直立形态，宽大于高的轮廓可能是横向的噪声
            clr = (0, 255, 0)
        #扩展边界框
        w = int(w + 0.5 * w)
        h = int(h + 0.4 * h)
        x = int(x - w * 0.25)
        y = int(y - h * 0.2)
        #修正边界框，避免超出图像范围
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if clr == (255, 0, 0): #保留蓝色标记的有效轮廓
            boxes.append([x, y, (x + w), (y + h)])
            scores.append(arr_cont) #存储对应轮廓的面积

    return boxes, scores
