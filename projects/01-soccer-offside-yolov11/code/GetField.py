import cv2
import numpy as np

def get_field(img, write_dir, show=False, save_mask=True):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 获取绿色区域的掩码
    mask = cv2.inRange(hsv,(35,40,40),(70,255,255))

    if save_mask:
        cv2.imwrite("mask_green.jpg",mask)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    green = cv2.bitwise_and(img, mask_3ch)
    cv2.imwrite('green2.jpg',green)

    field_contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if field_contours:
        contour_areas = [cv2.contourArea(cnt) for cnt in field_contours]
        max_contour_idx = contour_areas.index(max(contour_areas))
        max_contour = field_contours[max_contour_idx]
    else:
        max_contour = None

    # 绘制最大的那个轮廓
    if max_contour is not None:
        game_area = np.zeros_like(mask)
        game_area_mask = cv2.drawContours(game_area, [max_contour], -1, (255,255,255), -1)
        game_area_mask_3ch = cv2.cvtColor(game_area_mask,cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(img, game_area_mask_3ch)#区掩码获得比赛区域

        # 只在需要时保存（图像处理时保存，视频处理时不保存以提升性能）
        if save_mask:
            cv2.imwrite(write_dir+"\\game_area.jpg", result)
        # 视频流程中无需弹窗，按需显示
        if show:
            cv2.namedWindow("Adaptive Window", cv2.WINDOW_NORMAL)
            cv2.imshow("Adaptive Window", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result
    else:
        print("未检测到轮廓")
        return 0


#cv2.imwrite("fieldcon.jpg",image)

def draw_all_contours(img, contours, output_path=None):
    """绘制所有轮廓到图像上"""
    result = img.copy()

    if not contours:
        print("没有检测到轮廓")
        return result

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]
        cv2.drawContours(result, [contour], -1, color, 2)

    if output_path:
        cv2.imwrite(output_path, result)

    return result

if __name__ == '__main__':
    img = cv2.imread('./images/test3.jpg')

    get_field(img,'./')