import sys
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTextEdit, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong']

model_weight_path = "resnet18_plant.pth"
# 类别名称
class_names = ['万年青', '三叶草', '八角金盘', '凌霄花', '山茶', '构树', '桂花', '桃花', '樱花', '水杉',
               '爬山虎', '牵牛花', '玉兰', '秋枫', '紫茉莉', '红叶石楠', '罗汉松', '腊梅', '臭椿', '芦荟',
               '芭蕉', '莲花', '蓝花楹', '铁树', '银杏', '香樟树']

plant_introductions = {
    '万年青': '万年青是天门冬科万年青属多年生常绿草本植物，叶片翠绿有光泽，四季常青。\n'
              '花语：健康、长寿、吉祥如意，常用于室内观赏和园林绿化。\n'
              '特性：喜温暖湿润环境，耐阴不耐寒，对土壤要求肥沃疏松。\n'
              '校园分布地：图书馆花坛',

    '三叶草': '三叶草是豆科三叶草属植物，叶片通常为三片小叶，花呈白色或粉色。\n'
              '花语：幸运、幸福，传说找到四叶草的人会得到好运。\n'
              '特性：适应性强，耐旱耐贫瘠，是优良的牧草和地被植物，根系有固氮作用。\n'
              '校园分布地：校园花坛、草坪、道路旁均有分布' ,

    '八角金盘': '八角金盘是五加科八角金盘属常绿灌木，叶片大而掌状分裂（通常8片），形似手掌。\n'
                '花语：八方来财、聚四方才气，常用于庭院和室内盆栽。\n'
                '特性：喜阴湿温暖环境，耐寒性较强，对有害气体有一定抗性。\n'
                '校园分布地：文学院后花园',

    '凌霄花': '凌霄花是紫葳科凌霄属攀援藤本植物，花呈漏斗状，颜色鲜艳（红、橙红）。\n'
              '花语：敬佩、声誉，象征着积极向上的精神。\n'
              '特性：喜阳光充足，耐旱耐贫瘠，攀援能力强，常用于墙面、棚架绿化。\n'
              '校园分布地：文化广场',

    '山茶': '山茶是山茶科山茶属常绿灌木或小乔木，花大而艳丽，花色丰富（红、粉、白等）。\n'
            '花语：理想的爱、谦逊，是中国传统名花之一。\n'
            '特性：喜温暖湿润，耐半阴，不耐严寒，花期通常在冬春季。\n'
            '校园分布地：行署楼、共青团花园',

    '构树': '构树是桑科构属落叶乔木，生长迅速，适应性强。\n'
            '特性：耐干旱、耐贫瘠、耐盐碱，常用于荒山绿化和造纸原料。\n'
            '价值：树皮可造纸，果实可食用，树叶可作饲料。\n'
            '校园分布地：一运附近',

    '桂花': '桂花是木犀科木犀属常绿灌木或小乔木，花香浓郁，是中国传统十大名花之一。\n'
            '花语：崇高、美好、吉祥、友好。\n'
            '特性：喜温暖湿润，耐半阴，花期9-10月，常用于庭院观赏和香料提取。\n'
            '校园分布地：南北区道路旁、花园均有分布',

    '桃花': '桃花是蔷薇科李属落叶小乔木，花型美丽，花色艳丽（粉、白等）。\n'
            '花语：爱情、美好生活，象征着春天的到来。\n'
            '特性：喜阳光充足，耐旱耐寒，花期3-4月，是重要的观赏花木。\n'
            '校园分布地：桃园、桃花山',

    '樱花': '樱花是蔷薇科李属落叶乔木，花型优美，花色淡雅（粉、白），花期短暂而绚烂。\n'
            '花语：希望、纯洁、生命，是日本国花。\n'
            '特性：喜阳光充足，耐寒性较强，对土壤要求肥沃排水良好，花期3-4月。\n'
            '校园分布地：橘园路旁、三十一教附近',

    '水杉': '水杉是杉科水杉属落叶乔木，是中国特有的珍稀孑遗植物，被称为"活化石"。\n'
            '特性：喜温暖湿润，耐寒性强，生长迅速，常用于园林绿化和用材林。\n'
            '价值：具有极高的科研价值和观赏价值，是国家一级保护植物。\n'
            '校园分布地：李园、音乐学院附近',

    '爬山虎': '爬山虎是葡萄科地锦属攀援藤本植物，叶片秋季变红，攀援能力极强。\n'
              '特性：喜阴湿环境，耐旱耐寒，对土壤要求不高，常用于墙面、山体绿化。\n'
              '价值：具有很好的遮阳、降温、滞尘效果。'
              '校园分布地：行署楼、一运',

    '牵牛花': '牵牛花是旋花科牵牛属一年生缠绕草本植物，花呈漏斗状，颜色丰富。\n'
              '花语：爱情永固、名誉，象征着顽强的生命力。\n'
              '特性：喜阳光充足，耐干旱，适应性强，清晨开花，午后闭合。\n'
              '校园分布地：杏园',

    '玉兰': '玉兰是木兰科木兰属落叶乔木，花大而洁白，香气浓郁，是中国传统名花，也是我们西南大学的校花\n'
            '花语：高洁、芬芳、纯洁，象征着美好的品德。\n'
            '特性：喜阳光充足，耐寒性较强，花期2-3月（先花后叶），常用于庭院观赏。\n'
            '校园分布地：行署楼、共青团花园、李园均有分布',

    '秋枫': '秋枫是大戟科秋枫属常绿或半常绿乔木，叶片秋季变红。\n'
            '特性：喜温暖湿润，耐水湿，耐寒性较强，生长迅速，常用于园林绿化和用材。\n'
            '价值：木材坚硬，可用于建筑、家具等，树皮可入药。'
            '校园分布地：梅园',

    '紫茉莉': '紫茉莉是紫茉莉科紫茉莉属一年生草本植物，花呈漏斗状，颜色多样（红、粉、白、黄等）。\n'
              '花语：质朴、胆小、怯懦，象征着纯洁的爱情。\n'
              '特性：喜温暖湿润，耐半阴，适应性强，傍晚开花，清晨闭合。\n'
              '校园分布地：杏园、共青团花园',

    '红叶石楠': '红叶石楠是蔷薇科石楠属常绿灌木或小乔木，新叶红色鲜艳，观赏价值高。\n'
                '特性：喜阳光充足，耐半阴，耐寒耐旱，适应性强，常用于园林绿化和色块布置。\n'
                '校园分布地：共青团花园',

    '罗汉松': '罗汉松是罗汉松科罗汉松属常绿乔木，树形优美，种子形似罗汉。\n'
              '花语：长寿、吉祥、守财，是优良的观赏树种。\n'
              '特性：喜温暖湿润，耐半阴，耐寒性较弱，生长缓慢，常用于盆栽和庭院绿化。'
              '校园分布地：共青团花园',

    '腊梅': '腊梅是蜡梅科蜡梅属落叶灌木，花黄似蜡，香气浓郁，在寒冬开放。\n'
            '花语：坚强、傲骨、高雅，象征着不屈不挠的精神。\n'
            '特性：喜阳光充足，耐寒性强，耐旱，花期11月至次年3月。\n'
            '校园分布地：橘园、八教、李园附近',

    '臭椿': '臭椿是苦木科臭椿属落叶乔木，生长迅速，适应性强。\n'
            '特性：耐干旱、耐贫瘠、耐盐碱，抗污染能力强，常用于荒山绿化和行道树。\n'
            '注意：树叶有特殊气味，部分人可能不适。\n'
            '校园分布地：杏园、二运旁',

    '芦荟': '芦荟是阿福花科芦荟属多年生常绿草本植物，叶片肥厚多汁，富含芦荟多糖。\n'
            '花语：青春之源、洁身自爱，象征着健康和美丽。\n'
            '特性：喜阳光充足，耐旱耐贫瘠，对土壤要求不高，具有药用和美容价值。\n'
            '校园分布地：橘园花坛',

    '芭蕉': '芭蕉是芭蕉科芭蕉属多年生草本植物，叶片大而宽阔，树形优美。\n'
            '特性：喜温暖湿润，不耐寒，对土壤要求肥沃排水良好，常用于庭院和热带景观布置。\n'
            '价值：果实可食用，叶片可用于包裹食物或编织。\n'
            '校园分布地：植保院、校史馆旁',

    '莲花': '莲花是莲科莲属多年生水生草本植物，花大而美丽，出淤泥而不染。\n'
            '花语：纯洁、高雅、正直、廉洁，是中国传统名花之一。\n'
            '特性：喜阳光充足，适宜在浅水中生长，花期6-9月，具有很高的观赏和食用价值。\n'
            '校园分布地：崇德湖',

    '蓝花楹': '蓝花楹是紫葳科蓝花楹属落叶乔木，花呈蓝紫色，形似漏斗，观赏价值极高。西南大学名花。\n'
              '花语：宁静、深远、忧郁，象征着美好的爱情。\n'
              '特性：喜温暖湿润，阳光充足，不耐寒，花期5-6月，常用于庭院和行道树。\n'
              '校园分布地：八教、四运、中图附近',

    '铁树': '铁树是苏铁科苏铁属常绿乔木，树形古朴，叶片坚硬有光泽。\n'
            '花语：坚强、不屈不挠，象征着长寿和吉祥。\n'
            '特性：喜阳光充足，耐旱耐贫瘠，生长缓慢，寿命长，常用于盆栽和庭院绿化。\n'
            '校园分布地：共青团花园',

    '银杏': '银杏是银杏科银杏属落叶乔木，是中国特有的珍稀孑遗植物，被称为"活化石"。\n'
            '花语：坚韧、沉着、永恒的爱。\n'
            '特性：喜阳光充足，耐寒耐旱，适应性强，秋季叶片变黄，是优良的观赏和用材树种。\n'
            '校园分布地：八教、文学院',

    '香樟树': '香樟树是樟科樟属常绿乔木，树形高大挺拔，枝叶茂密，有特殊香气。西南大学数量最多的树。\n'
              '特性：喜温暖湿润，耐半阴，耐寒性较强，抗污染能力强，常用于行道树和庭院绿化。\n'
              '价值：木材坚硬耐腐，可用于建筑、家具等，枝叶可提取樟脑和樟油。\n'
              '校园分布地：随处可见',
}

# ==================================================
# 数据预处理和模型加载（修改为ResNet50）
# ==================================================
# 数据预处理（保持不变，和训练时一致）
predict_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 加载模型（关键修改：ResNet50）
def load_model(num_classes, weight_path):
    # 关键修改1：改为ResNet50 + ImageNet V2预训练权重
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # 全连接层适配类别数（自动匹配ResNet50的2048维输入）
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载ResNet50的训练权重
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


# 加载模型（全局初始化）
num_classes = len(class_names)
model, device = load_model(num_classes, model_weight_path)


# ==================================================
# UI主窗口类（保持不变）
# ==================================================
class PlantPredictUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # 窗口配置
        self.setWindowTitle("西南大学常见树种识别系统")
        self.setGeometry(100, 100, 1200, 800)  # 窗口位置和大小

        # 中心部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. 顶部按钮区域
        btn_layout = QHBoxLayout()

        # 选择图片按钮
        self.select_btn = QPushButton("选择图片")
        self.select_btn.clicked.connect(self.select_image)
        self.select_btn.setStyleSheet("font-size: 14px; padding: 10px 20px;")
        btn_layout.addWidget(self.select_btn)

        # 预测按钮
        self.predict_btn = QPushButton("开始识别")
        self.predict_btn.clicked.connect(self.predict_image)
        self.predict_btn.setStyleSheet("font-size: 14px; padding: 10px 20px; background-color: #4CAF50; color: white;")
        self.predict_btn.setEnabled(False)  # 初始禁用
        btn_layout.addWidget(self.predict_btn)

        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # 2. 图片显示区域
        img_layout = QHBoxLayout()

        # 原始图片显示
        self.raw_img_label = QLabel("原始图片")
        self.raw_img_label.setStyleSheet("border: 1px solid #cccccc;")
        self.raw_img_label.setAlignment(Qt.AlignCenter)
        self.raw_img_label.setMinimumSize(400, 400)
        img_layout.addWidget(self.raw_img_label)

        # 预测结果显示（概率分布图）
        self.result_img_label = QLabel("概率分布")
        self.result_img_label.setStyleSheet("border: 1px solid #cccccc;")
        self.result_img_label.setAlignment(Qt.AlignCenter)
        self.result_img_label.setMinimumSize(500, 400)
        img_layout.addWidget(self.result_img_label)

        main_layout.addLayout(img_layout)

        # 3. 预测结果和介绍区域
        result_layout = QHBoxLayout()

        # 预测结果文本框
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("font-size: 14px; padding: 10px;")
        self.result_text.setMinimumHeight(200)
        result_layout.addWidget(self.result_text)

        # 植物介绍文本框（带滚动条）
        self.intro_text = QTextEdit()
        self.intro_text.setReadOnly(True)
        self.intro_text.setStyleSheet("font-size: 20px; padding: 10px;")
        self.intro_text.setMinimumHeight(200)
        result_layout.addWidget(self.intro_text)

        main_layout.addLayout(result_layout)

        # 存储当前选中的图片路径
        self.current_image_path = None

    def select_image(self):
        """选择图片并显示"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.gif)"
        )
        if file_path:
            self.current_image_path = file_path
            # 显示原始图片
            self.show_raw_image(file_path)
            # 启用预测按钮
            self.predict_btn.setEnabled(True)
            # 清空之前的结果
            self.result_text.clear()
            self.intro_text.clear()
            self.result_img_label.setPixmap(QPixmap())
            self.result_img_label.setText("概率分布")

    def show_raw_image(self, image_path):

        pixmap = QPixmap(image_path)
        # 缩放图片以适应标签大小（保持比例）
        scaled_pixmap = pixmap.scaled(
            self.raw_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.raw_img_label.setPixmap(scaled_pixmap)
        self.raw_img_label.setText("")  # 清空文本

    def predict_image(self):
        """预测图片并显示结果"""
        if not self.current_image_path:
            return

        try:
            # 加载和预处理图片
            image = Image.open(self.current_image_path).convert("RGB")
            input_tensor = predict_transform(image).unsqueeze(0).to(device)

            # 预测
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                predicted_class = class_names[predicted_idx]
                confidence = probabilities[0][predicted_idx].item() * 100

            # 生成概率分布图
            self.generate_probability_plot(probabilities, predicted_idx)

            # 显示预测结果
            result_str = f"预测结果：{predicted_class}\n"
            result_str += f"置信度：{confidence:.2f}%\n"
            result_str += "\n各类别概率（前10名）：\n"
            # 获取概率排序（降序）
            prob_sorted_idx = torch.argsort(probabilities, dim=1, descending=True).squeeze().cpu().numpy()
            for i in range(min(10, len(class_names))):
                idx = prob_sorted_idx[i]
                result_str += f"{i + 1}. {class_names[idx]}: {probabilities[0][idx].item() * 100:.2f}%\n"
            self.result_text.setText(result_str)

            # 显示植物介绍
            intro = plant_introductions.get(predicted_class)
            self.intro_text.setText(f"【{predicted_class}】\n\n{intro}")

        except Exception as e:
            self.result_text.setText(f"预测失败：{str(e)}")

    def generate_probability_plot(self, probabilities, predicted_idx):
        # 确保matplotlib使用非交互后端（避免QT冲突）
        plt.switch_backend('Agg')
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        # 只显示前5个概率最高的类别
        prob_sorted_idx = torch.argsort(probabilities, dim=1, descending=True).squeeze().cpu().numpy()
        top5_idx = prob_sorted_idx[:5]
        top5_classes = [class_names[i] for i in top5_idx]
        top5_probs = [probabilities[0][i].item() * 100 for i in top5_idx]
        # 设置颜色（预测类别为红色，其他为浅蓝色）
        colors = ["red" if i == predicted_idx else "lightblue" for i in top5_idx]

        # 绘制柱状图
        ax.bar(top5_classes, top5_probs, color=colors)
        ax.set_xlabel("植物类别", fontsize=12)
        ax.set_ylabel("概率 (%)", fontsize=12)
        ax.set_title("各类别概率分布（前5名）", fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        # 转换为QPixmap并显示
        qimage = QImage.fromData(buf.getvalue(), 'PNG')
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.result_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.result_img_label.setPixmap(scaled_pixmap)
        self.result_img_label.setText("")  # 清空文本
        plt.close(fig)
        buf.close()

# ==================================================
# 运行UI
# ==================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = PlantPredictUI()
    ui.show()
    sys.exit(app.exec_())