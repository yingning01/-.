import torch
import pandas as pd
import cv2
from torchvision import transforms
import torchvision.models as models
import os
import numpy as np
import sys
# 文件夹路径
test_folder = 'sys.argv[1]'  # 测试集文件夹路径

# 读取模型权重
model = models.shufflenet_v2_x0_5(pretrained=False)
model.load_state_dict(torch.load('best_model_weights.pth'))
model.eval()

def make_square(image):
    height, width, _ = image.shape
    
    # 找到长宽中的最大值作为正方形的边长
    max_dimension = max(height, width)
    
    # 创建一个空白的正方形图像
    square_image = np.zeros((max_dimension, max_dimension, 3), dtype=np.uint8)
    
    # 计算需要填充的上下左右边界
    top = (max_dimension - height) // 2
    bottom = max_dimension - height - top
    left = (max_dimension - width) // 2
    right = max_dimension - width - left
    
    # 填充图像
    square_image[top:top+height, left:left+width] = image
    
    return square_image
# 数据预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 初始化空的数据列表和标签列表
test_data = []
file_names = []

# 读取测试图片并进行预测
for file in os.listdir(test_folder):
    image_path = os.path.join(test_folder, file)
    image = cv2.imread(image_path)
    image = make_square(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 默认颜色通道顺序为 BGR，需要转换为 RGB
    image = transform(image)
    test_data.append(image)
    file_names.append(file)

# 转换为 PyTorch 张量
test_data = torch.stack(test_data)

# 利用模型进行预测
predictions = []
with torch.no_grad():
    outputs = model(test_data)
    _, predicted = torch.max(outputs, 1)
    predictions = predicted.numpy()

# 保存预测结果到 CSV 文件
results = pd.DataFrame({'File Name': file_names, 'output Label': predictions})
# 替换标签为 True 和 False
results['output Label'] = results['output Label'].replace({0: True, 1: False})
results.to_csv('output.csv', index=False)
