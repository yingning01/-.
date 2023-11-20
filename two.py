import os
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 文件夹路径
folders = ['/home/aivs/Desktop/小功能模块测试/良品图', '/home/aivs/Desktop/小功能模块测试/良品图/漏件图']  # 替换为实际文件夹路径

# 初始化空的数据列表和标签列表
data = []
labels = []

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

# 为每个文件夹中的图片赋予标签
for label, folder in enumerate(folders):
    for file in os.listdir(folder):
        image_path = os.path.join(folder, file)
        image = cv2.imread(image_path)
        #image = make_square(image)
        # 在这里你可能需要对图像进行预处理，比如缩放、归一化等
        # 假设你已经对图像进行了预处理
        data.append(image)
        labels.append(label)  # 根据文件夹的顺序为图片赋予标签（0表示良品，1表示次品）

# 转换为NumPy数组
data = np.array(data)
labels = np.array(labels)
# 划分训练集和验证集
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 数据转换为 PyTorch 张量
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(train_data, train_labels, transform=transform)
val_dataset = CustomDataset(val_data, val_labels, transform=transform)

# 定义 SqueezeNet 模型
model = models.squeezenet1_0(pretrained=False)
# 替换输出层为二分类的全连接层
model.classifier[1] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
#model.num_classes = 2

#model_shufflenet_v2 = models.shufflenet_v2_x0_5(pretrained=False)

# 定义损失函数和优化器
import torch.nn.functional as F
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20)

best_accuracy = 0.0
best_weights = None

for epoch in range(50):  # 假设进行 5 个 epoch 的训练
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = F.cross_entropy(output, labels.to(torch.long))
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            output = model(images)
            val_loss += F.cross_entropy(output, labels.to(torch.long))
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
      # 检查验证集准确率并保存最优权重
    val_accuracy = correct / total  # 根据你的代码计算准确率的部分
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_weights = model.state_dict().copy()  # 保存最优权重

      # 在训练结束后，保存最佳权重到文件
    if best_weights is not None:
       torch.save(best_weights, 'best_model_weights.pth')
    
    print(f"Epoch {epoch+1}: Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {correct/total}")
