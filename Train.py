from __future__ import print_function, division
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from ResNetModel import resnet50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
     transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

dataset =torchvision.datasets.ImageFolder(root='D:/DATASETS/flower_photos',transform=train_transform)

train_dataset, valid_dataset = train_test_split(dataset,test_size=0.2, random_state=0)
print(len(train_dataset))
print(len(valid_dataset))
train_loader =DataLoader(train_dataset,batch_size=4, shuffle=True,num_workers=0)#Batch Size定义：一次训练所选取的样本数。 Batch Size的大小影响模型的优化程度和速度。
valid_loader =DataLoader(valid_dataset,batch_size=4, shuffle=True,num_workers=0)#Batch Size定义：一次训练所选取的样本数。 Batch Size的大小影响模型的优化程度和速度。

print(len(train_dataset))
# for image in train_loader:
#     valid_image,valid_label=image
#     print('valid_label:',valid_label[0])
#     print('valid_image shape：',valid_image[0].shape)
#     print(valid_image[0].dtype)
#     plt.imshow(valid_image[0].permute(2, 1, 0))
#     plt.show()
#     break


model = resnet50()
model.load_state_dict(torch.load('weigths/resnet50.pth'))
inchannel = model.fc.in_features
print('inchannel:',inchannel)
model.fc = nn.Linear(inchannel, 5)#输出层设置为5，表示五分类
model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
loss_func = torch.nn.CrossEntropyLoss()
avg_loss=[]
avg_acc=[]

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target)  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            # print('预测值：',pred,"真实值：",target)
            # print('target.view_as(pred):',target.view_as(pred))
            # print('pred.eq(target.view_as(pred)):',pred.eq(target.view_as(pred)))
            # print('pred.eq(target.view_as(pred)).sum():',pred.eq(target.view_as(pred)).sum())
            # print('是否相等：',pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    avg_loss.append(test_loss)
    avg_acc.append(100. * correct / len(test_loader.dataset))

for epoch in range(1, 9):
    train(model,  DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, valid_loader)
torch.save(model, 'weigths/ResNetFlowermodel-epoch8.pth')
epoch=range(1,9)
#plt.plot(epoch, avg_loss, color='red')
plt.plot(epoch, avg_acc, label='acc changes',color='blue')
for a,b in zip(epoch,avg_acc):
    plt.text(a, b+0.05, '%.1f' % b, ha='center', va= 'bottom',fontsize=9)
plt.xlabel('epochs')# 横坐标描述
plt.ylabel('accuracy')# 纵坐标描述
plt.legend()#显示图例
plt.show()