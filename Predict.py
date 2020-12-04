from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

data_transform = transforms.Compose([
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# load image
img = Image.open("test.jpg")
print("----------------------------------")
plt.imshow(img)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)
img = img.to(DEVICE)

model=torch.load('weigths/ResNetFlowermodel-epoch8.pth',map_location='cpu')
#model.to(DEVICE)
flowers=['雏菊','蒲公英','玫瑰','向日葵','郁金香']
with torch.no_grad():
    output = torch.squeeze(model(img))
    print(output)
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
    print(flowers[predict_cla])
plt.show()