import torch
import torch.nn as nn

class CNN3D(nn.Module):
    def __init__(self, input_features, output_features, target_time_steps):
        super(CNN3D, self).__init__()
        # First layer: extracting spatial and temporal features
        self.conv1 = nn.Conv3d(
            in_channels=input_features,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)
        )
        
        # Second layer: further extract deep features
        self.conv2 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)
        )
        
        # Third layer: reduce the time step from 16 to 3, and the feature dimension is 1
        self.conv3 = nn.Conv3d(
            in_channels=128,
            out_channels=output_features,
            kernel_size=(3, 1, 1),
            stride=(5, 1, 1),
            padding=(0, 0, 0)
        )

        # Activation Function and Dropout
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout3d(p=0.3)

    def forward(self, x):
        # Input: (batch, feature, time, height, width)
        x = self.tanh(self.conv1(x))  # (1, 8, 16, 721, 1440) -> (1, 64, 16, 721, 1440)
        x = self.dropout(x)

        x = self.tanh(self.conv2(x))  # (1, 64, 16, 721, 1440) -> (1, 128, 16, 721, 1440)
        x = self.dropout(x)
        x = self.conv3(x)  # (1, 128, 16, 721, 1440) -> (1, 1, 3, 721, 1440)
        return x
    
normalized_data = torch.load('E:/data/normalized_train.pt')
from torch.utils.data import Dataset, DataLoader
from random import randint
import torch
import torch.nn as nn
import torch.optim as optim

class WeatherDataset1(Dataset):
    def __init__(self, input_data, train=False, validation=False, test=False):
        self.input = input_data
        self.train = train
        self.validation = validation
        self.test = test
        if self.train:
            self.input = self.input[:608]  
        elif self.validation:
            self.input = self.input[-160:-80] 
        elif self.test:
            self.input = self.input[-80:] 
        print("input: ",self.input.shape)

    def __len__(self):
        if self.train:
            return 608 - 20  # 确保索引不会超出范围
        else:
            return 80 - 20
    

    def __getitem__(self, idx):
        upper_input = torch.tensor(self.input).permute(1, 0, 2, 3)[:, idx:idx + 16, :, :]
        target_surface = self.input[idx + 16: idx + 16 + 3, 0, :, :]
        return upper_input, target_surface
    
DEVICE = torch.device("cuda:0")
model = CNN3D(input_features=8, output_features=1, target_time_steps=16).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# 创建数据集和数据加载器
train_dataset = WeatherDataset1(normalized_data, train=True)
train_loader = DataLoader(train_dataset, batch_size=1,shuffle=False, generator=torch.Generator(device='cpu'))

valid_dataset = WeatherDataset1(normalized_data, validation=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

custom_loss = torch.nn.MSELoss()
epochs = 100
log_file = "loss_log.txt"
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for i, (upper_air, target_surface) in enumerate(train_loader):    
        upper_air, target_surface = upper_air.to(DEVICE), target_surface.to(DEVICE)
        optimizer.zero_grad()  # 清除旧的梯度
        output_surface = model(upper_air)
        
        output_surface = output_surface.squeeze(0).squeeze(0)
        target_surface = target_surface.squeeze(0).squeeze(0)

        loss = custom_loss(output_surface, target_surface)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    
    # 验证过程
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for j,(upper_air, target_surface) in enumerate(valid_loader):
            upper_air, target_surface = upper_air.to(DEVICE), target_surface.to(DEVICE)
            output_surface = model(upper_air)
            output_surface = output_surface.squeeze(0).squeeze(0)
            target_surface = target_surface.squeeze(0).squeeze(0)
            loss = custom_loss(output_surface, target_surface)
            valid_loss += loss.item()

    # 计算验证集的平均损失
    valid_loss /= len(valid_loader)

    print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {valid_loss}")
    with open(log_file, "a") as f:
        f.write(f"{epoch},{train_loss},{valid_loss}\n")
    if (epoch + 1) % 20 == 0:
        checkpoint_path = f'model_checkpoints/model_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }, checkpoint_path)
        print(f"Model checkpoint saved at epoch {epoch + 1}")
        
torch.save(model, '3DCNN_0005_100ep.pth')