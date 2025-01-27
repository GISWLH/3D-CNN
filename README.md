# 3D-CNN
A implementation of "Deep learning for multi-year ENSO forecasts"

## model

```
import torch
import torch.nn as nn
import os

class CNN3D(nn.Module):
    def __init__(self, input_features, output_features, target_time_steps):
        super(CNN3D, self).__init__()
        # First layer: extracting spatial and temporal features
        self.conv1 = nn.Conv3d(
            in_channels=input_features,
            out_channels=12,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)
        )
        
        # Second layer: further extract deep features
        self.conv2 = nn.Conv3d(
            in_channels=12,
            out_channels=24,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)
        )
        
        # Third layer: reduce the time step from 16 to 3, and the feature dimension is 1
        self.conv3 = nn.Conv3d(
            in_channels=24,
            out_channels=output_features,
            kernel_size=(3, 1, 1),
            stride=(5, 1, 1),
            padding=(0, 0, 0)
        )

        # Activation Function and Dropout
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout3d(p=0.1)

    def forward(self, x):
        # Input: (batch, feature, time, height, width)
        x = self.tanh(self.conv1(x))  # (1, 8, 16, 721, 1440) -> (1, 64, 16, 721, 1440)
        x = self.dropout(x)

        x = self.tanh(self.conv2(x))  # (1, 64, 16, 721, 1440) -> (1, 128, 16, 721, 1440)
        x = self.dropout(x)
        x = self.conv3(x)  # (1, 128, 16, 721, 1440) -> (1, 1, 3, 721, 1440)
        return x
```

## infer

```

DEVICE = torch.device("cuda:0")
the_model = CNN3D(input_features=8, output_features=1, target_time_steps=16).to(DEVICE)
#the_model.load_state_dict(torch.load('D:/Onedrive/Acdemic/weather/model/geoformer_epoch_50.pth')['model_state_dict'])
the_model = torch.load('D:/Onedrive/Acdemic/weather/model/3DCNN_0005_100ep.pth')
with torch.no_grad():
    the_model.eval()
    for step,(upper_air, target_surface) in enumerate(test_loader):
        upper_air, target_surface = upper_air.to(DEVICE), target_surface.to(DEVICE)
        output = the_model(upper_air.cuda())
        if step == 0:
            break

```

```
import torch
from utils import plot
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
outimg = output.squeeze(0).squeeze(0)[0].cpu().detach().numpy() * std_all[0].numpy() + mean_all[0].numpy()
lat = np.linspace(90, -90, 721)
lon = np.linspace(0, 359.75, 1440)
sst_dataarray = xr.DataArray(
    outimg[0, :, :],
    coords=[("lat", lat), ("lon", lon)],
    name="sst"
)
fig = plt.figure()
proj = ccrs.Robinson() #ccrs.Robinson()ccrs.Mollweide()Mollweide()
ax = fig.add_subplot(111, projection=proj)
levels = np.linspace(270, 310, num=19)
plot.one_map_flat(sst_dataarray, ax, levels=levels, cmap="RdBu_r", mask_ocean=False, add_coastlines=True, add_land=False, plotfunc="pcolormesh")

```

![image-20250127144144425](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/image-20250127144144425.png)
