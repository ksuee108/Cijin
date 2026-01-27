import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
 
#-----------------------------ResNet50-----------------------------------------------
def prepare_50_model(num_classes=2):
    print("Preparing DeepLabV3 ResNet50 model...")
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.classifier = nn.Sequential(
        nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(1024, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(512, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(256, num_classes, kernel_size=1)
    )
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    return model

"""
Uncomment the following lines to train on the DeepLabV3 ResNet101 model.
"""
# import torch.nn as nn
#-----------------------------ResNet101-----------------------------------------------
from torchvision.models.segmentation import deeplabv3_resnet101
 
def prepare_101_model(num_classes=2):
    print("Preparing DeepLabV3 ResNet101 model...")
    model = deeplabv3_resnet101(weights='DEFAULT')
    model.classifier = nn.Sequential(
        nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(1024, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(512, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(256, num_classes, kernel_size=1)
)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    return model

#-----------------------------MobileNetV3-----------------------------------------------

"""
Uncomment the following lines to train on the DeepLabV3 mobilNet model.
"""


from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
 
def prepare_mobilenet_model(num_classes=2):
    print("Preparing DeepLabV3 MobileNetV3 model...")
    model = deeplabv3_mobilenet_v3_large(weights='DEFAULT')

    model.classifier = nn.Sequential(
        nn.Conv2d(960, 1024, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(1024, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(512, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(256, num_classes, kernel_size=1)
    )

    # 修改 aux_classifier：輸入通道數來自 feature map（通常是 MobileNetV3 的中間層，應該是 40）
    model.aux_classifier = nn.Sequential(
        nn.Conv2d(40, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(256, num_classes, kernel_size=1)
    )
    return model

#-----------------------------U-net-----------------------------------------------
"""
class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=2):
        super(UNet, self).__init__()
        # Encoder (6層)
        self.enc1 = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2)

        self.enc6 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool6 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Conv2d(512, 1024, 3, padding=1)

        # Decoder (6層，對應回 enc6..enc1)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = nn.Conv2d(512 + 512, 512, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = nn.Conv2d(256 + 256, 256, 3, padding=1)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Conv2d(128 + 128, 128, 3, padding=1)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec5 = nn.Conv2d(32 + 32, 32, 3, padding=1)

        self.up6 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec6 = nn.Conv2d(16 + 16, 16, 3, padding=1)

        self.out_conv = nn.Conv2d(16, output_channels, 1)

    def forward(self, x):
        # Encoder
        c1 = F.relu(self.enc1(x)); p1 = self.pool1(c1)     # 16
        c2 = F.relu(self.enc2(p1)); p2 = self.pool2(c2)    # 32
        c3 = F.relu(self.enc3(p2)); p3 = self.pool3(c3)    # 64
        c4 = F.relu(self.enc4(p3)); p4 = self.pool4(c4)    # 128
        c5 = F.relu(self.enc5(p4)); p5 = self.pool5(c5)    # 256
        c6 = F.relu(self.enc6(p5)); p6 = self.pool6(c6)    # 512

        b = F.relu(self.bottleneck(p6))                    # 1024

        # Decoder（每層對應 concat c6->c1）
        u1 = self.up1(b)                        # 1024->512
        d1 = F.relu(self.dec1(torch.cat([u1, c6], dim=1))) # [512+512]->512

        u2 = self.up2(d1)                       # 512->256
        d2 = F.relu(self.dec2(torch.cat([u2, c5], dim=1))) # [256+256]->256

        u3 = self.up3(d2)                       # 256->128
        d3 = F.relu(self.dec3(torch.cat([u3, c4], dim=1))) # [128+128]->128

        u4 = self.up4(d3)                       # 128->64
        d4 = F.relu(self.dec4(torch.cat([u4, c3], dim=1))) # [64+64]->64

        u5 = self.up5(d4)                       # 64->32
        d5 = F.relu(self.dec5(torch.cat([u5, c2], dim=1))) # [32+32]->32

        u6 = self.up6(d5)                       # 32->16
        d6 = F.relu(self.dec6(torch.cat([u6, c1], dim=1))) # [16+16]->16

        return self.out_conv(d6)                # -> 2 類 logits
"""
class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=2):
        super(UNet, self).__init__()

        # 下采样
        self.enc1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)

        # 中間層
        self.bottleneck = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        # 上采样
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)

        # 輸出層
        self.out_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # 下采样
        c1 = F.relu(self.enc1(x))
        p1 = self.pool1(c1)

        c2 = F.relu(self.enc2(p1))
        p2 = self.pool2(c2)

        c3 = F.relu(self.enc3(p2))
        p3 = self.pool3(c3)

        c4 = F.relu(self.enc4(p3))
        p4 = self.pool4(c4)

        # 中間
        bottleneck = F.relu(self.bottleneck(p4))

        # 上采样
        u1 = self.up1(bottleneck)
        m1 = torch.cat([u1, c4], dim=1)
        c6 = F.relu(self.dec1(m1))

        u2 = self.up2(c6)
        m2 = torch.cat([u2, c3], dim=1)
        c7 = F.relu(self.dec2(m2))

        u3 = self.up3(c7)
        m3 = torch.cat([u3, c2], dim=1)
        c8 = F.relu(self.dec3(m3))

        u4 = self.up4(c8)
        m4 = torch.cat([u4, c1], dim=1)
        c9 = F.relu(self.dec4(m4))

        return self.out_conv(c9)
# 定義 Dataset
class PatchDataset(Dataset):
    def __init__(self, images, masks):
        # numpy -> torch tensor，並調整成 [B, C, H, W]
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        self.masks = torch.tensor(masks, dtype=torch.long).squeeze(3)  # 直接去掉通道維度
   
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

#----------------------------------CNN-LSTM------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # [B, C, H, W]
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(combined_conv, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width),
                torch.zeros(batch_size, self.hidden_dim, height, width))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=(3, 3), dropout=0.5, l2_reg=1e-4):
        super(ConvLSTM, self).__init__()
        padding = kernel_size[0] // 2

        layers = []
        self.cells = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.cells.append(ConvLSTMCell(prev_dim, h_dim, kernel_size, padding))
            prev_dim = h_dim

        self.dropout = nn.Dropout(dropout)
        self.final_conv = nn.Conv2d(prev_dim, 2, kernel_size=1)
        
    def forward(self, x):
        # x shape: [B, T, C, H, W]
        b, t, c, h, w = x.size()
        cur_input = x

        for i, cell in enumerate(self.cells):
            h_state, c_state = cell.init_hidden(b, (h, w))
            outputs = []
            for time_step in range(t):
                h_state, c_state = cell(cur_input[:, time_step, :, :, :], (h_state, c_state))
                outputs.append(h_state)
            cur_input = torch.stack(outputs, dim=1)  # [B, T, hidden_dim, H, W]

        # 取最後一個時間步
        last_output = cur_input[:, -1, :, :, :]  # [B, hidden_dim, H, W]
        last_output = self.dropout(last_output)
        out = self.final_conv(last_output)  # [B, 2, H, W]
        return out

# 在 Dataset 裡加一個時間步維度
class PatchDataset(Dataset):
    def __init__(self, images, masks, timesteps=3):
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        self.masks = torch.tensor(masks, dtype=torch.float32).permute(0, 3, 1, 2)
        self.timesteps = timesteps

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        # 重複同一張圖片形成時序
        img_seq = img.unsqueeze(0).repeat(self.timesteps, 1, 1, 1)  # [T, C, H, W]
        return img_seq, mask

#----------------------------------CNN------------------------------------------
class CNN(nn.Module):
    def __init__(self, dropout_rate=0.1, l2_reg=1e-4):
        super(CNN, self).__init__()

        # Downsampling
        self.c1 = nn.Conv2d(3, 64, kernel_size=6, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.c2 = nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.drop2 = nn.Dropout2d(dropout_rate)

        self.c3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.drop3 = nn.Dropout2d(dropout_rate)

        self.c4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.drop4 = nn.Dropout2d(dropout_rate)

        # Upsampling
        self.u1_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.u2_conv = nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False)

        # Output layer
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # Downsampling
        x = F.relu(self.bn1(self.c1(x)))
        x = F.max_pool2d(x, 2)  # /2

        x = F.relu(self.bn2(self.c2(x)))
        x = self.drop2(x)
        x = F.max_pool2d(x, 2)  # /4

        x = F.relu(self.bn3(self.c3(x)))
        x = self.drop3(x)
        x = F.max_pool2d(x, 2)  # /8

        x = F.relu(self.bn4(self.c4(x)))
        x = self.drop4(x)
        x = F.max_pool2d(x, 2)  # /16

        # Upsampling
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.u1_conv(x))

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.u2_conv(x))

        # Output
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.out_conv(x)

        return x

class PatchDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx].transpose(2, 0, 1), dtype=torch.float32)  # HWC → CHW
        mask = torch.tensor(self.masks[idx].squeeze(), dtype=torch.long)  # 單通道也轉成 CHW
        return img, mask