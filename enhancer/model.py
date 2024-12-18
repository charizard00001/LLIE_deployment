import torch.nn as nn
import torch.nn.functional as f
import torch
import os
from PIL import Image
import numpy as np
import cv2
import time
from scipy.stats import gaussian_kde
from xgboost import XGBRegressor

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model = 300, nhead = 10, dim_feedforward = 512, batch_first = True)
        self.transformer = nn.TransformerEncoder(self.layer, num_layers = 2)
        self.extra = nn.Sequential(
            nn.Conv2d(3, 128, (3, 3), padding = 'same'),
            nn.GELU(),
            nn.Conv2d(128, 3, (3, 3), padding = 'same')
        )
    def forward(self, x):
        x = self.img_to_patch(x, 10)
        x = self.transformer(x)
        x = self.patch_to_img(x, 10, 3, 400, 600)
        x = self.extra(x)
        return f.sigmoid(x)
    def img_to_patch(self, x, patch_size):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(1,2)            
        x = x.flatten(2,4)   
        return x
    def patch_to_img(self, x, patch_size, C, H, W):
        x = x.view(-1, H*W//(patch_size)**2, C , patch_size , patch_size)
        x = x.view(-1, H // patch_size, W // patch_size, C, patch_size, patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(-1, C, H, W)
        return x

def hist_from_quant(quant, c=0.75):
    x = np.linspace(0, 1, 255)
    kde = gaussian_kde(quant, c*quant.std())
    pdf = kde(x)
    return pdf

def convert(img, hist):
    input_hist, _ = np.histogram(img, bins=256, range=(0, 1))
    input_hist = input_hist / np.sum(input_hist)
    desired_hist = hist / np.sum(hist)
    input_cumsum = np.cumsum(input_hist)
    desired_cumsum = np.cumsum(desired_hist)
    mapping_func = np.interp(input_cumsum, desired_cumsum, np.linspace(0, 1, 255))
    matched_image = np.interp(img, np.linspace(0, 1, 256), mapping_func)
    return matched_image

def process_image(uploaded_image):

    device = torch.device('cpu')
    print('Loading trained models ...')
    print()
        
    model = Model()
    model = model.to(device)
    checkpoint = torch.load("models/transformer_conv_transform_new_input.pt", map_location = device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model_xgb = XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.2)
    model_xgb.load_model('models/my_xgb_model.model')

    print('Loaded trained models ...')
    print()

    allowed_extensions = {'.png', '.jpg', '.jpeg'}
    low = []

    extension = os.path.splitext(uploaded_image.name)[1].lower()
    if extension in allowed_extensions:
        x = Image.open(uploaded_image).convert('RGB')
        x = x.resize((600, 400), Image.LANCZOS)
        x = torch.from_numpy(np.array(x) / 255.0).permute(2, 0, 1)
        low.append(x)
        low = torch.stack(low)

    print('Preprocessing Images ...')
    start = time.time()
    out = model_xgb.predict(np.array([np.histogram(img, bins=256, range=(0,1))[0] for img in low.reshape(-1, 400, 600)]))
    new_input = []
    for i in range(len(low)):
        im=[]
        hist = hist_from_quant(out[3*i])
        im.append(convert(low[i][0], hist))
        hist = hist_from_quant(out[3*i+1])
        im.append(convert(low[i][1], hist))
        hist = hist_from_quant(out[3*i+2])
        im.append(convert(low[i][2], hist))
        new_input.append(im)

    new_input = torch.from_numpy(np.array(new_input)).float()    

    with torch.no_grad():
        result = model(new_input.to(device))

    image_array = result[0].permute(1, 2, 0).cpu().numpy() * 255 
    image_array = image_array.astype('uint8')

    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    output_image = Image.fromarray(image_rgb)
    return output_image

