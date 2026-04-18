import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torch.amp import GradScaler, autocast  
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
BATCH_SIZE = 8 
EPOCHS = 50           
LEARNING_RATE = 1e-4  

class BreastCancerDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.masks = []
        categories = ['benign', 'malignant', 'normal']
        
        if not os.path.exists(root_dir):
            return

        for cat in categories:
            folder_path = os.path.join(root_dir, cat)
            if not os.path.exists(folder_path):
                continue
                
            for f in os.listdir(folder_path):
                if f.lower().endswith(".png") and "_mask" not in f.lower():
                    img_path = os.path.join(folder_path, f)
                    mask_path = img_path.replace(".png", "_mask.png")
                    
                    if os.path.exists(mask_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        
        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        return image, (mask > 0.5).float()

model = smp.Unet(
    encoder_name="resnet18", 
    in_channels=1, 
    classes=1, 
    activation='sigmoid'
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = smp.losses.DiceLoss(mode='binary')
scaler = GradScaler('cuda') 

def train_model(train_loader):
    print(f"Training started on {device}... Target: {EPOCHS} Epochs")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_dice = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                
                # حساب Accuracy (Dice Coefficient)
                pred = (outputs > 0.5).float()
                intersection = (pred * masks).sum()
                dice = (2. * intersection) / (pred.sum() + masks.sum() + 1e-8)
                total_dice += dice.item()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_dice / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy (Dice): {avg_acc*100:.2f}%")

def save_visual_result(dataset):
    model.eval()
    test_idx = min(20, len(dataset)-1)
    image, mask = dataset[test_idx]
    
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device))
        pred = (pred > 0.5).float().cpu().squeeze().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(image.squeeze(), cmap='gray'); plt.title("Original Image")
    plt.subplot(1, 3, 2); plt.imshow(mask.squeeze(), cmap='gray'); plt.title("True Mask")
    plt.subplot(1, 3, 3); plt.imshow(pred, cmap='jet'); plt.title("Model Prediction")
    
    result_path = r"D:\UNI.projects\latest_result.png"
    plt.savefig(result_path)
    plt.close()
    print(f"Result image saved to: {result_path}")

if __name__ == "__main__":
    data_path = r"D:\UNI.projects\Dataset_BUSI_with_GT"
    
    sub_path = os.path.join(data_path, "Dataset_BUSI_with_GT")
    if os.path.exists(os.path.join(sub_path, "benign")):
        data_path = sub_path

    dataset = BreastCancerDataset(data_path)

    if len(dataset) == 0:
        print(f"Error: No images found in {data_path}")
    else:
        print(f"Success! Found {len(dataset)} image-mask pairs.")
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        train_model(loader)
        save_visual_result(dataset)
        
        model_save_path = r"D:\UNI.projects\breast_cancer_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(r"All Done! Results are in D:\UNI.projects")