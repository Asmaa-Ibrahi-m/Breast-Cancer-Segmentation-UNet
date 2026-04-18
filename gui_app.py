import os
import cv2
import torch
import numpy as np
import gradio as gr
import segmentation_models_pytorch as smp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
MODEL_PATH = r"D:\UNI.projects\breast_cancer_model.pth"


model = smp.Unet(
    encoder_name="resnet18", 
    in_channels=1, 
    classes=1, 
    activation='sigmoid'
).to(device)


if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Model loaded successfully!")
else:
    print("Error: Model file not found. Please check the path.")


def predict_image(input_img):
    if input_img is None:
        return None, None
    
    if len(input_img.shape) == 3:
        gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = input_img
    
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    
   
    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)
    
    with torch.no_grad():
        output = model(tensor)
        mask = (output > 0.5).float().cpu().squeeze().numpy()
    
   
    mask_uint8 = (mask * 255).astype(np.uint8)
    overlay = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = mask_uint8 
  
    combined = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
    
    return mask_uint8, combined


with gr.Blocks(title="AI Breast Cancer System") as demo:
    gr.Markdown("#  Breast Cancer AI Detection System")
    gr.Markdown("Upload an ultrasound image to see the tumor segmentation.")
    
    with gr.Row():
        with gr.Column():
            input_component = gr.Image(label="Upload Ultrasound Image")
            submit_btn = gr.Button("Analyze Image", variant="primary")
        
        with gr.Column():
            mask_output = gr.Image(label="Binary Mask")
            overlay_output = gr.Image(label="Tumor Overlay (Visualized)")
            
    submit_btn.click(
        fn=predict_image, 
        inputs=input_component, 
        outputs=[mask_output, overlay_output]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)