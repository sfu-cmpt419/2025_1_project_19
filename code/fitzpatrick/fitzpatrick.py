import torch
import torch.nn as nn
import torchvision.transforms as T

import timm
from PIL import Image
import numpy as np

def load_effnetv2m(checkpoint_path, num_classes=6, device="cpu"):
    # create the model definition
    model = timm.create_model('efficientnetv2_rw_m', pretrained=False)
    # replace classifier head
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    # load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval() 
    
    return model

def predict_single_image(model, image_path, device="cpu"):
    """
    Given a model and a path to a single image,
    returns the predicted Fitzpatrick scale in [1..6].
    """
    # define the same transforms used in validation
    transform = T.Compose([
        T.Resize((224, 224)),  # same image_size as training
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    
    # load and preprocess image
    image = Image.open(image_path).convert("RGB")
    x = transform(image) 
    x = x.unsqueeze(0)   
    
    with torch.no_grad():
        x = x.to(device)
        logits = model(x)  # shape [1, 6]
        # Get predicted index in [0..5]
        predicted_idx = torch.argmax(logits, dim=1).item()
    
    # convert [0..5] -> [1..6]
    predicted_fitz = predicted_idx + 1
    return predicted_fitz

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # load efficientv2m model
    checkpoint_path = "./trained-models/best_efficientnetv2m_fitzpatrick.pt"
    model = load_effnetv2m(checkpoint_path, num_classes=6, device=device)
    
    # predict on a custom image
    test_image = "../../images/sample-image.jpg"  # path to a face image
    prediction = predict_single_image(model, test_image, device)
    print("🎯 Fitzpatrick Prediction: ", prediction, "(scale: 1–6)")
    print("📸 Image:", test_image)
