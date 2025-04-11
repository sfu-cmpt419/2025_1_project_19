import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from PIL import Image
import numpy as np
import cv2

# it doesn't make sense to map categorical values to numerical
# so, we use dl-model and ITA both for fitzpatrick scale
def compute_ita_from_image(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image at {image_path}")
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L_mean = lab[..., 0].mean()  # 0..100 in OpenCV's LAB
    b_mean = lab[..., 2].mean()  # -127..128
    epsilon = 1e-6
    ita = np.arctan((L_mean - 50) / (b_mean + epsilon)) * (180.0 / np.pi)
    return ita

def ita_to_fitzpatrick_letter(ita):
    # based on ITA thresholds used in fitzpatrick17k
    if ita > 55:
        return "I"
    elif ita > 41:
        return "II"
    elif ita > 28:
        return "III"
    elif ita > 10:
        return "IV"
    elif ita > -30:
        return "V"
    else:
        return "VI"

def ita_to_monk_scale(ita, min_ita=-60, max_ita=90):
    # clip ITA to [-60..90] then linearly map to [1..10]
    ita_clipped = np.clip(ita, min_ita, max_ita)
    monk = 1 + 9 * (max_ita - ita_clipped) / (max_ita - min_ita)
    return int(round(monk))

def load_effnetv2m(checkpoint_path, num_classes=6, device="cpu"):
    model = timm.create_model('efficientnetv2_rw_m', pretrained=False)
    
    # Replace classifier head
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() 
    
    return model

def predict_single_image(model, image_path, device="cpu"):
    # returns the predicted Fitzpatrick scale in [1..6] from the model.
    # and computes ITA for the same image, converting to fitzpatrick and monk

    # model-based fitzpatrick
    transform = T.Compose([
        T.Resize((224, 224)),  # same size as training
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    
    # load + preprocess
    img_pil = Image.open(image_path).convert("RGB")
    x = transform(img_pil)
    x = x.unsqueeze(0)  # [1, C, H, W]
    
    with torch.no_grad():
        x = x.to(device)
        logits = model(x)  # shape [1, 6]
        predicted_idx = torch.argmax(logits, dim=1).item()  # [0..5]
    
    # convert [0..5] -> [1..6]
    pred_fitz_int = predicted_idx + 1
    
    # ITA-based calculations
    ita = compute_ita_from_image(image_path)
    ita_fitz_letter = ita_to_fitzpatrick_letter(ita)
    monk_scale = ita_to_monk_scale(ita)

    # Return all info
    return {
        "model_fitz_1to6": pred_fitz_int,
        "ita_value": ita,
        "ita_fitz_letter": ita_fitz_letter,
        "monk_scale": monk_scale
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # load model
    checkpoint_path = "./trained-models/best_efficientnetv2m_fitzpatrick.pt"
    model = load_effnetv2m(checkpoint_path, num_classes=6, device=device)
    
    # predict on a sample image
    test_image = "../../images/sample-image.jpg" 
    results = predict_single_image(model, test_image, device)
    
    # print results
    print("🌈🔍 Skin Tone Detection Results 🔍🌈")
    print(f"📊ML-Model's Fitzpatrick (1..6): {results['model_fitz_1to6']}")
    print(f"📈ITA value: {results['ita_value']:.3f}")
    print(f"🔡ITA-based Fitz letter scale [I..VI]: {results['ita_fitz_letter']}")
    print(f"🎨ITA-based Monk Skin Tone scale [1..10]: {results['monk_scale']}")
