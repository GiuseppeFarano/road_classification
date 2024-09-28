import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
import sys

# Ensure you replace 'YOUR_MODEL_PATH' with the actual path to your model
MODEL_PATH_CE = 'resnet50_CE.pth'  # Change this to your saved model path
MODEL_PATH_ORD = 'resnet50_ORD.pth'  # Change this to your saved model path

# Device configuration
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Prepare the image transformations
image_transforms = transforms.Compose([
    transforms.CenterCrop(size=256),
    transforms.Resize(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Load the model
def load_model(model_path):
    # Load your model architecture here (you can copy the architecture code from the notebook)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Linear(256, 3)
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


# Make a prediction
def predict(model, image_path):
    # Load and prepare the image
    test_image = Image.open(image_path).convert('RGB')
    test_image_tensor = image_transforms(test_image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Model outputs probabilities
        outputs = model(test_image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        topk, topclass = probabilities.topk(1, dim=1)

        # Print predicted class and probabilities
        print(f"Predicted Road Quality: {topclass.item()} with probability: {topk.item():.4f}")

    # Optionally display the image
    plt.imshow(test_image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Load the models
    model_CE = load_model(MODEL_PATH_CE)
    model_ORD = load_model(MODEL_PATH_ORD)

    # Provide the image path for inference
    image_path = "immagine_test_strada.png"  # Change this to the image you want to test
    predict(model_CE, image_path)  # Use model_CE for Cross Entropy predictions
    # To use the Ordinal Regression model, uncomment the next line
    # predict(model_ORD, image_path)
