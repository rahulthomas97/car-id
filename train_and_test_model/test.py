import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from tqdm import tqdm

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to the saved model checkpoint
CHECKPOINT_PATH = "./model/model_final.pth"

# Load the test dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.StanfordCars(root="./", split="test", transform=transform, download=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the trained model
def load_model(checkpoint_path):
    model = models.resnet50(pretrained=False)  # Use the same architecture as during training
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.5),torch.nn.Linear(model.fc.in_features, 196))
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")
    return model

# Check if checkpoint exists
if os.path.exists(CHECKPOINT_PATH):
    model = load_model(CHECKPOINT_PATH)
else:
    raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}")

# Define the evaluation function
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    batch_idx=0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()


            _, predicted = torch.max(outputs, 1)
            # print(predicted.shape)
            # for i in range(len(predicted)):
            #     if predicted[i] != labels[i]:
            #         print(batch_idx*32+i+1,test_dataset.classes[predicted[i]], "-----------",test_dataset.classes[labels[i]])
                
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_idx+=1
            print(total, correct)

            progress_bar.set_postfix(loss=running_loss / len(test_loader), acc=100 * correct / total)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Run evaluation
evaluate_model(model, test_loader)
