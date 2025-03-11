import argparse
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def validate(model, valloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(valloader, desc="Validation", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(valloader)

def train(args):
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Stanford Cars dataset
    dataset = datasets.StanfordCars(root=args.data_dir, split="train", download=False, transform=transform_train)
    
    # Split the dataset into training and validation sets
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    trainset = torch.utils.data.Subset(dataset, train_idx)
    valset = torch.utils.data.Subset(
        datasets.StanfordCars(root=args.data_dir, split="train", download=False, transform=transform_val),
        val_idx
    )
    

    #Training with the entire dataset to improve performance
    #TODO: Reduce size of valset and experiment 
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    if os.path.exists(os.path.join(args.model_dir, "model_final.pth")):
        print("Loading checkpoint...")
        model = models.resnet50()
        model.fc = torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(model.fc.in_features, args.num_classes))
        checkpoint = torch.load(os.path.join(args.model_dir, "model_final.pth"), map_location=device)
        model.load_state_dict(checkpoint)
        print("Checkpoint loaded!")
    else:
        print("No checkpoint found, using pretrained ResNet model.")
        model = models.resnet50(weights='DEFAULT')
        model.fc = torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(model.fc.in_features, args.num_classes))
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = args.patience

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs} - Training", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1} complete. Average Training Loss: {avg_train_loss:.4f}")

        #TODO: Uncomment this block to enable validation. This is disabled for now to speed up training and to get the most out of the limited dataset

        # avg_val_loss = validate(model, valloader, criterion, device)
        # print(f"Epoch {epoch+1} complete. Average Validation Loss: {avg_val_loss:.4f}")
        # torch.save(model.state_dict(), os.path.join(args.model_dir, f"model_{epoch}_{avg_val_loss}.pth"))

        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     early_stop_counter = 0
        #     torch.save(model.state_dict(), os.path.join(args.model_dir, "best_model.pth"))
        #     print("Best model saved.")
        # else:
        #     early_stop_counter += 1
        #     if early_stop_counter >= patience:
        #         print("Early stopping triggered.")
        #         break

        scheduler.step()

    torch.save(model.state_dict(), os.path.join(args.model_dir, "model_final.pth"))
    print("Final model saved to", os.path.join(args.model_dir, "model_final.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-classes", type=int, default=196)
    parser.add_argument("--data-dir", type=str, default="./")
    parser.add_argument("--model-dir", type=str, default="./model")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    
    args = parser.parse_args()
    train(args)
