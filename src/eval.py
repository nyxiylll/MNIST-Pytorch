import torch
import joblib

def evaluation(model,train_set,device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for features , labels in train_set:
            features = features.to(device)
            labels = labels.to(device)
            output = model(features)
            pred = torch.argmax(output,dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total

    print(f"Accuracy is {accuracy}")
