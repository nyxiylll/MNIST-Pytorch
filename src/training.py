import torch 
from model import Model
from device import device
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import joblib
from tqdm import tqdm


def train_model(model,train_loader,criterion,optimizer,device,epochs=20):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader,desc=f"Epoch [{epoch+1}/{epochs}]",leave=True)

        for features , labels in loop:
            features = features.to(device).float()
            labels = labels.to(device)
            print("FEATURE DTYPE:", features.dtype)
            print("MODEL WEIGHT DTYPE:", next(model.parameters()).dtype)
            optimizer.zero_grad()

            output = model(features)
            loss = criterion(output,labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}] | Avg loss : {epoch_loss:.4f}")

    torch.save(model.state_dict(),"../Model/model.pth")
    print("Model saved successfully")


























def train(model,train_loader,criterion,optimizer):
    model.train()
    for epoch in range(100):
        print(f"Epoch : {epoch}")
        for features , labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
    print("Model trained successfully!")
    joblib.dump(model,"../Model/model.pkl")
def load_model():
    model = joblib.load("../Model/model.pkl")
    return model

if __name__ == "__main__":
    pass