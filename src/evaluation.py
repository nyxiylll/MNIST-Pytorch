import joblib
from load_data import load_
from config import train_path , test_path
from dataset import test_config
from loader import test_load
from eval import evaluation
from device import device
import torch 
from model import Model

model = Model()
model.load_state_dict(torch.load("../Model/model.pth"))
model.to(device)
_ , X_test , _ , y_test = load_(train_path,test_path)

test_set = test_config(X_test,y_test)

test_loader = test_load(test_set)

evaluation(model,test_loader,device)

