from load_data import load_
from config import train_path , test_path 
from dataset import train_config , test_config
from loader import train_load , test_load
from model import Model
from training import train_model
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import joblib
from device import device

X_train , X_test , y_train , y_test = load_(train_path,test_path)
print("Data loading Done!")
train_set = train_config(X_train , y_train )
test_set = test_config(X_test , y_test)
print("Data Configration Done")

train_loader = train_load(train_set)
test_loader = test_load(test_set)
print("Done making DataLoaders")
#Data loading DOne

#Now training 
model = Model()

optimizer = Adam(model.parameters(),lr=0.01)
criterion = CrossEntropyLoss()

train_model(model,train_loader,criterion,optimizer,device,epochs=20)









