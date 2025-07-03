import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboard import summary

from utils import TrainerClass
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import Model_1k
from models import Model_5k
from utils import create_data_loader
from torchsummary import summary



train_loader, val_loader, test_loader = create_data_loader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Model_5k().to(device)
model2 = Model_1k().to(device)
summary(model, (3, 32, 32))
summary(model2, (3, 32, 32))
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.CrossEntropyLoss()
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
#
# Trainer = TrainerClass(model, train_loader, val_loader, optimizer, loss_fn, scheduler, device)
# model = Trainer.train(epochs=200)
#
# _, test_accuracy = Trainer.evaluate(model, test_loader)
# print('Test accuracy is', test_accuracy)