import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import time

class TrainerClass:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, scheduler, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

    def train(self, epochs):
        start_time = time.time()
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0
            total_train_samples = 0
            self.model.train(True)
            for inputs, targets in self.train_dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_size = inputs.size(0)
                total_train_loss += loss.item() * batch_size
                total_train_samples += batch_size

            avg_train_loss = total_train_loss / total_train_samples
            self.model.train(False)
            val_loss, accur = self.evaluate(self.model, self.val_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss_train: {avg_train_loss:.6f}, Loss_val: {val_loss:.6f}, Accuracy: {accur} ")

            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"LR after step: {current_lr:.6f}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total train time: {total_time:.2f} s., "
              f"Average epoch time: {total_time / epochs:.2f} s.")
        return self.model


    def evaluate(self, model, dataloader):
        model.eval()
        total_loss = 0
        total_samples = 0
        num_correct = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = model(inputs)
                loss = self.criterion(logits, targets)

                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size  # аккумулируем по всем объектам
                total_samples += batch_size

                y_pred = torch.argmax(logits, dim=1)
                num_correct += torch.sum(y_pred == targets)



        avg_loss = total_loss / total_samples
        accur = num_correct / total_samples
        return avg_loss, accur



