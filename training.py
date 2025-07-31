import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    'DATA_DIR': 'data',
    'MODEL_DIR': 'models',
    'BATCH_SIZE': 32,
    'EPOCHS': 10,
    'LEARNING_RATE': 0.001,
    'LOG_INTERVAL': 100,
}

class SensorDataset(Dataset):
    def __init__(self, data_dir: str, transform: transforms.Compose):
        self.data_dir = data_dir
        self.transform = transform
        self.data = pd.read_csv(os.path.join(data_dir, 'data.csv'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.data_dir, self.data.iloc[idx, 0])
        image = self.transform(image_path)
        label = self.data.iloc[idx, 1]
        return image, label

class FlowTheory(nn.Module):
    def __init__(self):
        super(FlowTheory, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VelocityThreshold(nn.Module):
    def __init__(self):
        super(VelocityThreshold, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TrainingPipeline:
    def __init__(self, data_dir: str, model_dir: str):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FlowTheory()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG['LEARNING_RATE'])
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        logger.info('Training model...')
        start_time = time.time()
        dataset = SensorDataset(self.data_dir, transforms.Compose([transforms.ToTensor()]))
        data_loader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
        for epoch in range(CONFIG['EPOCHS']):
            for batch in data_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if epoch % CONFIG['LOG_INTERVAL'] == 0:
                    logger.info(f'Epoch {epoch+1}, Loss: {loss.item()}')
        end_time = time.time()
        logger.info(f'Training completed in {end_time - start_time} seconds')

    def evaluate(self):
        logger.info('Evaluating model...')
        dataset = SensorDataset(self.data_dir, transforms.Compose([transforms.ToTensor()]))
        data_loader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in data_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        logger.info(f'Accuracy: {accuracy:.4f}')

    def save_model(self):
        logger.info('Saving model...')
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'model.pth'))

def main():
    data_dir = CONFIG['DATA_DIR']
    model_dir = CONFIG['MODEL_DIR']
    pipeline = TrainingPipeline(data_dir, model_dir)
    pipeline.train()
    pipeline.evaluate()
    pipeline.save_model()

if __name__ == '__main__':
    main()