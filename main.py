import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import csv

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

# define DataSet
class TrainingDataSet(Dataset):
	def __init__(self, img_data, label_data, transform=None, target_transform=None):
		self.img_datas = torch.Tensor(img_data).to(torch.float32).cuda()
		self.label_datas = torch.Tensor(label_data).type(torch.LongTensor).cuda()
		self.transform = transform
		self.target_transform = target_transform
		print(img_data.dtype)

	def __len__(self):
		return len(self.img_datas)

	def __getitem__(self, idx):
		image = self.img_datas[idx]
		label = self.label_datas[idx]

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)

		return image, label

class TestingDataSet(Dataset):
	def __init__(self, annotations_file, transform=None, target_transform=None):
		test_data = pd.read_csv(annotations_file)
		self.img_datas = torch.Tensor(test_data.values/255).to(torch.float32).cuda()
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.img_datas)

	def __getitem__(self, idx):
		image = self.img_datas[idx]

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)

		return image

class Net(nn.Module):	
	def __init__(self):
		super(Net, self).__init__()
		  
		self.features = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		  
		self.classifier = nn.Sequential(
			nn.Dropout(p = 0.5),
			nn.Linear(64 * 7 * 7, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(p = 0.5),
			nn.Linear(512, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(p = 0.5),
			nn.Linear(512, 10),
		)
		  
		for m in self.features.children():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
		
		for m in self.classifier.children():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform(m.weight)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
				

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		
		return x

def read_train_data(annotations_file):
	train_data = pd.read_csv(annotations_file)
	img_datas = train_data.drop(columns=['label']).values / 255
	label_datas = train_data['label'].values
	return img_datas, label_datas

def train(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	model.train()
	progress = tqdm(total = size)
	for batch, (X, y) in enumerate(dataloader):
		# X = X.to(torch.float32)
		# X, y = X.to(device), y.to(device)

		# Compute prediction error
		pred = model(X.view(-1, 1,28,28))

		optimizer.zero_grad()

		loss = loss_fn(pred, y)

		# Backpropagation
		loss.backward()
		optimizer.step()

		progress.update(len(X))

		if batch % 100 == 0:
			loss, current = loss.item(), (batch + 1) * len(X)
			# print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def validate(dataloader, model, loss_fn):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X = X.to(torch.float32)
			X, y = X.to(device), y.to(device)
			pred = model(X.view(-1, 1,28,28))
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= size
	print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def test(dataloader, model, loss_fn):
	result = [['ImageId','Label']]
	with torch.no_grad():
		for idx, X in enumerate(dataloader):
			pred = model(X.view(-1, 1,28,28)).argmax()
			result.append([idx+1, pred.numpy(force=True)])
	return result




#read csv into DataSet
imgs_data, labels_data = read_train_data('data/train.csv')
x_train, x_validation, y_train, y_validation = train_test_split(imgs_data, labels_data, test_size=0.4, random_state=42)

train_dataset = TrainingDataSet(x_train, y_train, transform = None)
valid_dataset = TrainingDataSet(x_validation, y_validation, transform = None)
test_dataset = TestingDataSet('data/test.csv', transform = None)

# Show data
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
# 	sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
# 	img, label = train_dataset[sample_idx]
# 	img = np.reshape(img,(28,28))
# 	figure.add_subplot(rows, cols, i)
# 	plt.title(label)
# 	plt.axis("off")
# 	plt.imshow(img, cmap="gray")
# plt.show()

#load into dataLoader
batch_size = 64
train_dataloader = DataLoader(train_dataset, 64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, 64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


for X, y in train_dataloader:
	print(f"Shape of X [N, C, H, W]: {X.shape}")
	print(f"Shape of y: {y.shape} {y.dtype}")
	break

# Get cpu, gpu or mps device for training.
device = (
	"cuda"
	if torch.cuda.is_available()
	else "mps"
	if torch.backends.mps.is_available()
	else "cpu"
)
print(f"Using {device} device")

model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

max_epoch = 100

for epoch in range(max_epoch):
	print(f'Epochs: {epoch} / {max_epoch}')
	train(train_dataloader, model, loss_fn, optimizer)
	validate(valid_dataloader, model, loss_fn)
	scheduler1.step()
	scheduler2.step()
	print('-------')

test_result = test(test_dataloader, model, loss_fn)
with open('data/result.csv',mode='w', newline='') as file:
	writer = csv.writer(file)  # 创建CSV写入器
	writer.writerows(test_result)  # 写入数据行










