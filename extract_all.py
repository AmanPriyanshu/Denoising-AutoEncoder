import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from models import Encoder, Decoder
from tqdm import tqdm

def add_noise(inputs,noise_factor=0.3):
     noisy = inputs+torch.randn_like(inputs) * noise_factor
     noisy = torch.clip(noisy,0.,1.)
     return noisy

data_dir = 'dataset'
noise_factor = 0.8
encoder = Encoder(encoded_space_dim=4,fc2_input_dim=128)
encoder.load_state_dict(torch.load('./model/encoder.pt'))
decoder = Decoder(encoded_space_dim=4,fc2_input_dim=128)
decoder.load_state_dict(torch.load('./model/decoder.pt'))

test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

test_transform = transforms.Compose([
transforms.ToTensor(),
])

test_dataset.transform = test_transform
encoder.eval()
decoder.eval()

test_dataset = torch.stack([i[0] for i in tqdm(test_dataset)])
for idx, img in tqdm(enumerate(test_dataset), total=len(test_dataset)):
	img = img.unsqueeze(0)
	image_noisy = add_noise(img,noise_factor)
	decoded_data = decoder(encoder(image_noisy)).detach()
	plt.cla()
	plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
	plt.axis('off')
	plt.savefig("./images/original/"+"0"*(5-len(str(idx)))+str(idx)+".png", bbox_inches='tight')
	plt.cla()
	plt.imshow(image_noisy.cpu().squeeze().numpy(), cmap='gist_gray')
	plt.axis('off')
	plt.savefig("./images/noisy/"+"0"*(5-len(str(idx)))+str(idx)+".png", bbox_inches='tight')
	plt.cla()
	plt.imshow(decoded_data.cpu().squeeze().numpy(), cmap='gist_gray')
	plt.axis('off')
	plt.savefig("./images/denoised/"+"0"*(5-len(str(idx)))+str(idx)+".png", bbox_inches='tight')