from sklearn.metrics.pairwise import cosine_similarity
import os

train_dataloader = DataLoader(datasets, shuffle=True, batch_size=512)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
loss_function = nn.MSELoss()
model = AutoEncoder()

PATH = os.listdir("/content/drive/MyDrive/HAN2HAN/CharacterClustering")[-1]
print(PATH)
PATH = "/content/drive/MyDrive/HAN2HAN/CharacterClustering/" + PATH
model.load_state_dict(torch.load(PATH))

model.to(device)
optimizer = torch.optim.AdamW(model.parameters())
print("device :",device)

