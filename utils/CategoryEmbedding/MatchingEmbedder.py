from sklearn.metrics.pairwise import cosine_similarity
import os

def knock_the_door(embed_word):
  dic = {}
  for i in range(2402):
    find_cos = cosine_similarity(cos_embed[i:i+1],embed_word)
    best_cos = np.argmax(find_cos)
    print(common_han[i],char_labels[best_cos], best_cos, find_cos[0][best_cos])
    dic[common_han[i]] = best_cos
  return dic

char_dictionary = knock_the_door(char_embedding)





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
