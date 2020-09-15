
# coding: utf-8
# author: Tianyou Xiao (txiao3) & Ziyu Song (zsong10)

# In[1]:


import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time
from torch.autograd import Variable


# In[2]:


LR = 0.01 # learning rate set to 0.01
BATCH_SIZE = 64 # batch size set  to 64
EPOCH = 250 # run for 250 epochs


# In[3]:


df = pd.read_csv('./ml-latest-small/ratings.csv')
df.head()


# In[4]:


from sklearn import model_selection as ms
#75:25 train:test
train_set, test_set = ms.train_test_split(df, test_size=0.25)
len(train_set), len(test_set)


# In[5]:


training_set = np.array(train_set, dtype = 'int')
testing_set = np.array(test_set, dtype = 'int')


# In[6]:


num_users = df.userId.unique().shape[0]
num_items = df.movieId.unique().shape[0]
num_users, num_items


# In[7]:


movie_movieId = df.movieId.unique().tolist()
movie_movieId.sort()
d = dict()
for i in range(0, len(movie_movieId)):
    d[movie_movieId[i]] = i


# In[8]:


def user_item_matrix(data):
    # load ratings into 2d numpy array
    ratings = np.zeros((num_users, num_items))
    for row in data.itertuples():
        ratings[row[1]-1, d[row[2]]] = row[3]
    return ratings


# In[9]:


ratings = user_item_matrix(df)
train = user_item_matrix(train_set)
test = user_item_matrix(test_set)


# In[10]:


train = torch.FloatTensor(train)
input = Variable(train).unsqueeze(0)
input.shape


# In[11]:


train = torch.FloatTensor(train)
test = torch.FloatTensor(test)
input = Variable(train).unsqueeze(0)
target = input
torch_dataset = Data.TensorDataset(input, target)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)


# In[12]:


# define the neural network's structure
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_items, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 50)
        self.fc4 = nn.Linear(50, num_items)
        self.activation = nn.Sigmoid()
        self.activation_t = nn.Tanh()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation_t(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


# In[13]:


net = Net()
loss_func = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))


# In[14]:


# training process
loss_his = []
st = time.time()
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(loader): 
        # for each training step
        output = net(b_x)  
        output[target == 0] = 0
        # get output for every net
        loss = loss_func(output, b_y)  # compute loss for every net
        opt.zero_grad()                # clear gradients for next train
        loss.backward()                # backpropagation, compute gradients
        opt.step()                     # apply gradients
        loss_his.append(loss.data.numpy())     # loss recoder
        print('Epoch: ', epoch, '| Step: ', step, '| train loss: ', loss.data.numpy())

print('Runtime: ' + str(time.time()-st) + 'seconds')


# In[15]:


# training loss visualization
plt.plot(range(EPOCH), loss_his)
plt.title('training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[16]:


test_var = Variable(test).unsqueeze(0)
out = net(test_var)
out


# In[17]:


out = out.detach().numpy()[0,:,:]


# In[18]:


# evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_mse(pred, actual):
    # Ignore zero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(actual, pred)
def get_mae(pred, actual):
    # Ignore zero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(actual, pred)

test = user_item_matrix(test_set)
print('Autoencoder RMSE: ' + str(math.sqrt(get_mse(out, test))))
print('Autoencoder MAE: ' + str(get_mae(out, test)))

