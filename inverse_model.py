import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import ObjPushDataset
from push_env import PushingEnv


class InverseModelNet(torch.nn.Module):
    def __init__(self):
        super(InverseModelNet, self).__init__()
        self.fc1 = nn.Linear(4, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class InverseModel:
    def __init__(self):
        self.net = InverseModelNet()

        train_dir = 'push_dataset/train'
        test_dir = 'push_dataset/test'
        bsize = 64

        self.train_loader = DataLoader(ObjPushDataset(train_dir), batch_size=bsize, shuffle=True)
        self.valid_loader = DataLoader(ObjPushDataset(test_dir), batch_size=bsize, shuffle=True)  

    def train(self, num_epochs=2):
        criterion = nn.MSELoss()
        #optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        optimizer=optim.Adadelta(self.net.parameters())

        valid_losses = np.zeros(num_epochs+1)
        train_losses = np.zeros(num_epochs+1)

        train_loss, valid_loss = self.eval(criterion)
        print('epoch 0: train loss ', train_loss, ', validation loss ', valid_loss)
        train_losses[0] = train_loss
        valid_losses[0] = valid_loss

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                obj1 = data['obj1']
                obj2 = data['obj2']
                push = data['push']
                inputs = torch.cat((obj1, obj2), axis=1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs.float())
                loss = criterion(outputs, push)
                loss.backward()
                optimizer.step()

                # print statistics
                #running_loss += loss.item()
                #if i % 10 == 9:    # print every 2000 mini-batches
                #    print('[%d, %5d] loss: %.6f' %
                #          (epoch + 1, i + 1, running_loss / 2000))
                #    running_loss = 0.0

            # evaluate loss
            train_loss, valid_loss = self.eval(criterion)
            print('epoch ', epoch+1, ': train loss ', train_loss, ', validation loss ', valid_loss)
            train_losses[epoch+1] = train_loss
            valid_losses[epoch+1] = valid_loss

        print('Finished Training')
        return train_losses, valid_losses

    def eval(self, criterion):
        self.net.eval()
        train_loss = 0
        valid_loss = 0

        for data in self.train_loader:
            obj1 = data['obj1']
            obj2 = data['obj2']
            push = data['push']
            inputs = torch.cat((obj1, obj2), axis=1)
            output = self.net(inputs.float())
            loss = criterion(output,push)
            train_loss += loss.item()

        for data in self.valid_loader:
            obj1 = data['obj1']
            obj2 = data['obj2']
            push = data['push']
            inputs = torch.cat((obj1, obj2), axis=1)
            output = self.net(inputs.float())
            loss = criterion(output,push)
            valid_loss += loss.item()

        train_loss = train_loss/len(self.train_loader.dataset)
        valid_loss = valid_loss/len(self.valid_loader.dataset)

        return train_loss, valid_loss

    def infer(self, init_obj, goal_obj):
        x = torch.cat((init_obj, goal_obj), axis=1)
        return self.net(x)

    def save(self, PATH):
        torch.save(self.net.state_dict(), PATH)

    def load(self, PATH):
        self.net.load_state_dict(torch.load(PATH))
        self.net.eval()


def plan_CEM(model, env):
    push_ang = np.random.random() * np.pi * 2 - np.pi
    push_len = np.random.random() * self.push_len_range + self.push_len_min
    


if __name__ == "__main__":
    model = InverseModel()
    num_epochs=30
    train_losses, valid_losses = model.train(num_epochs=num_epochs)
    model.save(PATH="inverse_model_save.pt")

    plt.figure()
    plt.plot(range(num_epochs+1), train_losses)
    plt.plot(range(num_epochs+1), valid_losses)
    plt.title("Inverse Model Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.ylim(0, train_losses[1] * 2.0)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()
    plt.savefig("inverse_model_training.png")

    env = PushingEnv(ifRender=False)
    num_trials = 10
    errors = np.zeros(num_trials)
    # save one push
    errors[0] = env.plan_inverse_model(model, img_save_name="inverse", seed=0)
    print("test loss:", errors[0])
    # try 10 random seeds
    for seed in range(1,10):
        errors[seed] = env.plan_inverse_model(model, seed=seed)
        print("test loss:", errors[seed])
    
    print("average loss:", np.mean(errors))
