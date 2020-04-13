import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import scipy

from dataset import ObjPushDataset
from push_env import PushingEnv


class ForwardModelNet(torch.nn.Module):
    def __init__(self):
        super(ForwardModelNet, self).__init__()
        self.fc1 = nn.Linear(6, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ForwardModel:
    def __init__(self):
        self.net = ForwardModelNet()

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
                inputs = torch.cat((obj1, push.double()), axis=1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs.float())
                loss = criterion(outputs.double(), obj2)
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
            inputs = torch.cat((obj1, push.double()), axis=1)
            output = self.net(inputs.float())
            loss = criterion(output,obj2)
            train_loss += loss.item()

        for data in self.valid_loader:
            obj1 = data['obj1']
            obj2 = data['obj2']
            push = data['push']
            inputs = torch.cat((obj1, push.double()), axis=1)
            output = self.net(inputs.float())
            loss = criterion(output,obj2)
            valid_loss += loss.item()

        train_loss = train_loss/len(self.train_loader.dataset)
        valid_loss = valid_loss/len(self.valid_loader.dataset)

        return train_loss, valid_loss

    def infer_fwd(self, init_obj, push):
        x = torch.cat((init_obj, push), axis=1)
        return self.net(x)

    def infer(self, init_obj, goal_obj, env):

        num_samples = 100
        best_action = None
        best_loss = float('inf')

        for i in range(num_samples):
            #push_ang, push_len = sample_ang_len(mu, sigma)
            start_x, start_y, end_x, end_y = env.sample_push(init_obj[0], init_obj[1])
            init_obj = np.array([obj_x, obj_y])
            init_obj = torch.FloatTensor(init_obj).unsqueeze(0)
            push = np.array([start_x, start_y, end_x, end_y])
            push = torch.FloatTensor(push).unsqueeze(0)

            final_obj_pred = self.infer_fwd(init_obj, push)
            final_obj_pred = goal_obj.numpy().flatten()
            goal_obj = goal_obj.numpy().flatten()

            loss = np.linalg.norm(goal_obj - final_obj_pred)

            if loss < best_loss:
                best_loss = loss
                best_action = np.array([start_x, start_y, end_x, end_y])

        return best_action

    def save(self, PATH):
        torch.save(self.net.state_dict(), PATH)

    def load(self, PATH):
        self.net.load_state_dict(torch.load(PATH))
        self.net.eval()


def sample_ang_len(mu_ang, kappa_ang, mu_len, sigma_len):
    push_len_min = 0.06 # 0.06 ensures no contact with box empiracally
    push_len_range = 0.04
    push_len_max = push_len_min + push_len_range

    push_ang = np.random.vonmises(mu_ang, kappa_ang)
    push_len = scipy.stats.truncnorm((push_min - mu_len)/sigma_len, (push_max - mu_len)/sigma_len, loc=mu_len, scale=sigma_len)

    return push_ang, push_len




if __name__ == "__main__":
    model = ForwardModel(ifrender=True)
    num_epochs=40
    train_losses, valid_losses = model.train(num_epochs=num_epochs)
    model.save(PATH="forward_model_save.pt")

    plt.figure()
    plt.plot(range(num_epochs+1), train_losses)
    plt.plot(range(num_epochs+1), valid_losses)
    plt.title("Forward Model Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()
    plt.savefig("forward_model_training.png")

    env = PushingEnv(ifRender=False)
    num_trials = 10
    errors = np.zeros(num_trials)
    # save one push
    errors[0] = env.plan_forward_model(model, img_save_name="forward", seed=0)
    print("test loss:", errors[0])
    # try 10 random seeds
    for seed in range(1,10):
        errors[seed] = env.plan_forward_model(model, seed=seed)
        print("test loss:", errors[seed])
    
    print("average loss:", np.mean(errors))