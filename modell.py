import enum
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, input_shape, actions_count,train_buffer_size=300,lr=0.001):
        super(Net, self).__init__()

        l1=100
        l2=50

        p1=20

        v1=30

        self.train_buffer=[]
        self.train_buffer_size=train_buffer_size
        self.input_size = int(np.prod(input_shape))
        self.body = nn.Sequential(
            nn.Linear(self.input_size, l1),
            nn.ELU(),
            nn.Linear(l1, l2),
            nn.ELU()
        )
        self.policy = nn.Sequential(
            nn.Linear(l2, p1),
            nn.ELU(),
            nn.Linear(p1, actions_count),
            nn.Softmax(dim=1)
        )
        self.value = nn.Sequential(
            nn.Linear(l2, v1),
            nn.ELU(),
            nn.Linear(v1, 1)
        )

        self.History={
                "policy_loss":[],
                "value_loss":[],
                "solution_length":[],
                "t":[],
                "training_loss":[]

        } 

    def forward(self, batch, value_only=False):
        x = batch.view((-1, self.input_size))
        body_out = self.body(x)
        value_out = self.value(body_out)
        if value_only:
            return value_out
        policy_out = self.policy(body_out)
        return policy_out, value_out

    def predict(self,input):
        with torch.no_grad():
            out=self(input)
        return out

    def fit(self,new_sample,epochs=1,lr=0.001):
        self.train_buffer.append(new_sample)
        while len(self.train_buffer)>self.train_buffer_size:
            self.train_buffer.pop(0)

        optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        poLoss=nn.CrossEntropyLoss()
        
        W_total=0
        vl_total=0
        pl_total=0
        sol_len_total=0

        for i in range(epochs):
            loss=0
            for sample in self.train_buffer:
                policy,value=self(sample["x"])
                w=sample["weight"][:,None]

                #print(f"{value = }")
                #print(f"{policy = }")
                #print(f"{w = }")

                #print(f"{sample['value_tgt'][:,None] = }")
                #print(f"{sample['policy_tgt'] = }")
                vl=(value-sample["value_tgt"][:,None])**2
                pl=(policy-sample["policy_tgt"])**2
                pl=torch.sum(pl,dim=-1)[:,None]
                #print(f"{pl = }")
                #print(f"{vl =}")
                er=torch.sum((vl+pl)*w)/torch.sum(w)
                loss+=er

                with torch.no_grad():
                    vl_total +=torch.sum(vl).numpy()
                    pl_total +=torch.sum(pl).numpy()
                    W_total  +=torch.sum(sample["weight"]).numpy()
                    sol_len_total +=len(sample["weight"])

            loss/=len(self.train_buffer)
            loss.backward()
            optimizer.step()
            self.zero_grad()
            optimizer.zero_grad()

        div=epochs*len(self.train_buffer)
        self.History["policy_loss"].append(pl_total/sol_len_total/div)
        self.History["value_loss"].append(vl_total/sol_len_total/div)
        self.History["solution_length"].append(sol_len_total/div)
        self.History["training_loss"].append(loss.detach().numpy())


    def plot_History(self):

        plt.plot(np.array(self.History["solution_length"]),label="solution length")
        #plt.plot(self.History["policy_loss"],label="policy loss not weigthed")
        #plt.plot(self.History["value_loss"],label="value loss not weigted")
        plt.plot(np.array(self.History["training_loss"])/10,label="training_loss (weigthed)")

        plt.legend()
        plt.show()

    def clear_History(self):
        self.History={
                "policy_loss":[],
                "value_loss":[],
                "solution_length":[],
                "t":[],
                "training_loss":[]
        } 