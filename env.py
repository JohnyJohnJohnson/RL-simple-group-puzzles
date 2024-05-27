import numpy as np
import torch 
from groups import VecGroup, Z12, Z6



class Env:

    vG=VecGroup(Z6,3)
    Actions={
        0:vG(3,2,1),
        1:vG(1,1,1),
        2:vG(0,1,0),
        3:vG(0,0,1)
        }


    Inv_Actions={
        0:vG(3,4,5),
        1:vG(5,5,5),
        2:vG(0,5,0),
        3:vG(0,0,5)
        }
    
    N=len(Actions)

    def __init__(self,state=vG(0,0,0)):
        self.state=state
        pass

    def apply_action(self,numb):
        return Env(self.state+Env.Actions[numb])
        

    def is_goal(self):
        return self.state==Env.vG(0,0,0)
    
    def reward(self):
        if self.is_goal():
              return 1
        else:
              return -1 
    

    def to_tensor(self):
        res =torch.zeros(3*6)

        for i,s in enumerate(self.state.state):
            res[6*i+s.state]=1
        return res

    def scramble(self,n):
        s=self.state
        action_list=np.random.choice(Env.N,size=n)
        for choice in action_list:
            s=s+Env.Inv_Actions[choice]
        self.state=s


    def __repr__(self):
        return f"Env(vG{self.state})"

    
    ### Visualization , kann weg ###

    def to_tex(self):

        beg=r"$("
        end=r"$)"
        st =[repr(s) for s in self.state.state]
        body=",".join(st)
        return beg + body + end