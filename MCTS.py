import numpy as np
from numpy import array
import torch
from torch import tensor

from LOGING import DONT_LOG_IT as LOG
import matplotlib.pyplot as plt
import matplotlib

import networkx as nx

class MCT:  # global tree responsibilities

    @LOG
    def __init__(self,modell,Env,c_param=1,viLoss_param=1,state=None, DEBUG=False,k_scramble=0,policy_trust=0,abelsch=False):
        self.DEBUG=DEBUG
        self.nodeCnt=0
        self.policy_trust=policy_trust
        self.viLoss_param=viLoss_param
        self.modell=modell
        self.abelsch=abelsch
        self.Env=Env
        if state is not None:
            self.root=MCT_element(self,state=state)
        else:
            self.root=MCT_element(self,state=Env())
        
        self.root.state.scramble(k_scramble)

        self.root_state=state
        self.cur_ptr=self.root
        self.c_param=c_param
        # Display properties
        self.G=None
        self.edgeList=None
        self.nodeLabels=None
        pass
    
    @LOG
    def totalSimCnt(self):
        return self.root.simCnt

    @LOG
    def update(self):
        selected=self.root.descent()
        solved_state=selected.update()
        selected.update_backprop()
        return solved_state

    def display_tree(self,save_as=None,size_factor=1,fontMultiplyer=1):
        size_factor*=10
        self.G=nx.DiGraph()
        self.edgeList=[]
        self.nodeLabels={}
        self.colors={}
        self.sizes={}

        self.root._addToDisplay()
        #self.G.add_nodes_from(self.nodeLabels)
        #self.G.add_edges_from(self.edgeList)
        #pos = nx.nx_agraph.graphviz_layout(self.G, prog="twopi") # requires pygraphviz
        pos=nx.planar_layout(self.G)
        node_colors = [self.colors[key] for key in self.G.nodes.keys()]
        node_sizes = [np.sqrt(self.sizes[key])*size_factor for key in self.G.nodes.keys()]
        nx.draw(self.G,pos=pos,node_color=node_colors,node_size=node_sizes,labels=self.nodeLabels,font_size=2*fontMultiplyer)
        nx.draw_networkx_edge_labels(self.G,pos=pos,font_size=2*fontMultiplyer)
        if save_as is not None:
            plt.savefig(save_as)
        plt.show()

class MCT_element:  # local tree responsibilities

    list_to_tensor=lambda lst: tensor(array(lst),dtype=torch.float32)

    @LOG
    def __init__(self,mct:MCT,from_element=None,action=None,state=None):


        self.parent=from_element
        self.derivedBy=action

        self.id= mct.nodeCnt
        mct.nodeCnt+=1

        self.mct=mct # linked to tree

        if state is None and from_element is not None:
            self.state=mct.Env.apply_action(from_element.state,action)
        elif state is None:
            self.state=mct.Env()
        else:
            self.state=state

        # DEBUG print toogle
        if mct.DEBUG:
            self.DB=self._DB_ON
        else:
            self.DB=self._DB_OFF

        self.childs=[]

        policy,value=self.mct.modell.predict(self.state.to_tensor())
        self.predicted_value=float(value[0][0])
        self.DB(f"{policy = } \n\n {value = }")

        mem={
            "W":{a:min(float(value[0][0].numpy()),-1.0) for a in self.mct.Env.Actions},
            "N":{a:0 for a in self.mct.Env.Actions},
            "P":{a:float(policy[0][a].numpy()) for a in self.mct.Env.Actions},
            "L":{a:0 for a in self.mct.Env.Actions}  
        }
        self.mem=mem

        # default color value for networkx display
        self.colorVal=0
        self.DB(f"in MCT_element.__init__")
        self.DB(f"{self.state}")


        pass


    def _DB_ON(*messages):
        print(*messages)

    
    def _DB_OFF(*mesages):
        pass



    def W(self,a):
        assert max(self.mem["W"].values())<=0
        return self.mem["W"][a]

    def L(self,a):
        return 0 # virtuall loss not needed
        return self.mem["L"][a]

    def N(self,a):
        return self.mem["N"][a]
        pass
    
    def P(self,a):
        return self.mem["P"][a]
        pass
    
    def U(self,a):
        self.DB(f"in MCT_element.U")
        res=np.sqrt(sum([self.N(i) for i in self.mct.Env.Actions if i !=a ]))*self.mct.c_param
        res*= self.P(a) 
        res/=(1+self.N(a))
        res+=self.mct.policy_trust*self.P(a)

        return res
    
    def Q(self,a):
        return self.W(a)-self.L(a)
        
    
    def A(self):            
        max_val=-np.inf
        argmax=None
        for a in self.mct.Env.Actions:
            cur_val=self.U(a)+self.Q(a)
            if  cur_val>max_val:
                argmax=a
                max_val=cur_val
        return argmax


    def descent(self):
        if self.is_leaf():
            return self
        A=self.A()
        self.mem["L"][A]+=1
        maxChild=self.childs[A]
        return maxChild.descent()

    def get_action_chain(self):
        if self.parent is not None:
            return self.parent.get_action_chain().append(self.derivedBy)
        else:
            return []

    def get_state(self):
        state= self.state
        self.DB(f" in MCT_element.get_state : {state = }")
        return state
        pass
    
    
    def is_goal(self):
        return self.mct.Env.is_goal(self.get_state())
        pass
    
    def is_root(self):
        if self.parent is None:
            return True
        else:
            return False
        pass

    def is_leaf(self):
        if self.childs:
            return False
        else:
            return True

    def update(self):
        solved_state=None
        for i,action in self.mct.Env.Actions.items():

            child=MCT_element(mct=self.mct,from_element=self,action=i)
            self.childs.append(child)
            self.mem["W"][i]=min(child.predicted_value,0)
            if child.is_goal():
                solved_state=child
        return solved_state 

    
    def update_backprop(self):
        if not self.is_root():
            self.parent.mem["W"][self.derivedBy]=  max(self.mem["W"].values())-1
            self.parent.mem["N"][self.derivedBy]+=1
            self.parent.mem["L"][self.derivedBy]-=self.mct.viLoss_param

            self.parent.update_backprop()


    def get_value_and_policy_target(self):
        value_target=-np.inf
        argmax=None
        policy_target=np.zeros(self.mct.Env.N)
        for a in self.mct.Env.Actions:
            child=self.childs[a]
            cur_val=child.predicted_value + child.state.reward()
            if cur_val>value_target:
                value_target=cur_val
                argmax=a
        policy_target[argmax]=1
        return value_target,policy_target
        pass

    def _extract_train_data_list(self,_Dxi=0):
        #changed
        if not self.is_leaf():
            value_target,policy_target=self.get_value_and_policy_target()
            res=[
                ( 
                    self.state.to_tensor(),
                    -_Dxi,
                    policy_target
                )
            ]

        else:
            res=[]
        if self.is_root():
            return res
        else:
           return res+self.parent._extract_train_data_list(_Dxi=_Dxi+1) 

    def extract_train_data(self):
        res={
            "x":[],
            "value_tgt":[],
            "policy_tgt":[],
            "weight":[]
        }

        lst= self._extract_train_data_list()

        for i,(x,value_tgt,policy_tgt) in enumerate(lst):
            res["policy_tgt"].append(policy_tgt)
            res["value_tgt"].append(value_tgt)
            res["x"].append(x)
            res["weight"].append(1/(i+1))

        if self.mct.abelsch:
            # if abelsch order doesnt matter
            policy_tgt=res["policy_tgt"]
            policy_tgt= np.array(policy_tgt)
            new_policy=0
            new_policy_tgt=[]
            for policy in reversed(policy_tgt):
                new_policy+=policy
                new_policy_tgt.append(new_policy/new_policy.sum())
            res["policy_tgt"]=list(reversed(new_policy_tgt))


        res["policy_tgt"]=MCT_element.list_to_tensor(res["policy_tgt"]) 
        res["value_tgt"]=MCT_element.list_to_tensor(res["value_tgt"])
        res["x"]=MCT_element.list_to_tensor(res["x"])
        res["weight"]=MCT_element.list_to_tensor(res["weight"])

        return res

    def _addToDisplay(self):

        def rnd(exp):
            return f"{exp: 0.2f}"

        if self.is_goal():
            color="#AA4444"
        elif self.is_leaf():
            color="#44FFff"
        elif self.is_root():
            color="#4444FF"
        else:
            color="#AA9999"

        self.mct.colors.update({self.id:color})

        if not self.is_root():
            size= self.parent.mem["N"][self.derivedBy] +2
        else:
            size=sum(self.mem["N"])+3
        
        self.mct.sizes.update({self.id:size})
        self.mct.nodeLabels.update(  { self.id:f"node {self.id} \n {self.state.to_tex()}\n val_pred={self.predicted_value:0.2f}" } )
        self.mct.G.add_node(self.id)
        for a,child in enumerate(self.childs):
            self.mct.edgeList.append((self.id,child.id))
            self.mct.G.add_edge(self.id,child.id,P=rnd(self.P(a)),W=rnd(self.W(a)),A=self.mct.Env.Actions[a],DEC=f"{self.Q(a)+self.U(a):0.2f}")
            child._addToDisplay()

        pass

    