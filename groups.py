class Z12:

    def __init__(self,x):
        if  isinstance(x,Z12):
            self.state=x.state
        else:    
            self.state=x%12

    def __add__(self,other):
        res=self.state + other.state
        res%=12
        return Z12(res)

    def __repr__(self):
        return str(self.state)
    def __eq__(self,other):
        return self.state == other.state

    def __hash__(self):
        return self.state

class Z6:

    def __init__(self,x):
        if  isinstance(x,Z12):
            self.state=x.state
        else:    
            self.state=x%6

    def __add__(self,other):
        res=self.state + other.state
        res%=6
        return Z12(res)

    def __repr__(self):
        return str(self.state)
    def __eq__(self,other):
        return self.state == other.state

    def __hash__(self):
        return self.state

def VecGroup(G,n):

    class New_G:

        def __init__(self,*args):
            assert len(args)==n
            self.state=tuple([G(arg) for arg in args])

        def __add__(self,other):
            new_state=[ a+b for a,b in zip(self.state,other.state) ]
            new_state=tuple(new_state)
            return New_G(*new_state)
            
        def __eq__(self,other):
            for a,b in zip(self.state,other.state):
                if a!=b:
                    return False
            return True

        def __hash__(self):
            return hash(self.state)

        def __repr__(self):
            return repr(self.state)

    return New_G 