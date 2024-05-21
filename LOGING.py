

# adds "print input and output" to a function

def LOG_IT(fnk):
    def new_funk(*args,**kwargs):

        res=fnk(*args,**kwargs)

        print(f"""
#### funktion: {fnk.__name__} ####

with arguments:

{args}

and kwargs:

{kwargs}

returns:

{res}

{"#"*100}
""")
        return res
    return new_funk
    


def DONT_LOG_IT(fnk):
    return (fnk)

