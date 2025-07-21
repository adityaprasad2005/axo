class demo:
    def __init__(self, a):
        self.attr_a = a
    
    def change_a(self):
        a_hash = self.attr_a 

        a_hash[0]= 1000
    

c = demo([0,1,1,2])
c.change_a() 
print(c.attr_a)