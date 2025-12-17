import numpy as np

x=np.array([1,2,3,4,5])
y =np.array([65,70,75,85,90])
n = len(x)
w,b =0,0  #initial weight and bias

for i in range(400):
    y_cap = w*x+b   #y_predict


    J = np.sum((y - y_cap)**2)/n   # Loss function
    print(f"{i} iteration is {J:.2f}")

        
    dj_dw = (-2*np.sum((y-y_cap)*x))/n   # partial derivative for loss with respect to w
    dj_db = (-2*np.sum(y-y_cap))/n       # partial derivative for loss with respect to b


    # 0.5 is the learning rate

    w = w - 0.05*(dj_dw)    # updating weight
    b = b - 0.05*(dj_db)    # updating bias
print(np.round(y_cap,2))
print("w:",w)
print("b:",b)
    
