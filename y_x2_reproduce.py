
#IMPORTS
import numpy as np  
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense,LeakyReLU
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

#MAXIMIZE NUMERICAL ACCURACY
K.set_floatx('float64')

#DATA
X_train= K.expand_dims(np.linspace(-2.5,2.5,5000),-1)
y_train=X_train**2

#NEURAL NETWORK DEFINITION
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(2, ))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(1))
model.load_weights('bestsquare.h5')



#TREE DEFINITION
def tree(x):
    if x<-1.163842797279:
        y=-3.667660713196*x+ -3.220932960510
    else:
        if x<0.323221951723:
            if x<0.112146496773:
                y=-1.003839254379*x+ -0.120662927628
            else:
                y=2.373108386993*x+ -0.499375820160
        else:
            if x>0.998715341091:
                y=3.470931053162*x+ -2.829265117645
            else:
                y=0.547069132328*x+ 0.090840339661
                
    return y


#NEURAL NETWORK RESULTS
y_tree=[]
for x in X_train:
    y_tree.append(tree(x.numpy()))
    
#TREE RESULTS
y_neuralnetwork=model.predict(X_train)

#MAXIMUM SQUARED ERROR
MAXSE=np.max(np.square(y_tree-y_neuralnetwork))
print('maximum squard error: ',MAXSE)

#EQUIVALENCE
plt.plot(y_tree,label='Decision Tree')
plt.plot(y_neuralnetwork,label='Neural Network')
plt.title('maximum squard error: '+str(MAXSE))
plt.legend()



