import numpy as np 
from keras import layers
from keras import Input
from keras.models import Model
from keras.models import load_model
from keras import regularizers
from keras import optimizers
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_splits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
ssc = StandardScaler()
ssc=MinMaxScaler()
ssc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')

data = load_breast_cancer()
train_data = data.data
y = data.target

oe = OneHotEncoder(handle_unknown='ignore', sparse=False)
train_labels= oe.fit_transform(y.reshape(-1,1))
train_labels = y
#digitizer followed by MinMaxScaler
train_data = ssc.fit_transform(train_data)
ssc=MinMaxScaler()
train_data = ssc.fit_transform(train_data)
s = [1,2,1,2]
a = np.array(s)
print(type(a))
class DQN():
    def __init__(self, input_size,output_size):
        self.model = self.create_model(input_size, output_size)
        self.target_model = self.create_model(input_size, output_size)
        pass
    def create_model(self, input_size, output_size):
       
        input_tensor = Input(shape=(input_size,))
        x = layers.Dense(32,activation='relu')(input_tensor)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(8, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        output_tensor = layers.Dense(output_size, activation='sigmoid')(x)
        model = Model(inputs=input_tensor, outputs=output_tensor)
        #print (model.summary())
        rm = optimizers.RMSprop(learning_rate=0.001, momentum=1e-5) #'rmsprop'
        model.compile(optimizer=rm,
                        loss='binary_crossentropy', metrics=['acc'])
        return model


train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.3, random_state=True)
dqn = DQN(train_data.shape[1], 1)
for ep in range(128) :
    dqn.model.fit(train_data, train_labels, epochs=1, batch_size=32)     
    e=dqn.model.evaluate(test_data, test_labels, batch_size=32)
    if (ep + 1) % 10 == 0 :
        dqn.target_model.set_weights(dqn.model.get_weights())
    #d = dqn.target_model.evaluate(test_data, test_labels, batch_size=32)

    #print(" e at ", ep, " ", e[1], " d ", d[1])
dqn.target_model.set_weights(dqn.model.get_weights())
e=dqn.model.evaluate(test_data, test_labels, batch_size=32)
d = dqn.target_model.evaluate(test_data, test_labels, batch_size=32)

print(' Final comparison ', e, " target = ", d)

results = dqn.model.predict(train_data, batch_size=32)
r = np.where( results > 0.5,1,0).reshape(-1,)
a= accuracy_score(train_labels, r )
print(" accuracy ", a)
tn, fp, fn, tp = confusion_matrix(train_labels, r).ravel()
print( " train tn ,", tn, " fp ", fp, " fn ", fn, " tp ", tp )
results = dqn.model.predict(test_data, batch_size=32)
r = np.where( results > 0.5,1,0).reshape(-1,)
a= accuracy_score(test_labels,r )
print(" test accuracy ", a)
tn, fp, fn, tp = confusion_matrix(test_labels, r).ravel()
print( " test tn ,", tn, " fp ", fp, " fn ", fn, " tp ", tp )
# dqn.model.save('c:/workdir/models')
# m2= load_model('c:/workdir/models')

# e = m2.evaluate(test_data, test_labels, batch_size=32)
# e=m2(test_data[0].reshape(1,-1), training=False)

# print(e.numpy())
# e=m2.predict(test_data[0].reshape(1,-1), batch_size=1)
# print(e)