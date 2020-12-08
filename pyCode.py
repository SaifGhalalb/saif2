# Imports
import numpy as np
import pandas as pd
from sklearn import svm
from flask import request
from flask import Flask

app= Flask(__name__)


num_samples_total = 32

# Generate data
dataset=pd.read_csv('\home\ubuntu\TrainingV5.csv')

x=dataset.iloc[:,0:num_samples_total]
y=dataset.iloc[:,num_samples_total]


#Initialize SVM classifier
clf = svm.SVC(kernel='linear',C=0.5)
clf = clf.fit(x, y)



ss='-0.110989015,9.674753958,1.657694603,0.004357693,-0.006754987,0.381317917,0.299606078,0.588452612,0.059368124,0.037392921,0.145403354,0.089763802,0.346276477,0.003524574,0.001398231,361.4858213,2.287323793,1.256381273,19.42765331,3.190331221,0.280003168,0.162333079,-0.754920542,9.156183243,0.501863003,-0.109083079,-0.092310466,0.501460731,10.27147007,2.688468218,0.170920089,0.070022613'
k=[]
for kv in ss.split(","):
    k.append(kv)
    print(kv)

print("-------------------")
print(k)
print("-------------------")

k=np.array(k)
predictions = clf.predict([k])
print("-------------------")

print(predictions)
print("-------------------")
send1="P: "+str(predictions)
# Predict the test set

@app.route("/t1")

def test():
    
    k2=[]
    a=request.args.get('e')
    if(a=="gg"):
        return "OK"
    for kv in a.split(","):
        k2.append(kv)
    p = clf.predict([k2])
    
    if(p==1):
        return "On Road"
    else:
        return "OFF Road"
     

app.run(host="0.0.0.0",port=5000)
    
    


