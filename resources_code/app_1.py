#!Users\holozadah\Documents\GitHub\Final\Website
from flask import Flask, render_template
from flask import request, redirect
app = Flask(__name__)
import time
import pandas as pd
import numpy as np
import random


# read the files
dff= pd.read_csv('groups.csv')
df = pd.read_csv('Lottery_data.csv')
df['result']=1

#Create an loosing tickets
first= []
second=[]
third=[]
fourth=[]
fifth=[]
sixth=[]
result=[]

for x in range(1000):
    first.append(random.randint(1,70))
    second.append(random.randint(1,70))
    third.append(random.randint(1,70))
    fourth.append(random.randint(1,70))
    fifth.append(random.randint(1,70))
    sixth.append(random.randint(1,25))
    result.append(0)


#Creating a dataframe for loosing tickets
df_loser= pd.DataFrame({'first':first,
                       'second':second,
                       'third':third,
                        'fourth':fourth,
                        'fifth':fifth,
                        'Yellow':sixth,
                        'result':result
                       }) 


# Concatinate winning and loosing tickets
frames=[df,df_loser]
result = pd.concat(frames)

#Import Random Forest Classifier from SKLEARN library
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, criterion='entropy',max_depth=10,random_state=0)

# Spilt and train the data
from sklearn.model_selection import train_test_split
X=result[['first','second','third','fourth','fifth','Yellow']]
Y= result['result']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

# Normalizing the data values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fit the data into the model
clf.fit(x_train,y_train)


def predict_results(numbers):
    
    # Calculate the total probability of the numbers
    prob_total = dff[dff['numbers'] == numbers[0]]['prob_1'].values *dff[dff['numbers'] == numbers[1]]['prob_2'].values*dff[dff['numbers'] == numbers[2]]['prob_3'].values *dff[dff['numbers'] == numbers[3]]['prob_4'].values *dff[dff['numbers'] ==numbers[4]]['prob_5'].values 
    
    # Results classification list
    range_result = ['Nominal Chance','Slight Chance','Low Chance','Average Chance', 'Good Chance','Better Chance','Best Chance']

    # Classify the results based on the total probabilities outcome
    if float(prob_total) > 1e-8:
        return (range_result[6])
    elif float(prob_total) > 1e-9:
        return (range_result[5])
    elif float(prob_total) > 1e-10:
        return (range_result[4])
    elif float(prob_total) > 1e-11:
        return (range_result[3])
    elif float(prob_total) > 1e-12:
        return (range_result[2])
    elif float(prob_total) > 1e-15:
        return (range_result[1])
    elif float(prob_total) <= 1e-15:
        return (range_result[0])
        

@app.route('/')
def lottery():

    return render_template('prediction_1.html', result= ["", "", ""])


@app.route('/predict', methods = ['POST', 'GET'])
def result_1():
    if request.method == 'POST':
      
        num1=int(request.form['num1'])
        num2=int(request.form['num2'])
        num3=int(request.form['num3'])
        num4=int(request.form['num4'])
        num5=int(request.form['num5'])
        num6=int(request.form['num6'])
        results=[num1,num2,num3,num4,num5]
        results.sort()
        results.append(num6)
        prediction = predict_results(results)
        results.append(prediction)
  
    return render_template('prediction_1.html', predicted_result = results)

@app.route('/generate', methods = ['POST', 'GET'])
def lot():
    first_nu =random.sample(range(1,70), 5)
    sixth_nu = random.randint(1,25)
    numbers = first_nu + [sixth_nu]
    re = clf.predict([numbers])
    if re == 1:
        numbers.remove(numbers[5])
        numbers.sort()
        prediction = predict_results(numbers)
        numbers.append(sixth_nu)
        numbers.append(prediction)

        return render_template('prediction_1.html',result= numbers)


if __name__ == '__main__':
    app.run(debug = True)
