
from flask import Flask,render_template,request,redirect
import pandas as pd
import pickle

scaler = pickle.load(open('scaler.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictor',methods=["GET", "POST"])
def pred():
    if(request.method == 'POST'):
        temp = request.form.get('temp')
        humid = request.form.get('humidity')
        ph = request.form.get('ph')
        rainfall = request.form.get('rainfall')

        
        x_df  = pd.DataFrame([[temp,humid,ph,rainfall]],columns = ['temperature', 'humidity', 'ph', 'rainfall'])
        
        
        res = model.predict(pd.DataFrame(scaler.transform(x_df),columns=['temperature', 'humidity', 'ph', 'rainfall']))
        
        
        crops = ['rice', 'wheat', 'Mung Bean', 'Tea', 'millet', 'maize', 'Lentil',
       'Jute', 'Coffee', 'Cotton', 'Ground Nut', 'Peas', 'Rubber',
       'Sugarcane', 'Tobacco', 'Kidney Beans', 'Moth Beans', 'Coconut',
       'Black gram', 'Adzuki Beans', 'Pigeon Peas', 'Chickpea', 'banana',
       'grapes', 'apple', 'mango', 'muskmelon', 'orange', 'papaya',
       'pomegranate', 'watermelon']

        return render_template('predictor.html',value = crops[res[0]])
    return render_template('predictor.html',value = "R")

if __name__ == '__main__':
    app.run(debug=True)

