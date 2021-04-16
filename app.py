import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    '''to get the input values'''
    feat = [int(x) for x in request.form.values()]
    final = [np.array(feat)]
    pred = model.predict(final)
    
    output = round(pred[0],2)
    
    return render_template('index.html', pred_text = 'The chances of rain are {}%'.format(output))

if __name__ == "__main__" :
    app.run(host = '0.0.0.0', port = 8080)

