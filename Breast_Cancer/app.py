from flask import Flask,render_template,request
import pickle
import numpy as np
app = Flask('__name__')
model=pickle.load(open('Breast_Cancer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    feature=[float(x) for x in request.form.values()]
    feature_final=np.array(feature).reshape(-1,30)
    prediction=model.predict(feature_final)
    

    if prediction=="M":
        return "<h1 style='color:green' 'text-align:center' >malignant</h1>"
    else:
        return "<h1 style='color:red'> benign</h1>"


if(__name__=='__main__'):
    app.run(debug=True)

### Add ###