from click.testing import Result
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from flask import send_from_directory


from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

## route for a home page
@app.route('/static/templates/static/irr.jpg')
def serve_file(filename):
    return send_from_directory('static', filename)



@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict_datapoint',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        #data=CustomData(
        gender = (request.form['gender']),
        education = [int(request.form['education'])],
        marital_status = [int(request.form['marriage'])],
        age = [int(request.form['age'])],
        bal_limit = [float(request.form['limit_bal'])],
        rs_6 = [int(request.form['april_rs'])],
        rs_5 = [int(request.form['may_rs'])],
        rs_4 = [int(request.form['june_rs'])],
        rs_3 = [int(request.form['july_rs'])],
        rs_2 = [int(request.form['august_rs'])],
        rs_1 = [int(request.form['september_rs'])],
        bill_6 = [int(request.form['bill_amt6'])],
        bill_5 = [int(request.form['bill_amt5'])],
        bill_4 = [int(request.form['bill_amt4'])],
        bill_3 = [int(request.form['bill_amt3'])],
        bill_2 = [int(request.form['bill_amt2'])],
        bill_1 = [int(request.form['bill_amt1'])],
        pay_6 = [int(request.form['pay_amt6'])],
        pay_5 = [int(request.form['pay_amt5'])],
        pay_4 = [int(request.form['pay_amt4'])],
        pay_3 = [int(request.form['pay_amt3'])],
        pay_2 = [int(request.form['pay_amt2'])],
        pay_1 = [int(request.form['pay_amt1'])]
        

    bill_amt_avg = [round(np.mean([bill_6, bill_5, bill_4, bill_3, bill_2, bill_1]), 2)]
    features = rs_1 + rs_2 + pay_1 + bill_1
    features = features + bal_limit + age + pay_2 + bill_2
    features = features + pay_3 + bill_3 + bill_4 + pay_4
    features = features + pay_6 + bill_5 + bill_6 + bill_amt_avg
    
    features_arr = [np.array(features)]

    pred_df=data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")

    predict_pipeline=PredictPipeline()
    print("Mid Prediction")
    results=predict_pipeline.predict(pred_df)
    print("after Prediction")
    return render_template('home.html',results=results[0])

    
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
