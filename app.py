from click.testing import Result
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from flask import send_from_directory


from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__,static_url_path='/D:/ccdp/templates/static/Star Landing.jpg')

app=application

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict_datapoint',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        print(request.form)
        data=CustomData(
        SEX=int(request.form['gender']),
        EDUCATION=int(request.form['education']),
        MARRIAGE=int(request.form['marriage']),
        AGE= int(request.form['age']),
        LIMIT_BAL=float(request.form['limit_bal']),
        PAY_1=int(request.form['april_rs']),
        PAY_2=int(request.form['may_rs']),
        PAY_3=int(request.form['june_rs']),
        PAY_4=int(request.form['july_rs']),
        PAY_5=int(request.form['august_rs']),
        PAY_6=int(request.form['september_rs']),
        BILL_AMT6=int(request.form['bill_amt6']),  
        BILL_AMT5=int(request.form['bill_amt5']),  
        BILL_AMT4=int(request.form['bill_amt4']),  
        BILL_AMT3=int(request.form['bill_amt3']),  
        BILL_AMT2=int(request.form['bill_amt2']),  
        BILL_AMT1=int(request.form['bill_amt1']),  
        PAY_AMT6=int(request.form['pay_amt6']),    
        PAY_AMT5=int(request.form['pay_amt5']),    
        PAY_AMT4=int(request.form['pay_amt4']),    
        PAY_AMT3=int(request.form['pay_amt3']),    
        PAY_AMT2=int(request.form['pay_amt2']),    
        PAY_AMT1=int(request.form['pay_amt1'])
        )

    bill_amt_avg = round(np.mean([data.BILL_AMT6, data.BILL_AMT5, data.BILL_AMT4, data.BILL_AMT3, data.BILL_AMT2, data.BILL_AMT1]), 2)
    features = [data.PAY_6, data.PAY_5, data.PAY_AMT1, data.BILL_AMT1]
    features += [data.LIMIT_BAL, data.AGE, data.PAY_AMT2, data.BILL_AMT2]
    features += [data.PAY_AMT3, data.BILL_AMT3, data.BILL_AMT4, data.PAY_AMT4]
    features += [data.PAY_AMT6, data.BILL_AMT5, data.BILL_AMT6, bill_amt_avg]

    
    sum_of_features = sum(features)

    features_arr = [np.array(features).reshape(1, -1)]

    pred_df=data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")

    predict_pipeline=PredictPipeline()
    print("Mid Prediction")

    result=predict_pipeline.predict(pred_df)
    
    # Convert the result to an integer (0 or 1)
    prediction = int(result[0])
    

    return render_template('home.html', result=result[0])

    
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
