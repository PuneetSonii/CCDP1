from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

application=Flask(__name__)

app=application

## route for a home page

@app.route('/')
def index():
    return render_template('index')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        gender = gender_encode(int(request.form.get['gender']))
        education = education_encode(int(request.form.get['education']))
        marital_status = marital_encode(int(request.form.get['marriage']))
        age = [int(request.form.get['age'])]
        bal_limit = [int(request.form.get['limit_bal'])]
        rs_6 = [int(request.form.get['april_rs'])]
        rs_5 = [int(request.form.get['may_rs'])]
        rs_4 = [int(request.form.get['june_rs'])]
        rs_3 = [int(request.form.get['july_rs'])]
        rs_2 = [int(request.form.get['august_rs'])]
        rs_1 = [int(request.form.get['september_rs'])]
        bill_6 = [int(request.form.get['bill_amt6'])]
        bill_5 = [int(request.form.get['bill_amt5'])]
        bill_4 = [int(request.form.get['bill_amt4'])]
        bill_3 = [int(request.form.get['bill_amt3'])]
        bill_2 = [int(request.form.get['bill_amt2'])]
        bill_1 = [int(request.form.get['bill_amt1'])]
        pay_6 = [int(request.form.get['pay_amt6'])]
        pay_5 = [int(request.form.get['pay_amt5'])]
        pay_4 = [int(request.form.get['pay_amt4'])]
        pay_3 = [int(request.form.get['pay_amt3'])]
        pay_2 = [int(request.form.get['pay_amt2'])]
        pay_1 = [int(request.form.get['pay_amt1'])]

    bill_amt_avg = [round(np.mean([bill_6, bill_5, bill_4, bill_3, bill_2, bill_1]), 2)]
    features = rs_1 + rs_2 + pay_1 + bill_1
    features = features + bal_limit + age + pay_2 + bill_2
    features = features + pay_3 + bill_3 + bill_4 + pay_4
    features = features + pay_6 + bill_5 + bill_6 + bill_amt_avg
    
    features_arr = [np.array(features)]

    pred_df=data.get_data_as_data_frame()
    print(pred_df)


    predict_pipeline=predictPipeline()
    result=predict_pipeline.predict(pred_df)
    return render_template('home.html',results=results[0])
    #result = ""
    #if prediction == 1:
    #  result = "This customer IS LIKELY TO DEFAULT next month."
    #else:
    #  result = "This customer IS NOT LIKELY TO DEFAULT next month."
