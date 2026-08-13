from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_churn():
    if request.method=='GET':
        return render_template('Home.html')
    else:
        data=request.form.to_dict()
        custom_data=CustomData(data)
        df=custom_data.tranform_to_df()
        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(df)
        return render_template('home.html',results=result[0])





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)