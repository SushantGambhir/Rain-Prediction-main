from flask import Flask,render_template,request
from flask_cors import cross_origin
import pandas as pd
import pickle



app = Flask(__name__, template_folder="template")
# model = pickle.load(open('model.sav','rb'))
model = pickle.load(open("./models/cat.pkl","rb"))
print("Model Loaded")

@app.route("/",methods=['GET'])
@cross_origin()
def home():
	return render_template("index.html")

@app.route("/about",methods=['GET'])
def about():
	return render_template("about.html")

@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
	if request.method == "POST":
		date = request.form['date']
		day = float(pd.to_datetime(date, format="%Y-%m-%dT").day)
		month = float(pd.to_datetime(date, format="%Y-%m-%dT").month)
		# MinTemp
		minTemp = float(11.5)
		# MaxTemp
		maxTemp = float(21.2)
		# Rainfall
		rainfall = float(1)
		# Evaporation
		evaporation = float(4.3)
		# Sunshine
		sunshine = float(11.4)
		# Wind Gust Speed
		windGustSpeed = float(32.6)
		# Wind Speed 9am
		windSpeed9am = float(9.8)
		# Wind Speed 3pm
		windSpeed3pm = float(19.6)
		# Humidity 9am
		humidity9am = float(64.2)
		# Humidity 3pm
		humidity3pm = float(57.3)
		# Pressure 9am
		pressure9am = float(1017.8)
		# Pressure 3pm
		pressure3pm = float(1014.3)
		# Temperature 9am
		temp9am = float(12.7)
		# Temperature 3pm
		temp3pm = float(18.6)
		# Cloud 9am
		cloud9am = float(8)
		# Cloud 3pm
		cloud3pm = float(0)
		# Cloud 3pm
		location = float(request.form['location'])
		# Wind Dir 9am
		winddDir9am = float(5)
		# Wind Dir 3pm
		winddDir3pm = float(3)
		# Wind Gust Dir
		windGustDir = float(4)
		# Rain Today
		rainToday = float(0)

		input_lst = [location , minTemp , maxTemp , rainfall , evaporation , sunshine ,
					 windGustDir , windGustSpeed , winddDir9am , winddDir3pm , windSpeed9am , windSpeed3pm ,
					 humidity9am , humidity3pm , pressure9am , pressure3pm , cloud9am , cloud3pm , temp9am , temp3pm ,
					 rainToday , month , day]
		pred = model.predict(input_lst)
		output = pred
		if output == 0:
			return render_template("after_sunny.html")
		else:
			return render_template("after_rainy.html")
	return render_template("predictor.html")

if __name__=='__main__':
	app.run(debug=True)