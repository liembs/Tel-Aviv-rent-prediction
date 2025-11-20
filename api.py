from flask import Flask, request, render_template
import pandas as pd
import joblib
import traceback
import numpy as np
from assets_data_prep import prepare_data
from sklearn.linear_model import ElasticNet

app = Flask(__name__)

# טען את המודל המאומן
model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'area': float(request.form.get('area', 0)),
            'building_tax': float(request.form.get('building_tax', 0)),
            'distance_from_center': float(request.form.get('distance_from_center', 0)),
            'elevator': int(request.form.get('elevator', 0)),
            'floor': float(request.form.get('floor', 0)),
            'handicap': int(request.form.get('handicap', 0)),
            'has_balcony': int(request.form.get('has_balcony', 0)),
            'has_bars': int(request.form.get('has_bars', 0)),
            'has_safe_room': int(request.form.get('has_safe_room', 0)),
            'is_furnished': int(request.form.get('is_furnished', 0)),
            'is_renovated': int(request.form.get('is_renovated', 0)),
            'monthly_arnona': float(request.form.get('monthly_arnona', 0)),
            'neighborhood': request.form.get('neighborhood', ''),
            'property_type': request.form.get('property_type', ''),
            'total_floors': float(request.form.get('total_floors', 0)),

            # שדות נדרשים נוספים
            'description': '',
            'address': '',
            'garden_area': 0,
        }

        print("📥 Data received from form:")
        print(data)

        df_input = pd.DataFrame([data])
        df_prepared = prepare_data(df_input, mode="test")

        print("✅ Data after preparation:")
        print(df_prepared.columns.tolist())

        prediction = model.predict(df_prepared)[0]
        prediction_rounded = f"{prediction:,.0f}"

        return render_template('result.html', prediction=prediction_rounded)

    except Exception as e:
        return f"<h2>❌ שגיאה:</h2><pre>{traceback.format_exc()}</pre>"

if __name__ == '__main__':
    app.run(debug=True)
