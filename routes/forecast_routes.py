from flask import Blueprint, render_template

forecast_bp = Blueprint('forecast', __name__)

@forecast_bp.route('/forecast')
def forecast():
    return render_template("forecast.html", title="Forecasting")
