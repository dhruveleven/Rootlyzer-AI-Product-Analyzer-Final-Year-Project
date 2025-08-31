from flask import Blueprint, render_template

anomaly_bp = Blueprint('anomaly', __name__)

@anomaly_bp.route('/anomaly')
def anomaly():
    return render_template("anomaly.html", title="Anomaly Detection")
