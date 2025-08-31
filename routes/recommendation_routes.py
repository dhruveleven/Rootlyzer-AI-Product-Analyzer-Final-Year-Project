from flask import Blueprint, render_template

recommendation_bp = Blueprint('recommend', __name__)

@recommendation_bp.route('/recommend')
def recommend():
    return render_template("recommend.html", title="Recommendations")
