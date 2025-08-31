from flask import Blueprint, render_template

rootcause_bp = Blueprint('rca', __name__)

@rootcause_bp.route('/rca')
def rca():
    return render_template("rca.html", title="Root Cause Analysis")
