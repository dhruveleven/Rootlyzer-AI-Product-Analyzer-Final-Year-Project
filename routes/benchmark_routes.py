from flask import Blueprint, render_template

benchmark_bp = Blueprint('benchmark', __name__)

@benchmark_bp.route('/benchmark')
def benchmark():
    return render_template("benchmark.html", title="Benchmark")
