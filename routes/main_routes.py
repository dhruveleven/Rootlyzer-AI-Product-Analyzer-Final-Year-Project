import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from werkzeug.utils import secure_filename

main_bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "json"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@main_bp.route("/")
def landing():
    return render_template("landing.html")

@main_bp.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or file.filename == "":
            flash("Please select a file to upload.", "error")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Allowed file types: csv, xlsx, json.", "error")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # Placeholder: pretend we processed the file
        flash("Upload successful. Processing complete (placeholder).", "success")
        return redirect(url_for("main.results", filename=filename))

    return render_template("upload.html")

@main_bp.route("/results")
def results():
    filename = request.args.get("filename", "sample.csv")
    # We pass placeholders the Results page can show
    placeholders = {
        "report_ready": True,
        "dashboard_ready": True,
        "anomaly_ready": True,
        "rca_ready": True,
        "forecast_ready": True,
        "benchmark_ready": True,
        "recommend_ready": True,
    }
    return render_template("results.html", filename=filename, placeholders=placeholders)
