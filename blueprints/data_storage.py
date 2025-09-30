from flask import (Blueprint,
                   render_template)

bp = Blueprint(name="data_storage", import_name=__name__, url_prefix="/data_storage")


@bp.route("/data_storage", methods=["GET", "POST"])
def data_storage():
    return render_template("templates_ds/data_storage.html")

@bp.route("/detection_result", methods=["GET", "POST"])
def detection_result_storage():
    return render_template("templates_ds/detection_result_storage.html")