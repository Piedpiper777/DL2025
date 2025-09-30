from flask import (Blueprint,
                   render_template)

bp = Blueprint(name="ai_report", import_name=__name__, url_prefix="/ai_report")


@bp.route("/ar", methods=["GET", "POST"])
def ai_report():
    return render_template("templates_ar/ai_report.html")