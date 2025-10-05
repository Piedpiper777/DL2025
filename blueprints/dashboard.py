import json

from flask import (Blueprint,
                   render_template,
                   request, redirect,
                   url_for,
                   jsonify,
                   session)
from decorators import login_required

bp = Blueprint(name="dashboard", import_name=__name__, url_prefix="/dashboard")


@bp.route("/", methods=["GET", "POST"])
@login_required
def dashboard():
    # TODO 加入用户信息
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "dashboard.html",
        zh=zh_json["Dashboard"]
    )

@bp.route("/application", methods=["GET", "POST"])
@login_required
def dashboard_application():
    # TODO 加入用户信息
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "application.html",
        zh=zh_json["Dashboard"]
    )
#1.缺陷分类
@bp.route("/cls", methods=["GET", "POST"])
@login_required
def dashboard_cls():
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_cls/dashboard.html",
        zh=zh_json["Dashboard"]
    )

#2.缺陷分割
@bp.route("/seg", methods=["GET", "POST"])
@login_required
def dashboard_seg():
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_seg/dashboard.html",
        zh=zh_json["Dashboard"]
    )

#3.缺陷检测
@bp.route("/det", methods=["GET", "POST"])
@login_required
def dashboard_det():
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_det/dashboard.html",
        zh=zh_json["Dashboard"]
    )

#4.故障诊断
@bp.route("/fd", methods=["GET", "POST"])
@login_required
def dashboard_fd():
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_fd/dashboard.html",
        zh=zh_json["Dashboard"]
    )

#5.图像采集 
@bp.route("/ic", methods=["GET", "POST"])
@login_required
def dashboard_ic():
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_ic/dashboard.html",
        zh=zh_json["Dashboard"]
    )

#6.数据存储
@bp.route("/ds", methods=["GET", "POST"])
@login_required
def dashboard_ds():
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_ds/dashboard.html",
        zh=zh_json["Dashboard"]
    )

#7.AI报告生成
@bp.route("/ar", methods=["GET", "POST"])
@login_required
def dashboard_ar():
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_ar/dashboard.html",
        zh=zh_json["Dashboard"]
    )

#8.剩余寿命预测
@bp.route("/rul", methods=["GET", "POST"])
@login_required
def dashboard_rul():
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_rul/dashboard.html",
        zh=zh_json["Dashboard"]
    )

#9.大模型知识图谱
@bp.route("/kg", methods=["GET", "POST"])
@login_required
def dashboard_kg():
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_lk/dashboard.html",
        zh=zh_json["Dashboard"]
    )

# test 
# @bp.route("/test", methods=["GET", "POST"])
# @login_required
# def dashboard_test():
#     # username = session["user"]
#     with open("language/text-zh.json", "r", encoding="utf-8") as f:
#         zh_json = json.load(f)
#     return render_template(
#         "templates_test/dashboard.html",
#         zh=zh_json["Dashboard"]
#     )