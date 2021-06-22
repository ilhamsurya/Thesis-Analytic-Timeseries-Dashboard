import os, time
from flask import (
    render_template,
    jsonify,
    Response,
)
import flask
from geojson import Point, Feature, FeatureCollection
from app import server


MAPBOX_ACCESS_KEY = os.environ.get("MAPBOX_ACCESS_KEY")

# from flask_mysqldb import MySQL
# from conn.py import conn

# Home route
@server.route("/")
def index():
    return render_template("landingpage.html")


# User registration route
@server.route("/register")
def register():
    return render_template("auth/register.html")


# User login route
@server.route("/login")
def login():
    return render_template("auth/login.html")


# User Dashboard roaute
@server.route("/timeseries/")
def timeseries():
    return flask.redirect("/timeseries")


# # User Dashboard roaute
# @app.route("/map")
# def heatmap():
#     return render_template("dashboard/map.html")


# 404 Error handler
@server.errorhandler(404)
def resource_not_found(e):
    return render_template("auth/404.html"), 404
