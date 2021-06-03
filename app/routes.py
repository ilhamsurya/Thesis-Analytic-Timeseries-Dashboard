import os, time
from flask import (
    render_template,
    jsonify,
    Response,
)
import flask
from geojson import Point, Feature, FeatureCollection
from app import app


MAPBOX_ACCESS_KEY = os.environ.get("MAPBOX_ACCESS_KEY")

# from flask_mysqldb import MySQL
# from conn.py import conn

# Home route
@app.route("/")
def index():
    return render_template("landingpage.html")


# User registration route
@app.route("/register")
def register():
    return render_template("auth/register.html")


# User login route
@app.route("/login")
def login():
    return render_template("auth/login.html")


# User Dashboard roaute
@app.route("/timeseries/")
def timeseries():
    return flask.redirect("/timeseries")


# User Dashboard roaute
@app.route("/map")
def heatmap():
    return render_template("dashboard/map.html")


@app.route("/mapbox_js")
def mapbox_js():

    return render_template(
        "mapbox_js.html",
        ACCESS_KEY=MAPBOX_ACCESS_KEY,
    )


# 404 Error handler
@app.errorhandler(404)
def resource_not_found(e):
    return render_template("auth/404.html"), 404
