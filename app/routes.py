import os, time
from flask import (
    render_template,
    jsonify,
    Response,
)
import flask
from geojson import Point, Feature, FeatureCollection
from app import app
from app.backend.mapbox import (
    create_route_url,
    create_stop_location_detail,
    create_stop_locations_details,
    get_route_data,
)

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
    route_data, waypoints = get_route_data()

    stop_locations = create_stop_locations_details()

    return render_template(
        "mapbox_js.html",
        ACCESS_KEY=MAPBOX_ACCESS_KEY,
        route_data=route_data,
        stop_locations=stop_locations,
    )


@app.route("/mapbox_gl")
def mapbox_gl():
    route_data, waypoints = get_route_data()

    stop_locations = create_stop_locations_details()

    # For each stop location, add the waypoint index
    # that we got from the route data
    for stop_location in stop_locations:
        waypoint_index = stop_location.properties["route_index"]
        waypoint = waypoints[waypoint_index]
        stop_location.properties["location_index"] = route_data["geometry"][
            "coordinates"
        ].index(waypoint["location"])

    return render_template(
        "mapbox_gl.html",
        ACCESS_KEY=MAPBOX_ACCESS_KEY,
        route_data=route_data,
        stop_locations=stop_locations,
    )


# 404 Error handler
@app.errorhandler(404)
def resource_not_found(e):
    return render_template("auth/404.html"), 404
