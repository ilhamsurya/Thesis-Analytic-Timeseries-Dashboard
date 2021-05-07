import os, sys
import datetime as dt
from flask import (
    render_template,
    url_for,
    redirect,
    request,
    make_response,
    abort,
    jsonify,
    session,
    flash,
)
from app import app
from flask_mysqldb import MySQL


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] ='time_series'
mysql = MySQL(app)
# from flask_mysqldb import MySQL
# from conn.py import conn

# Home route
@app.route("/kategori")
def kategori():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM kategori")
    rv = cur.fetchall()
    cur.close()
    return render_template('kategori.html', kategori = rv)


# 404 Error handler
@app.errorhandler(404)
def resource_not_found(e):
    return render_template("auth/404.html"), 404
