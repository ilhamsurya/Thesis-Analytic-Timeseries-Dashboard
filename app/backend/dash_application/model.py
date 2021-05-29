import os, sys
import datetime as dt
from .data import create_dataframe
import pandas as pd
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
# from flask_mysqldb import MySQL


# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] ='time_series'
# mysql = MySQL(app)
# from flask_mysqldb import MySQL
# from conn.py import conn

# Home route
@app.route("/kategori")
def kategori():
    df = create_dataframe()
    # df['Tanggal'] = 
    # df['tanggal'] = pd.to_datetime(df['tanggal'])
    # df = df.groupby(['tanggal','kategori'], as_index=False)['ID'].count()
    # df = df.set_index('tanggal')
    # df = df.loc['2014-01-01':'2019-12-30']
    # df = df.groupby([pd.Grouper(freq="M"), 'kategori'])['ID'].count().reset_index()
    # print(df)
    
    return render_template('kategori.html', kategori = df)


# 404 Error handler
@app.errorhandler(404)
def resource_not_found(e):
    return render_template("auth/404.html"), 404
