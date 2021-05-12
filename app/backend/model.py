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
    # cur.execute("SELECT COUNT(ID_KEJADIAN) as JUMLAH_KEJADIAN, sub_kategori.nama_sub_kategori as nama_sub_kategori FROM kejadian,sub_kategori WHERE kejadian.id_sub_kategori = sub_kategori.id_sub_kategori GROUP BY kejadian.ID_SUB_KATEGORI ORDER BY JUMLAH_KEJADIAN DESC")
    cur.execute("SELECT COUNT(kejadian.ID_KEJADIAN) as banyak, YEAR(punya.TGL_KEJADIAN) as tahun, kategori.nama_kategori FROM KEJADIAN, PUNYA, kategori, sub_kategori WHERE kategori.id_kategori = sub_kategori.id_kategori and kejadian.id_sub_kategori = sub_kategori.id_sub_kategori and KEJADIAN.ID_KEJADIAN = PUNYA.ID_KEJADIAN and sub_kategori.id_kategori = 15 group BY tahun")
    
    rv = cur.fetchall()
    cur.close()
    return render_template('kategori.html', kategori = rv)


# 404 Error handler
@app.errorhandler(404)
def resource_not_found(e):
    return render_template("auth/404.html"), 404
