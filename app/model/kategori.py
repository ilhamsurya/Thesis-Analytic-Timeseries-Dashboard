from flask import Flask, render_template, request
from flask_mysqldb import MySQL
from conn.py import conn


#config database
@app.route('/kategori')
def kategori():
    return "test"
    # cursor = conn.cursor()
    # hasil = cursor.execute("SELECT * FROM kategori")
    # if hasil > 0 :
    #     result = cursor.fetchall()
    #     return render_template('templates/testing.html', result = result)


