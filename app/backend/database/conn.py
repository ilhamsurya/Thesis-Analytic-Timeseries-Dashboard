import mysql.connector as mysql


def connect():
    try:
        conn = mysql.connect(
            user="root",
            password="",
            database="sistem-ta",
            host="localhost"
        )
        print(conn)

    except:
        conn = None

    return conn
