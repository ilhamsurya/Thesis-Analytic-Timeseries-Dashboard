import mysql.connector as mysql


def connect():
    try:
        conn = mysql.connect(
            user="root",
            password="",
            database="time_series",
            host="localhost"
        )
        print(conn)

    except:
        conn = None

    return conn
