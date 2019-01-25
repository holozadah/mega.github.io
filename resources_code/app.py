import pymysql
from flask import Flask, render_template, request
app = Flask(__name__)

class Database:
    def __init__(self):
        host = "localhost"
        user = "root"
        db = "lottery"
        passwd = "1234"

        self.con = pymysql.connect(
            host = host,
            user = user,
            db = db,
            password = passwd,
            cursorclass=pymysql.cursors.DictCursor 
        )
        self.cur = self.con.cursor()

    def list_lottery(self):
        self.cur.execute("SELECT * FROM lotterysalesbyzip")
        result = self.cur.fetchall()
        return result

@app.route('/')
def lottery():
     
 
    def db_query():
        db = Database()
        lott = db.list_lottery()
 
        return lott
 
    res = db_query()
 
    return render_template('lottery.html', result=res, content_type='application/json')
