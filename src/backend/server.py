# imports
import random
import psutil
import json
import requests
import time
from lib import jsontools
from flask import Flask, request
from flask_cors import CORS

# create flask app
app = Flask(__name__)

# cors-ify the app
CORS(app)

# set the public ip address
ip=requests.get('https://api64.ipify.org?format=json').json()['ip']

# AUTHORIZATION
serverkey = None
with open("secret.txt") as f:
    serverkey = f.read().strip()

# general purpose auth function with locally defined server secret
def auth(password):
    if serverkey == password.strip():
        return True
    else:
        return False

def updatecurrent(data):
    # check validity of json uploaded
        if(jsontools.validate(data)): # returns true if data meets criteria, false if not
            # create a duplicate of the 'current.json' file renamed to the date (backup)
            open("./backups/"+str(int(time.time()))+".json", "w").write(open("./backups/current.json").read().replace("'",'"')) # replace normal apostrophe with double quote or it wont work
           
            # rewrite 'current.json' to consist of non duplicate keys from upload and old records
            merged = jsontools.merge('backups/current.json',data)
            
            with open('./backups/current.json', "w") as current:
                current.write(str(merged).replace("'",'"')) # replace normal apostrophe with double quote or it wont work
            return "done",200
        else:
            return "Bad json data",400

def getcurrent():
    return json.load(open('backups/current.json'))

# HUB API ROUTES

# api route for getting server statistics
@app.route('/api/hub/getstats')
def getstats():
    return {
        #"user":str(os.getlogin()),
        "ip":ip,
        "boot-time":round(psutil.boot_time()),
        "cpu-count":psutil.cpu_count(),
        "load-average":psutil.cpu_percent(interval=None),
        "memory":{
            "total":round(psutil.virtual_memory().total * 0.000000001,2),
            "used":round(psutil.virtual_memory().used * 0.000000001,2),
            "percent":psutil.virtual_memory().percent,
        },
        "swap":{
            "total":round(psutil.swap_memory().total * 0.000000001,2),
            "used":round(psutil.swap_memory().used * 0.000000001,2),
           "percent":psutil.swap_memory().percent,
        },
        "disk":{
            "total":round(psutil.disk_usage('/').total * 0.000000001,2),
            "used":round(psutil.disk_usage('/').used * 0.000000001,2),      
            "percent":psutil.disk_usage('/').percent
        }
    }

# MITSURI API ROUTES

@app.route('/api/mitsuri/getgifs')
def getgifs():
    return getcurrent()

@app.route('/api/mitsuri/setgifs', methods=["POST"])
def setgifs():
    # check auth to access this endpoint
    if(auth(request.authorization['password'])):
        return updatecurrent(request.json)
    else:
        return "Incorrect authorization.",401

@app.route('/api/mitsuri/syncgifs', methods=["POST"])
def syncgifs():
    # process of syncing

    # check timesfavorited in json sent, if its more than current.json we need to push changes, if its less we need to pull changes, if its the same nothign needs to happen
    if(auth(request.authorization['password'])):
        if(jsontools.validate(request.json)):
            request_times = request.json['_state']['timesFavorited']
            current_times = jsontools.getcurrentfavorited()
            if(request_times == current_times):
                # print("equal - no action needed")
                return {"status":"equal"},200
            elif(request_times > current_times):
                # print('more - need to update current')
                updatecurrent(request.json)
                return {"status":"ahead"},200
            elif(request_times < current_times):
                # print('less - need to update posted')
                return {"status":"behind","new":getcurrent()},200


    # OTHER: if we go into the realm of automated check all clients connected
    return "An Error Has Occurred",500

@app.route('/api/mitsuri/ryangif', methods=["GET"])
def ryangif():
    gifs = getcurrent()
    index = random.randint(0,len(gifs['_state']['favorites']))
    return {"url":gifs['_state']['favorites'][index]['url'],"total":gifs['_state']['timesFavorited'],"index":index}

@app.route('/api')
def root():
    return "Api release >> 12.22.22 >> Ryan Zmuda"


# print("REMINDER >> Create src/backend/secret.txt && src/backend/backups/current.json")
# print("REMINDER >> Current secret is set to -> "+serverkey)
# app.run(host='0.0.0.0', use_reloader=True, port=5055, threaded=True, debug=True)

# TODO this code quality and readability is so trash and not maintainable