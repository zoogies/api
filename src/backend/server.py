# imports
import random
import psutil
import json
import requests
import time
import datetime
from lib import jsontools
from flask import Flask, request
from flask_cors import CORS
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ===============================
# FLASK OPERATIONS
# ===============================

# create flask app
app = Flask(__name__)

# cors-ify the app
CORS(app)

# ===============================
# OTHER SETUP
# ===============================

# set the public ip address
ip=requests.get('https://api64.ipify.org?format=json').json()['ip']

# set up the gpt2 zoogies model related things
model_dir = "/hub_api/models/zoogies_one_epoch"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained(model_dir)
device = "cpu"
model.to(device)

# get the running version
# Open the package.json file
with open("package.json", "r") as f:
    # Load the contents of the file as a JSON object
    version = json.load(f)["version"]

# AUTHORIZATION
serverkey = None
with open("/hub_api/backups/secret.txt") as f:
    serverkey = f.read().strip()

# ===============================
# UTILITY FUNCTIONS
# ===============================

# general purpose auth function with locally defined server secret
def auth(password):
    if serverkey == password.strip():
        return True
    else:
        return False

def updatecurrent(data): # TODO MOVE THIS FUNC OUT OF THIS MAIN FILE
    # check validity of json uploaded
        if(jsontools.validate(data)): # returns true if data meets criteria, false if not
            # create a duplicate of the 'current.json' file renamed to the date (backup)
            open("/hub_api/backups/"+str(int(time.time()))+".json", "w").write(open("/hub_api/backups/current.json").read().replace("'",'"')) # replace normal apostrophe with double quote or it wont work
           
            # rewrite 'current.json' to consist of non duplicate keys from upload and old records
            merged = jsontools.merge('/hub_api/backups/current.json',data)
            
            with open('/hub_api/backups/current.json', "w") as current:
                current.write(str(merged).replace("'",'"')) # replace normal apostrophe with double quote or it wont work
            return "done",200
        else:
            return "Bad json data",400

def getcurrent():
    return json.load(open('/hub_api/backups/current.json'))

# ===============================
# HUB API ROUTES
# ===============================

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

# ===============================
# MITSURI GIF API ROUTES
# ===============================

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
    # check timesfavorited in json sent, if its more than current.json we need to push changes, if its less we need to pull changes, if its the same nothing needs to happen
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

# ===============================
# ZOOGIES GPT2 MODEL ROUTES
# ===============================

# route for getting the available models for the dropdown list on the hub playground

available_models = [{'name':'zoogies_one_epoch','description':'GPT2-125M fine tuned one epoch on 60k lines of discord logs up to 2022.'}] # potentially work in a release date here and then client side can sort by newest

@app.route('/api/completion/models', methods=['GET'])
def get_models():
    try:
        return{'models': available_models}
    except Exception as e:
        return {'error':str(e)}


@app.route('/api/completion', methods=['POST'])
def generate_text():
    try:
        prompt = request.form.get('prompt')
        max_length = int(request.form.get('max_length', 200))
        num_return_sequences = int(request.form.get('num_return_sequences', 1))
        temperature = float(request.form.get('temperature', 0.9))
        top_k = int(request.form.get('top_k', 90))
        top_p = float(request.form.get('top_p', 0.9))
        no_repeat_ngram_size = int(request.form.get('no_repeat_ngram_size', 1))

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_text = [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in output]
        return {'generated_text': generated_text}
    except Exception as e:
        return {'error':str(e)}

# ===============================
# BASE ROUTE VERSION NUMBER
# ===============================

@app.route('/api')
def root():
    return "stable release >> v"+version+" >> Ryan Zmuda, 2022-"+str(datetime.date.today().year)