import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/mushroom_classifier", methods=['GET', 'POST'])
def classifier():
    if request.method=='GET':
        return render_template("classifier.html")
    else:
        prediction = None
        capShape = request.form["shape"]          
        capSurface = request.form["surface"]            
        capColor = request.form["color"]
        bruises = request.form["bruises"]      
        odor = request.form["odor"]                   
        gillAttachment = request.form["attachment"]        
        gillSpacing = request.form["spacing"]           
        gillSize = request.form["size"]             
        gillColor = request.form["gill-color"]             
        stalkShape = request.form["stalk-shape"]            
        stalkRoot = request.form["root"]            
        stalkSurfaceAboveRing = request.form["ssa-ring"]
        stalkSurfaceBelowRing = request.form["ssb-ring"]
        stalkColorAboveRing = request.form["sca-ring"]
        stalkColorBelowRing = request.form["scb-ring"]
        veilType = request.form["veil-type"]             
        veilColor = request.form["veil-color"]            
        ringNumber = request.form["ring-number"]             
        ringType = request.form["ring-type"]            
        sporePrintColor = request.form["sp-color"]      
        population = request.form["population"]          
        habitat = request.form["habitat"]

@app.route("/prediction")
def pred():
    return render_template('prediction.html')

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)