import os
import sys
sys.path.append("src")
import pandas as pd
from exception import CustomException
from utils import load_object
import pickle as pkl
from sklearn.preprocessing import OrdinalEncoder




class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            encoder = OrdinalEncoder()
            model_path=os.path.join("artifacts","model.sav")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            preprocessor = pkl.load(open(preprocessor_path, 'rb'))
            predictor = pkl.load(open(model_path, 'rb'))
            #model=load_object(file_path=model_path)
            #preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=predictor.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                capShape:str,
                capSurface:str,
                capColor:str,
                bruises:str,
                odor:str,
                gillAttachment:str,
                gillSpacing,
                gillSize:str,
                gillColor:str,
                stalkShape:str,
                stalkRoot:str,
                stalkSurfaceAboveRing:str,
                stalkSurfaceBelowRing:str,
                stalkColorAboveRing:str,
                stalkColorBelowRing:str,
                veilType:str,
                veilColor:str,
                ringNumber:str,
                ringType,
                sporePrintColor:str,
                population:str,
                habitat:str):

        self.capShape = capShape
        self.capSurface = capSurface
        self.capColor = capColor
        self.bruises = bruises
        self.odor = odor
        self.gillAttachment = gillAttachment
        self.gillSpacing = gillSpacing
        self.gillSize = gillSize
        self.gillColor = gillColor
        self.stalkShape = stalkShape
        self.stalkRoot = stalkRoot
        self.stalkSurfaceAboveRing = stalkSurfaceAboveRing
        self.stalkSurfaceBelowRing = stalkSurfaceBelowRing
        self.stalkColorAboveRing = stalkColorAboveRing
        self.stalkColorBelowRing = stalkColorBelowRing
        self.veilType = veilType
        self.veilColor = veilColor
        self.ringNumber = ringNumber
        self.ringType = ringType
        self.sporePrintColor = sporePrintColor
        self.population = population
        self.habitat = habitat

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "capShape":[self.capShape], 
                "capSurface":[self.capSurface], 
                "capColor":[self.capColor], 
                "bruises":[self.bruises],
                "odor":[self.odor], 
                "gillAttachment":[self.gillAttachment],
                "gillSpacing":[self.gillSpacing],
                "gillSize":[self.gillSize], 
                "gillColor":[self.gillColor],
                "stalkShape":[self.stalkShape], 
                "stalkRoot":[self.stalkRoot],
                "stalkSurfaceAboveRing":[self.stalkSurfaceAboveRing], 
                "stalkSurfaceBelowRing":[self.stalkSurfaceBelowRing], 
                "stalkColorAboveRing":[self.stalkColorAboveRing],
                "stalkColorBelowRing":[self.stalkColorBelowRing], 
                "veilType":[self.veilType], 
                "veilColor":[self.veilColor],
                "ringNumber":[self.ringNumber], 
                "ringType":[self.ringType], 
                "sporePrintColor":[self.sporePrintColor],
                "population":[self.population], 
                "habitat":[self.habitat]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

