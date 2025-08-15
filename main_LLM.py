import os
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# == directory == (CHANGE DATASET PATH HERE)
dir_OA = r"C:\Users\Tunnel\Documents\Soil_Identification\GCS Training\OA"
dir_OBOA =  r"C:\Users\Tunnel\Documents\Soil_Identification\GCS Training\OBOA"

directory = [
    dir_OA, dir_OBOA
]


# == model setup ==
model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash', api_key = 'AIzaSyC1itsgyzSi5N80uhJvFUdhn9zL04DAJRw')

prompt = ChatPromptTemplate.from_messages([
    ('system', 
    """You are a geotechnical engineer with 20 years of experience identifying soils. There are 2 types of soils that you are working with, OA [1] and OBOA [2]. \n
    Use the description of soil given: \n
    OA: very dense, yellowish brown mottled with light greenish grey, slightly gravelly and clayey, medium to coarse SAND, fine gravels are sub-rounded to sub-angular. \n
    OBOA: very dense, greenish grey and yellowish brown, slightly gravelley, very clayey, fine to coarse sand, fine gravels are sub-rounded to sub-angular, partially weathered. \n
    Return the requested response object in english. \n '{format_instructions}' \n"""
     ), #customise prompt here
    ('human', [
        {
            'type': 'image_url',
            'image_url': {'url': 'data:image/jpeg;base64,{image_data}'},
        },
    ]),
])

class Soil(BaseModel):
    name: str = Field(description = 'The name of the soil shown in the image, out of the 2 categories of soil given. Give their respective number only.')    #customise commands here
    colour: str = Field(description = 'The colour of the soil shown in the image.')
    wetness: str = Field(description = 'The perceived wetness of the soil shown in the image.')
    size: str = Field(description = 'The grain size of the soil shown in the image.')
    cofidence: str = Field(description = 'How accurate or sure you are of your soil classification, out of 10.')

parser = PydanticOutputParser(pydantic_object = Soil)

chain = prompt | model | parser


# == encode image to feed model ==
def encode_image_to_base64(image_path):
    with open (image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def image_processing(directory, max_worker = 5):
    for file in os.listdir(directory):
            file_path = os.path.join(directory, file)

            base64_image = encode_image_to_base64(file_path)
        
            result = chain.invoke({
            'format_instructions': parser.get_format_instructions(),
            'image_data': base64_image
            })
            
            print(f'file: {file}')
            print(result)
            print("--------------------------------------------------------")


# == command to execute model ==
for folder in directory:
    if os.path.exists(folder):
        print(f'processing {folder}...')
        image_processing(folder)
    else:
        print(f'Folder not found: {folder}')