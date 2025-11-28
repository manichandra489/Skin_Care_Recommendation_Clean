import torch
import clip
import PIL.Image
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from typing import Any, Optional
from pydantic import BaseModel

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
modeli, preprocess = clip.load("ViT-B/32", device=device)

pc = Pinecone(api_key="pcsk_734a9H_98bCfspdWWc4XgFoZBPsYB2CNw498LBK53KNDyF1WjVGsBcN8rsTzSYVWCc1GkQ")  #pcsk_xEske_CL1K1Kxm8Zu2ncNFPaVW9TKroYJHYJn7KYb6Vtug66GUF5q8mFPLN9JbpWkGhgP

index_name = "skindisease-symptoms-gpt-4"

# Create new index if it doesn’t exist
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

#if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#image_path = os.path.join(image_folder, filename)
def RAG(image,index_name,api):
  image = preprocess(PIL.Image.open(image)).unsqueeze(0).to(device)
  pc = Pinecone(api_key=api)
  index = pc.Index(index_name)
  with torch.no_grad():
      image_features = modeli.encode_image(image)

  image_features /= image_features.norm(dim=-1, keepdim=True)
  query_vector = image_features.cpu().numpy().flatten().tolist()
  results = index.query(
      vector=query_vector,
      top_k=1,
      include_metadata=True
  )
  results = results["matches"]
  return results

from pydantic import BaseModel
from typing import Optional,List
from typing import Annotated
import operator
class State(BaseModel):
  image: Optional[str] = None
  eligible:bool=False
  age: Optional[str] = None
  gender: Optional[str] = None
  bauman_type: Annotated[str, operator.add]
  skin_disease: Optional[str] = None
  meds: Annotated[List[str], operator.add] = []
  medimage: Annotated[List[str], operator.add] = []
  des: Optional[str] = None
  toxic: Optional[str] = None

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize client with your API key
# llm = ChatOpenAI( # Renamed instance for clarity
#       api_key="sk-proj-...",
#       model="gpt-4o-mini",  # You can use "gpt-4o", "gpt-5", etc.
#       temperature=0.0)
llm =  ChatOpenAI(
    model="moonshotai/kimi-k2-instruct-0905",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("llm_api_key"), temperature=0.0
)

# Chat completion example
def chat(state:State):
  messages=[
          SystemMessage(content="Act as a loreal paris skin care product recommender."),
          HumanMessage(content=f"Please suggest a loreal skin care product for a {state.gender} of age {state.age} and skin type is {state.bauman_type}.")
      ]

  chat_response = llm.invoke(messages) # Corrected call and assigned to a new variable
  return {'des':chat_response.content} # Return the content

# The print statement and function call should be after the function definition
#result = chat(state)
#print(result)

from tavily import TavilyClient
client = TavilyClient(api_key="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")
def websearch():
  response = client.search(f"Please suggest a loreal skin care product for a boy of age 16 and skin type is oily, pigmented, sensitive, wrinkle.") # Give only lorealparisusa websites
  response['results'][0]['url']
  print(response['results'])

websearch()

import re
def extract(state:State):
  matches = re.findall(r"\*\*(.*?)\*\*", state.des)
  return({'meds':matches})
#matches=extract(result)

from tavily import TavilyClient
client = TavilyClient(api_key="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")
def websearch(state:State):
  url=[]
  for i in state.meds:
    response = client.search(f"Show image of {i}",include_images=True) # Give only lorealparisusa websites
    url.append(response['results'][0]['url'])
  return {'medimage':url}
#websearch('loreal charcoal')

def examiner(state:State):
  #for i in len(ingrediants)-1:
   # j = i+1
  messages=[
          SystemMessage(content="Act as a skin care specialist aware of harmful ingredients in skincare products that are harmfull to skin."),
          HumanMessage(content=f"Please state the anti toxic products to each other {state.meds}. State if the products can be used together. If not state which ones should not be used together and only from the given products list only.")
      ]

  chat_response = llm.invoke(messages) # Corrected call and assigned to a new variable
  return {'toxic': chat_response.content} # Return the content

skinapi="pcsk_xEske_CL1K1Kxm8Zu2ncNFPaVW9TKroYJHYJn7KYb6Vtug66GUF5q8mFPLN9JbpWkGhgP"

def skin_disease(state: State):
    result = RAG(
        image=state.image,
        index_name="clip-skd",
        api=skinapi
    )

    disease = result[0]["metadata"]["Disease"]

    if disease == "Normal":
        return "normal"
    else:
        return "abnormal"

#skin_disease()

def com(state: State):
    return {"Eligible": True}

def skin_disease_reduce(state:State):
  if RAG(state.image,"skindisease-symptoms-gpt-4","pcsk_734a9H_98bCfspdWWc4XgFoZBPsYB2CNw498LBK53KNDyF1WjVGsBcN8rsTzSYVWCc1GkQ")[0]['score']<0.55:
    return {"skin_disease":''}
  return {"skin_disease":RAG(state.image,"skindisease-symptoms-gpt-4","pcsk_734a9H_98bCfspdWWc4XgFoZBPsYB2CNw498LBK53KNDyF1WjVGsBcN8rsTzSYVWCc1GkQ")[0]['metadata']['Disease']}

RAG('test.png',"skindisease-symptoms-gpt-4","pcsk_734a9H_98bCfspdWWc4XgFoZBPsYB2CNw498LBK53KNDyF1WjVGsBcN8rsTzSYVWCc1GkQ")[0]

def oily(state:State):
  return {"bauman_type":f"{RAG(state.image,'clip-image-index',skinapi)[0]['metadata']['filename'].split('_')[0]} "}

def sense(state:State):
  if RAG(state.image,"clip-sens",skinapi)[0]['metadata']['filename'].split("_")[0]=='normal':
    return {"bauman_type":"resistant "}
  else:
    return {"bauman_type":"sensitive "}

def pig(state:State):
  if RAG(state.image,"clip-pig",skinapi)[0]['metadata']['filename'].split("_")[1]=='png':
    return {"bauman_type":'pigmentation '}
  else:
    return {"bauman_type":'non-pigmentation '}

def wri(state:State):
  if RAG(state.image,"clip-wri",skinapi)[0]['metadata']['filename'][0]=='w':
    return {"bauman_type":'wrinkle'}
  else:
    return {"bauman_type":'tight'}

gender_dict = {0:"Male",1:"Female"}
def category(age):
  if age>=0 and age<18:
    age="Child"
  elif age>=18 and age<30:
    age="Adult"
  elif age>=30 and age <50:
    age="Man"
  elif age>=50 and age<100:
    age="Old"
  return age

import joblib
import numpy as np
import PIL.Image # Changed from 'from PIL import Image'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

def age(state:State):
  # Load the image as grayscale and resize
  model = load_model("aagp.keras")
  img = load_img(state.image, color_mode="grayscale", target_size=(128, 128))
  img = np.array(img)

  # Reshape to add batch dimension and channel dimension for Keras model input
  # Expected shape is (batch_size, height, width, channels)
  img = img.reshape(1, 128, 128, 1)

  pred = model.predict(img)
  pred_gender = gender_dict[round(pred[0][0][0])]
  pred_age = round(pred[1][0][0]/100)
  return {'gender':pred_gender,'age':category(pred_age)}

builder = StateGraph(State)
builder.add_node("skin disease reduce", skin_disease_reduce)
builder.add_node("com", com)
builder.add_node("oily", oily)
builder.add_node("sense", sense)
builder.add_node("pig", pig)
builder.add_node("wri", wri)
builder.add_node("age", age)
builder.add_node("products", chat)
builder.add_node("extract", extract)
builder.add_node("examiner", examiner)
builder.add_node("ProImg", websearch)

builder.add_conditional_edges(
    START,
    skin_disease,
    {
        "normal": "com",
        "abnormal": END
    }
)
builder.add_edge("com", "skin disease reduce")
builder.add_edge("com", "oily")
builder.add_edge("com", "sense")
builder.add_edge("com", "pig")
builder.add_edge("com", "wri")
builder.add_edge("com", "age")
builder.add_edge("skin disease reduce", "products")
builder.add_edge("oily", "products")
builder.add_edge("sense", "products")
builder.add_edge("pig", "products")
builder.add_edge("wri", "products")
builder.add_edge("age", "products")
builder.add_edge("products", "extract")
builder.add_edge("extract", "ProImg")
builder.add_edge("ProImg", END)
builder.add_edge("extract", "examiner")
builder.add_edge("examiner", END)

graph = builder.compile()
import IPython.display
display(IPython.display.Image(graph.get_graph().draw_mermaid_png()))
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict

class ImageState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Optional[Any] = None

def run(img_state: ImageState):
    messages = graph.invoke({"image": img_state.image})
    return messages
messages = graph.invoke({"image": "test.png"})


