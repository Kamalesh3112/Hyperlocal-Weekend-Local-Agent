pip install langchain
pip install google-ai-generativelanguage==0.6.15
pip install requests
pip install faiss-cpu
pip install tiktoken
pip install -U langchain-community

import datetime
import random
import json

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
import requests

import sys
print(sys.path)

DEFAULT_LOCATION = "Hosur, Tamil Nadu, India"
TARGET_WEEKEND_START = datetime.date(2025, 4, 19)
TARGET_WEEKEND_END = datetime.date(2025, 4, 20)
GEMINI_MODEL_NAME = 'gemini-2.0-flash'

OPENWEATHER_API_KEY = "<-Your-Open-Weather_API_Key->"
EVENTBRITE_API_KEY = "<-Your-Eventbrite-API-Key->"

class Event:
  def __init__(self, name, location, date, time, category, description, url=None):
    self.name = name
    self.location = location
    self.date = date
    self.time = time
    self.category = category
    self.description = description
    self.url = url

  def __str__(self):
    return f"{self.name} at {self.location} on {self.date} ({self.time})"

class WeatherForecast:
  def __init__(self, date, temperature, description, precipitation_chance, conditions):
    self.date = date
    self.temperature = temperature
    self.precipitation_chance = precipitation_chance
    self.conditions = conditions

  def __str__(self):
    return f"{self.date}: {self.conditions} (Temp: {self.temperature}°C, Rain: {self.precipitation_chance*100:.0f}%)"

class Venue:
  def __init__(self, name, type, address, description=None, rating=None, url=None):
    self.name = name
    self.type = type
    self.address = address
    self.descriptiondescription =
    self.rating = rating
    self.url = url

  def __str__(self):
    return f"{self.name} ({self.type}) - {self.address}"

def fetch_local_events(location, date_range, keywords):
  """Fetches local events using the Eventbrite API."""
  start_date_str = date_range[0].strftime('%Y-%m-%d')
  end_date_str = date_range[1].strftime('%Y-%m-%d')
  url = f"https://www.eventbriteapi.com/v3/events/search/?q={','.join(keywords)}&location.address={location}&start_date.range_start={start_date_str}T00:00:00Z&start_date.range_end={end_date_str}T23:59:59Z&token={EVENTBRITE_API_KEY}"
  try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    events =[]
    if 'events' in data:
      for event_data in data['events']:
        name = event_data['name']['text']
        venue_id = event_data['venue_id']
        location_name = "Hosur"
        start_time = event_data['start']['local']
        description = event_data['description']['text']
        url = event_data['url']
        categories =[cat['short_name'] for cat in event_data.get('category', [])]
        events.append(Event(name, location_name, start_time[:10], start_time[11:16], categories, description, url))
    return "\n".join([str(e) for e in events]) if events else "No events found."
  except requests.exceptions.RequestException as e:
    return f"Error fetching events: {e}"

import os

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "823aecdb925ac9449c38a3b25edfeb9a")


def get_weather_forecast(location, date_range):
    """Gets weather forecast using OpenWeatherMap API."""
    try:
        # Step 1: Geocode the location to get lat/lon
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={OPENWEATHER_API_KEY}"
        geo_resp = requests.get(geo_url)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if not geo_data:
            return f"Could not find coordinates for {location}"

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]

        # Step 2: Get 5-day / 3-hour forecast
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"
        forecast_resp = requests.get(forecast_url)
        forecast_resp.raise_for_status()
        forecast_data = forecast_resp.json()

        forecasts = []
        for entry in forecast_data["list"]:
            forecast_time = datetime.datetime.fromtimestamp(entry["dt"])
            forecast_date = forecast_time.date()

            # Filter for target weekend and specific hours (e.g., 9AM or 3PM)
            if date_range[0] <= forecast_date <= date_range[1] and forecast_time.hour in [9, 15]:
                desc = entry["weather"][0]["description"].title()
                temp = entry["main"]["temp"]
                rain_chance = entry.get("pop", 0) * 100  # pop = probability of precipitation

                forecast_str = f"{forecast_time.strftime('%A %H:%M')} - {desc}, {temp:.1f}°C, Rain: {rain_chance:.0f}%"
                forecasts.append(forecast_str)

        return "\n".join(forecasts) if forecasts else "No weekend weather forecast found."

    except Exception as e:
        return f"Error fetching weather forecast: {e}"


GOOGLE_MAPS_PLATFORM_API_KEY = "AIzaSyAwTkhzP-5snp-NfeUs8h5gFBlOYyTSt-c"
def find_outdoor_venues(location, venue_type):
  """Finds outdoor venues using Google Places API."""
  query = f"family-friendly outdoor {venue_type} in {location}"
  url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={query}&key={GOOGLE_MAPS_PLATFORM_API_KEY}"
  try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    venues = []
    if 'results' in data:
      for result in data['results']:
        name = result['name']
        address = result['formatted_address']
        place_type = result.get('types', [])
        rating = result.get('rating')
        url = f"https://www.google.com/maps/search/?api=1&query={result['formatted_address']}"
        if any(t in place_type for t in ['park', 'tourist_attraction', 'natural_feature']):
          venues.append(Venue(name, ", ".join(place_type), address, rating=rating, url=url))
    return "\n".join([str(v) for v in venues]) if venues else "No outdoor venues found."
  except:
    return f"Error fetching outdoor venues: {e}"


@tool
def search_local_events_tool(query: str) -> str:
  """Searches for local events based on a query."""
  return fetch_local_events(DEFAULT_LOCATION, (TARGET_WEEKEND_START, TARGET_WEEKEND_END), query.split())

@tool
def get_weekend_weather_tool(location: str = DEFAULT_LOCATION) -> str:
  """Gets the weather forecast for the upcoming weekend using Open Weather API."""
  return get_weather_forecast(location, (TARGET_WEEKEND_START, TARGET_WEEKEND_END))

@tool
def find_family_outdoor_places_tool(query: str) -> str:
  """Finds family-friendly outdoor places based on a query using Google Places API."""
  return find_outdoor_venues(DEFAULT_LOCATION, query)


def create_rag_chain(llm, embeddings, documents):
  """Creates a RetreivalQA chain for RAG."""
  vectorstore = FAISS.from_documents(documents, embeddings)
  retriever = vectorstore.as_retriever()
  prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know
  {context}
  Question: {question}
  """
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"])
  chain = RetrievalQA.from_llm(llm=llm, retriever=retriever, prompt=PROMPT)
  return chain

def load_local_knowledge():
  """Loads local knowledge for RAG."""
  local_info = [
      "Hosur is an industrial city in Tamil Nadu, known for its pleasant climate.",
        "Kelavarapalli Dam is a popular spot near Hosur for picnics and relaxation.",
        "The region around Hosur has several parks and natural trails suitable for families.",
  ]
  return [Document(page_content=info) for info in local_info]

from langchain.tools import Tool

def initialize_weekend_planner_agent():
  """Initializes the weekend planner agent with function tools and RAG"""
  llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key='AIzaSyAwTkhzP-5snp-NfeUs8h5gFBlOYyTSt-c')
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyAwTkhzP-5snp-NfeUs8h5gFBlOYyTSt-c')
  local_knowledge = load_local_knowledge()
  rag_chain = create_rag_chain(llm, embeddings, local_knowledge)

  tools = [
    Tool(
        name="get_weekend_weather",
        func=get_weekend_weather_tool,
        description="Fetch weather info for the weekend."
    ),
    Tool(
        name="find_family_outdoor_places",
        func=find_family_outdoor_places_tool,
        description="Find outdoor family-friendly places nearby."
    ),
    Tool(
        name="local_knowledge_search",
        func=rag_chain.run,
        description="Search for general information about Hosur and surroundings."
    )
  ]


  agent = initialize_agent(
      tools,
      llm,
      agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
      verbose=True
  )
  return agent

def main():
  agent = initialize_weekend_planner_agent()
  user_request = "What are some family-friendly outdoor activities happening near Hosur this weekend, and what will be the weather be like?"
  print(f"User Request: {user_request}")
  try:
    response = agent.invoke({"input": user_request})
    print(f"Agent Response: {response}")
  except Exception as e:
    print(f"An error occurred: {e}")

  user_request_2= "Tell me about nice parks or trails for families in Hosur."
  print(f"\nUser Request: {user_request_2}")
  try:
    response_2 = agent.invoke({"input": user_request_2})
    print(f"Agent Response: {response_2}")
  except Exception as e:
    print(f"An error occurred: {e}")

if __name__ == "__main__":
  main()

