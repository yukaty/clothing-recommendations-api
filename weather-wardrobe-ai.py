import os
import requests
import yaml
from openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Initialize a Flask app
app = Flask(__name__)

def load_config(config_file='config.yaml'):
    """
    Load configuration settings from a YAML file.
    """
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def load_api_keys():
    """
    Load API keys from the .env file.
    """
    load_dotenv('.env', override=True)
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'weather_api_key': os.getenv('WEATHER_API_KEY')
    }

def get_weather_info(weather_api_url, api_key, latitude, longitude):
    """
    Fetch weather information from the weather API.
    """
    url = f"{weather_api_url}?units=metric&lat={latitude}&lon={longitude}&appid={api_key}"
    response = requests.get(url)
    return response.json()

def format_weather_info(data):
    """
    Format the weather information for display and further use.
    """
    weather_string = f"""Weather data:
    - City:        {data['name']}
    - Weather:     {data['weather'][0]['description']}
    - Temperature: {data['main']['temp']} °C
    - Feels like:  {data['main']['feels_like']} °C
    - Humidity:    {data['main']['humidity']} %
    - Wind speed:  {data['wind']['speed']} m/s
    - Cloudiness:  {data['clouds']['all']} %
    - Rain (h/mm): {data['rain']['1h'] if 'rain' in data and '1h' in data['rain'] else 'No rain'}
    - Snow (h/mm): {data['snow']['1h'] if 'snow' in data and '1h' in data['snow'] else 'No snow'}
    """
    return weather_string

def get_llm_response(prompt, client):
    """
    Get and return the response from the LLM for a given prompt.
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a concise and practical AI assistant who provides clear and actionable advice.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return completion.choices[0].message.content

@app.route('/weather_and_outfit', methods=['POST'])
def get_weather_and_outfit():
    # Load configuration and API keys
    config = load_config()
    api_keys = load_api_keys()

    # Get parameters from user input or use config defaults
    data = request.get_json()
    latitude = data.get('latitude', config['latitude'])
    longitude = data.get('longitude', config['longitude'])

    # Get weather information
    weather_data = get_weather_info(
        config['weather_api_url'],
        api_keys['weather_api_key'],
        latitude,
        longitude
    )

    # Format weather information
    weather_string = format_weather_info(weather_data)

    # Create a prompt for the LLM
    prompt = f"""
            Based on the following weather data, suggest an appropriate outdoor outfit.
            Provide simple and actionable recommendations in bullet points, considering factors like temperature, humidity, wind speed, and precipitation.
            Example: "Wear sunglasses and sunscreen."

            {weather_string}"""

    # Call OpenAI API
    client = OpenAI(api_key=api_keys['openai_api_key'])
    response = get_llm_response(prompt, client)

    # Return the combined response
    return jsonify({'weather_info': weather_string, 'recommendation': response})

if __name__ == "__main__":
    app.run(debug=True)