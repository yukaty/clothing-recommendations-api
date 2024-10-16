import os
import requests
import yaml
from openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Initialize a Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv('.env', override=True)


def load_config(config_file='config.yaml'):
    """
    Load configuration settings from a YAML file.
    """
    try:
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}


def get_weather_info(weather_api_url, api_key, latitude, longitude):
    """
    Fetch weather information from the weather API.
    """
    url = f"{weather_api_url}?units=metric&lat={latitude}&lon={longitude}&appid={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return {}, str(e)


def format_weather_info(data):
    """
    Format the weather information for display and further use.
    """
    try:
        return f"""Weather data:
    - City:        {data['name']}
    - Weather:     {data['weather'][0]['description']}
    - Temperature: {data['main']['temp']} °C
    - Feels like:  {data['main']['feels_like']} °C
    - Humidity:    {data['main']['humidity']} %
    - Wind speed:  {data['wind']['speed']} m/s
    - Cloudiness:  {data['clouds']['all']} %
    - Rain (h/mm): {data.get('rain', {}).get('1h', 'No rain')}
    - Snow (h/mm): {data.get('snow', {}).get('1h', 'No snow')}
    """
    except KeyError:
        return "Invalid weather data received."


def get_llm_response(prompt, client):
    """
    Get and return the response from the LLM for a given prompt.
    """
    try:
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
        return completion.choices[0].message.content, None
    except Exception as e:
        return "", str(e)


@app.route('/weather_and_outfit', methods=['POST'])
def get_weather_and_outfit():
    # Load configuration and API keys
    config = load_config()
    weather_api_url = config.get('weather_api_url')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    weather_api_key = os.getenv('WEATHER_API_KEY')

    if not all([weather_api_url, openai_api_key, weather_api_key]):
        return jsonify({'error': 'Missing configuration or API keys.'}), 500

    # Get parameters from user input or use config defaults
    data = request.get_json()
    latitude = data.get('latitude', config.get('latitude'))
    longitude = data.get('longitude', config.get('longitude'))

    if latitude is None or longitude is None:
        return jsonify({'error': 'Latitude and longitude must be provided.'}), 400

    # Get weather information
    weather_data, weather_error = get_weather_info(
        weather_api_url, weather_api_key, latitude, longitude
    )
    if weather_error:
        return jsonify({'error': f'Failed to fetch weather data: {weather_error}'}), 500

    # Format weather information
    weather_string = format_weather_info(weather_data)

    # Create a prompt for the LLM
    prompt = f"""
        Based on the following weather data, suggest an appropriate outdoor outfit.
        Provide simple and actionable recommendations in bullet points, considering factors like temperature, humidity, wind speed, and precipitation.
        Example: "Wear sunglasses and sunscreen."

        {weather_string}
    """

    # Call OpenAI API
    client = OpenAI(api_key=openai_api_key)
    response, llm_error = get_llm_response(prompt, client)
    if llm_error:
        return jsonify({'error': f'Failed to fetch LLM response: {llm_error}'}), 500

    # Return the combined response
    return jsonify({'weather_info': weather_string, 'recommendation': response})


if __name__ == "__main__":
    app.run(debug=True)