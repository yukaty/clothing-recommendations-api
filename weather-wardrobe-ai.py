import os
import requests
import yaml
from openai import OpenAI
from dotenv import load_dotenv

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

def print_llm_response(prompt, client):
    """
    Get and print the response from the LLM for a given prompt.
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
    response = completion.choices[0].message.content
    print(response)

def main():
    # Load configuration and API keys
    config = load_config()
    api_keys = load_api_keys()

    # Get weather information
    weather_data = get_weather_info(
        config['weather_api_url'],
        api_keys['weather_api_key'],
        config['latitude'],
        config['longitude']
    )

    # Format and print weather information
    weather_string = format_weather_info(weather_data)
    print(weather_string)

    # Create a prompt for the LLM
    prompt = f"""
            Based on the following weather data, suggest an appropriate outdoor outfit.
            Provide simple and actionable recommendations in bullet points, considering factors like temperature, humidity, wind speed, and precipitation.
            Example: "Wear sunglasses and sunscreen."

            {weather_string}"""

    # Print the LLM response
    client = OpenAI(api_key=api_keys['openai_api_key'])
    print_llm_response(prompt, client)

if __name__ == "__main__":
    main()
