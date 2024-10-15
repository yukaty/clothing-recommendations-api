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
    temperature = data['main']['temp']
    feels_like  = data['main']['feels_like']
    wind_speed  = data['wind']['speed']
    description = data['weather'][0]['description']
    city        = data['name']

    weather_string = f"""
    The temperature is {temperature}°C in {city}.
    It is currently {description}, with a wind speed of {wind_speed} m/s.
    The temperature currently feels like {feels_like}°C.
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
                "content": "You are a helpful but terse AI assistant who gets straight to the point.",
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
    Based on the following weather, suggest an appropriate outdoor outfit.
    Simplify the suggestions with bullet points, like "Wear sunglasses and sunscreen as it's partly cloudy."
    Forecast: {weather_string}"""

    # Print the LLM response
    client = OpenAI(api_key=api_keys['openai_api_key'])
    print_llm_response(prompt, client)

if __name__ == "__main__":
    main()
