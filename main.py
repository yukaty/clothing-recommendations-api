import os
import requests
import yaml
import json
from openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Initialize a Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv(".env", override=True)


def load_config(config_file="config.yaml"):
    """
    Load configuration settings from a YAML file.
    """
    try:
        with open(config_file, "r") as file:
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
        data = response.json()

        weather_data = {
            "city": data["name"],
            "weather": data["weather"][0]["description"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "cloudiness": data["clouds"]["all"],
            "rain": data.get("rain", {}).get("1h", "No rain"),
            "snow": data.get("snow", {}).get("1h", "No snow"),
        }
        return weather_data
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def format_prompt(prompt_template, weather_data):
    """
    Format the prompt template with weather data.
    """
    try:
        return prompt_template.format(
            city        = weather_data["city"],
            weather     = weather_data["weather"],
            temperature = weather_data["temperature"],
            feels_like  = weather_data["feels_like"],
            humidity    = weather_data["humidity"],
            wind_speed  = weather_data["wind_speed"],
            cloudiness  = weather_data["cloudiness"],
            rain        = weather_data["rain"],
            snow        = weather_data["snow"],
        )
    except KeyError as e:
        return f"Missing placeholder in prompt template: {e}"


def get_llm_response(prompt, client):
    """
    Get and return the response from the LLM for a given prompt.
    """
    try:
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {
                    "role": "system",
                    "content": "You are a structured assistant providing JSON-formatted clothing recommendations based on weather data.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return completion.choices[0].message.content, None
    except Exception as e:
        return "", str(e)


@app.route("/weather_and_outfit", methods=["POST"])
def get_weather_and_outfit():
    # Load configuration and API keys
    config = load_config()
    weather_api_url = config.get("weather_api_url")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    weather_api_key = os.getenv("WEATHER_API_KEY")
    prompt_template = config.get("prompt_template")

    if not all([weather_api_url, openai_api_key, weather_api_key, prompt_template]):
        return jsonify({"error": "Missing configuration or API keys."}), 500

    # Get parameters from user input or use config defaults
    data = request.get_json(silent=True) or {}
    latitude = data.get("latitude", config.get("default_latitude"))
    longitude = data.get("longitude", config.get("default_longitude"))

    if latitude is None or longitude is None:
        return jsonify({"error": "Latitude and longitude must be provided."}), 400

    # Get weather information
    weather_data = get_weather_info(weather_api_url, weather_api_key, latitude, longitude)
    if "error" in weather_data:
        return jsonify({"error": f"Failed to fetch weather data: {weather_data['error']}"}), 500

    # Create a prompt for the LLM
    prompt = format_prompt(prompt_template, weather_data)

    # Call OpenAI API
    client = OpenAI(api_key=openai_api_key)
    response_text, llm_error = get_llm_response(prompt, client)
    if llm_error:
        return jsonify({"error": f"Failed to fetch LLM response: {llm_error}"}), 500

    # Parse the response from OpenAI API as JSON
    llm_response = json.loads(response_text)

    # Return the combined response
    return jsonify({"weather_info": weather_data, "recommendation": llm_response})


if __name__ == "__main__":
    app.run(debug=True)
