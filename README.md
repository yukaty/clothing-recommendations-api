# Weather Wardrobe AI
This is an API that provides personalized outdoor outfit recommendations based on the current weather conditions.

This project was created as a learning exercise to explore the use of OpenAI API and Flask.

## Setup
1. Clone the repository
   ```sh
   git clone https://github.com/yukaty/weather-wardrobe-ai.git
   cd weather-wardrobe-ai
   ```

2. Create a virtual environment
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

4. Update configuration files
  - Create a `.env` file in the project root with the following keys:
     ```
     OPENAI_API_KEY=your_openai_api_key
     WEATHER_API_KEY=your_weather_api_key
     ```

5. Run the server
    ```sh
    python weather-wardrobe-ai.py
    ```

## cURL Example
```sh
curl -X POST http://127.0.0.1:5000/weather_and_outfit -H "Content-Type: application/json" -d '{"latitude": "51.5074", "longitude": "-0.1278"}'
```

## Requirements
- Python 3.7+
- Dependencies listed in `requirements.txt`
