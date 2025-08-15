import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import folium
from streamlit_folium import st_folium
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
import asyncio

# Configuration
st.set_page_config(
    page_title="🌤️ Advanced Weather Radar & Satellite System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class WeatherData:
    """Enhanced data class for comprehensive weather information"""
    location: str
    temperature: float
    feels_like: float
    humidity: int
    pressure: int
    visibility: float
    uv_index: float
    wind_speed: float
    wind_direction: int
    wind_gust: float
    dew_point: float
    wet_bulb_temp: float
    description: str
    icon: str
    timestamp: datetime
    precipitation: float = 0.0
    snow_depth: float = 0.0
    thunder_probability: float = 0.0

@dataclass
class HurricaneData:
    """Data class for hurricane/storm information"""
    name: str
    category: int
    wind_speed: float
    pressure: float
    lat: float
    lon: float
    movement_direction: int
    movement_speed: float
    forecast_path: List[Dict]

class WeatherTool(BaseTool):
    name: str = "get_current_weather"
    description: str = "Get current weather information for a specific location with comprehensive details"

    def _run(self, location: str) -> str:
        try:
            api_key = st.session_state.get('openweather_api_key')
            if not api_key:
                return "OpenWeather API key not provided"

            base_url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": location,
                "appid": api_key,
                "units": "metric"
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()

                # Calculate additional parameters
                temp = data['main']['temp']
                humidity = data['main']['humidity']
                pressure = data['main']['pressure']

                # Calculate dew point
                dew_point = temp - ((100 - humidity) / 5)

                # Calculate wet bulb temperature (approximation)
                wet_bulb = temp * np.arctan(0.151977 * np.sqrt(humidity + 8.313659)) + np.arctan(temp + humidity) - np.arctan(humidity - 1.676331) + 0.00391838 * np.power(humidity, 1.5) * np.arctan(0.023101 * humidity) - 4.686035

                weather_info = f"""
                Location: {data['name']}, {data['sys']['country']}
                Temperature: {data['main']['temp']}°C
                Feels like: {data['main']['feels_like']}°C
                Dew Point: {dew_point:.1f}°C
                Wet Bulb Temperature: {wet_bulb:.1f}°C
                Humidity: {data['main']['humidity']}%
                Pressure: {data['main']['pressure']} hPa
                Wind: {data['wind']['speed']} m/s
                Wind Gust: {data['wind'].get('gust', 'N/A')} m/s
                Wind Direction: {data['wind'].get('deg', 'N/A')}°
                Visibility: {data.get('visibility', 'N/A')} m
                Cloudiness: {data['clouds']['all']}%
                Description: {data['weather'][0]['description']}
                Precipitation (1h): {data.get('rain', {}).get('1h', 0)} mm
                Snow (1h): {data.get('snow', {}).get('1h', 0)} mm
                """
                return weather_info
            else:
                return f"Error fetching weather data: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, location: str) -> str:
        return self._run(location)

class ForecastTool(BaseTool):
    name: str = "get_weather_forecast"
    description: str = "Get comprehensive 5-day weather forecast with detailed parameters"

    def _run(self, location: str) -> str:
        try:
            api_key = st.session_state.get('openweather_api_key')
            if not api_key:
                return "OpenWeather API key not provided"

            base_url = "http://api.openweathermap.org/data/2.5/forecast"
            params = {
                "q": location,
                "appid": api_key,
                "units": "metric"
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                forecast_info = f"5-day comprehensive forecast for {data['city']['name']}:\n"

                for item in data['list'][:15]:  # Next 15 forecasts (5 days, 3-hour intervals)
                    dt = datetime.fromtimestamp(item['dt'])

                    # Calculate additional parameters
                    temp = item['main']['temp']
                    humidity = item['main']['humidity']
                    dew_point = temp - ((100 - humidity) / 5)

                    forecast_info += f"""
{dt.strftime('%Y-%m-%d %H:%M')}:
  Temperature: {item['main']['temp']}°C (feels like {item['main']['feels_like']}°C)
  Dew Point: {dew_point:.1f}°C
  Humidity: {humidity}%
  Pressure: {item['main']['pressure']} hPa
  Wind: {item['wind']['speed']} m/s, {item['wind'].get('deg', 'N/A')}°
  Wind Gust: {item['wind'].get('gust', 'N/A')} m/s
  Weather: {item['weather'][0]['description']}
  Rain: {item.get('rain', {}).get('3h', 0)} mm
  Snow: {item.get('snow', {}).get('3h', 0)} mm
  Thunder Probability: {item.get('pop', 0)*100:.0f}%
"""

                return forecast_info
            else:
                return f"Error fetching forecast data: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, location: str) -> str:
        return self._run(location)

class AirQualityTool(BaseTool):
    name: str = "get_air_quality"
    description: str = "Get comprehensive air quality information including UV index"

    def _run(self, location: str) -> str:
        try:
            api_key = st.session_state.get('openweather_api_key')
            if not api_key:
                return "OpenWeather API key not provided"

            # First get coordinates
            geo_url = "http://api.openweathermap.org/geo/1.0/direct"
            geo_params = {"q": location, "limit": 1, "appid": api_key}
            geo_response = requests.get(geo_url, params=geo_params)

            if geo_response.status_code == 200:
                geo_data = geo_response.json()
                if geo_data:
                    lat, lon = geo_data[0]['lat'], geo_data[0]['lon']

                    # Get air quality
                    aq_url = "http://api.openweathermap.org/data/2.5/air_pollution"
                    aq_params = {"lat": lat, "lon": lon, "appid": api_key}
                    aq_response = requests.get(aq_url, params=aq_params)

                    # Get UV Index
                    uv_url = "http://api.openweathermap.org/data/2.5/uvi"
                    uv_params = {"lat": lat, "lon": lon, "appid": api_key}
                    uv_response = requests.get(uv_url, params=uv_params)

                    result = f"Environmental data for {location}:\n"

                    if aq_response.status_code == 200:
                        aq_data = aq_response.json()
                        aqi = aq_data['list'][0]['main']['aqi']
                        components = aq_data['list'][0]['components']

                        aqi_levels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}

                        result += f"""
Air Quality:
  AQI Level: {aqi} ({aqi_levels.get(aqi, 'Unknown')})
  CO: {components['co']} μg/m³
  NO₂: {components['no2']} μg/m³
  O₃: {components['o3']} μg/m³
  PM2.5: {components['pm2_5']} μg/m³
  PM10: {components['pm10']} μg/m³
  SO₂: {components['so2']} μg/m³
  NH₃: {components['nh3']} μg/m³
"""

                    if uv_response.status_code == 200:
                        uv_data = uv_response.json()
                        uv_value = uv_data.get('value', 0)

                        uv_levels = {
                            (0, 2): "Low",
                            (3, 5): "Moderate",
                            (6, 7): "High",
                            (8, 10): "Very High",
                            (11, float('inf')): "Extreme"
                        }

                        uv_level = "Unknown"
                        for (low, high), level in uv_levels.items():
                            if low <= uv_value <= high:
                                uv_level = level
                                break

                        result += f"""
UV Index:
  Value: {uv_value:.1f}
  Level: {uv_level}
  Solar Power Potential: {min(uv_value * 10, 100):.0f}%
"""

                    return result
                else:
                    return "Location not found"
            else:
                return "Error fetching location coordinates"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, location: str) -> str:
        return self._run(location)

class HurricaneTool(BaseTool):
    name: str = "get_hurricane_data"
    description: str = "Get active hurricane and tropical storm information"

    def _run(self, region: str = "atlantic") -> str:
        try:
            # This is a placeholder for hurricane data - in real implementation,
            # you would use NOAA's API or similar hurricane tracking service
            hurricane_info = f"""
Hurricane/Tropical Storm Tracker for {region.title()} Region:

Note: This is a demo implementation. For real hurricane data, integrate with:
- NOAA Hurricane Database (HURDAT)
- National Hurricane Center (NHC) API
- Global Disaster Alert and Coordination System (GDACS)

Current Active Systems (Demo Data):
1. Hurricane Example
   - Category: 2
   - Wind Speed: 96 mph (154 km/h)
   - Pressure: 965 hPa
   - Location: 25.5°N, 80.2°W
   - Movement: NW at 12 mph
   - Forecast: Strengthening expected

For real-time hurricane data, please check:
- https://www.nhc.noaa.gov/
- https://www.gdacs.org/
"""
            return hurricane_info
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, region: str = "atlantic") -> str:
        return self._run(region)

class WeatherAgent:
    def __init__(self, gemini_api_key: str, openweather_api_key: str):
        """Initialize the enhanced weather agent with API keys"""
        self.gemini_api_key = gemini_api_key
        self.openweather_api_key = openweather_api_key

        # Configure Gemini
        genai.configure(api_key=gemini_api_key)

        # Initialize LangChain LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.3
        )

        # Initialize enhanced tools
        self.tools = [
            WeatherTool(),
            ForecastTool(),
            AirQualityTool(),
            HurricaneTool()
        ]

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def get_response(self, user_input: str) -> str:
        """Get response from the weather agent"""
        try:
            response = self.agent.run(user_input)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

class EnhancedWeatherAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.map_url = "http://maps.openweathermap.org/maps/2.0/weather"

    def get_current_weather(self, location: str) -> Dict:
        """Get enhanced current weather data"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()

                # Enhance data with calculated values
                temp = data['main']['temp']
                humidity = data['main']['humidity']

                # Calculate dew point
                data['dew_point'] = temp - ((100 - humidity) / 5)

                # Calculate wet bulb temperature (approximation)
                data['wet_bulb_temp'] = self._calculate_wet_bulb(temp, humidity)

                return data
            return None
        except Exception as e:
            st.error(f"Error fetching current weather: {e}")
            return None

    def get_forecast(self, location: str) -> Dict:
        """Get enhanced 5-day forecast data"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()

                # Enhance each forecast item
                for item in data['list']:
                    temp = item['main']['temp']
                    humidity = item['main']['humidity']
                    item['dew_point'] = temp - ((100 - humidity) / 5)
                    item['wet_bulb_temp'] = self._calculate_wet_bulb(temp, humidity)

                return data
            return None
        except Exception as e:
            st.error(f"Error fetching forecast: {e}")
            return None

    def get_air_quality(self, lat: float, lon: float) -> Dict:
        """Get comprehensive air quality data"""
        try:
            url = f"{self.base_url}/../air_pollution"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key
            }
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Error fetching air quality: {e}")
            return None

    def get_uv_index(self, lat: float, lon: float) -> Dict:
        """Get UV index data"""
        try:
            url = f"{self.base_url}/uvi"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key
            }
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Error fetching UV index: {e}")
            return None

    def get_weather_map_url(self, layer: str, lat: float, lon: float, zoom: int = 5) -> str:
        """Get weather map tile URL for radar/satellite imagery"""
        # Available layers: temp_new, precipitation_new, pressure_new, wind_new, clouds_new
        return f"{self.map_url}/{layer}/{zoom}/{int(lat)}/{int(lon)}.png?appid={self.api_key}"

    def _calculate_wet_bulb(self, temp: float, humidity: float) -> float:
        """Calculate wet bulb temperature"""
        try:
            return temp * np.arctan(0.151977 * np.sqrt(humidity + 8.313659)) + \
                   np.arctan(temp + humidity) - np.arctan(humidity - 1.676331) + \
                   0.00391838 * np.power(humidity, 1.5) * np.arctan(0.023101 * humidity) - 4.686035
        except:
            return temp  # Fallback to regular temperature

def create_weather_radar_map(weather_data: Dict, layer_name: str = "precipitation") -> folium.Map:
    """
    Create an interactive weather radar map using OpenWeather's live tile layers.
    """
    if not weather_data or 'coord' not in weather_data:
        return folium.Map(location=[20, 0], zoom_start=2)

    lat, lon = weather_data['coord']['lat'], weather_data['coord']['lon']

    # Create base map
    m = folium.Map(location=[lat, lon], zoom_start=6)

    # Map human-readable names to OpenWeather layer IDs
    layer_map = {
        "temperature": "temp_new",
        "precipitation": "precipitation_new",
        "wind": "wind_new",
        "pressure": "pressure_new",
        "clouds": "clouds_new"
    }

    # Get correct OpenWeather layer ID
    layer_id = layer_map.get(layer_name.lower(), "clouds_new")

    # Build OpenWeather tile URL
    tile_url = (
        f"https://tile.openweathermap.org/map/{layer_id}/{{z}}/{{x}}/{{y}}.png"
        f"?appid={st.session_state['openweather_api_key']}"
    )

    # Add OpenWeather radar layer
    folium.TileLayer(
        tiles=tile_url,
        attr="OpenWeatherMap",
        name=layer_name.title(),
        overlay=True,
        control=True
    ).add_to(m)

    # Add location marker
    folium.Marker(
        [lat, lon],
        popup=f"{weather_data['name']}<br>{weather_data['main']['temp']}°C",
        tooltip=weather_data['name'],
        icon=folium.Icon(color='red', icon='cloud')
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


    lat, lon = weather_data['coord']['lat'], weather_data['coord']['lon']

    # Create base map
    m = folium.Map(location=[lat, lon], zoom_start=8)

    # Add location marker
    folium.Marker(
        [lat, lon],
        popup=f"{weather_data['name']}<br>{weather_data['main']['temp']}°C",
        tooltip=weather_data['name'],
        icon=folium.Icon(color='red', icon='cloud')
    ).add_to(m)

    # Add weather layer (Note: This requires OpenWeatherMap's map API)
    # In a real implementation, you would overlay weather tiles
    folium.plugins.HeatMap([
        [lat + np.random.uniform(-0.5, 0.5), lon + np.random.uniform(-0.5, 0.5), 0.5]
        for _ in range(50)  # Demo heat points
    ]).add_to(m)

    return m

def create_hurricane_tracker_map() -> folium.Map:
    """Create hurricane tracking map"""
    # Create base map centered on Atlantic
    m = folium.Map(location=[25, -60], zoom_start=4)

    # Demo hurricane data (in real implementation, fetch from NOAA/NHC)
    hurricanes = [
        {
            'name': 'Hurricane Demo 1',
            'lat': 25.5,
            'lon': -80.2,
            'category': 2,
            'wind_speed': 96,
            'path': [[24, -82], [25.5, -80.2], [27, -78]]
        },
        {
            'name': 'Tropical Storm Demo 2',
            'lat': 18.2,
            'lon': -65.5,
            'category': 0,
            'wind_speed': 45,
            'path': [[16, -67], [18.2, -65.5], [20, -63]]
        }
    ]

    colors = ['red', 'orange', 'yellow', 'green', 'blue']

    for storm in hurricanes:
        color = colors[min(storm['category'], 4)]

        # Current position
        folium.Marker(
            [storm['lat'], storm['lon']],
            popup=f"""
            <b>{storm['name']}</b><br>
            Category: {storm['category']}<br>
            Wind Speed: {storm['wind_speed']} mph
            """,
            icon=folium.Icon(color=color, icon='warning-sign')
        ).add_to(m)

        # Storm path
        folium.PolyLine(
            storm['path'],
            color=color,
            weight=3,
            opacity=0.7
        ).add_to(m)

        # Forecast cone (simplified)
        folium.Circle(
            [storm['lat'], storm['lon']],
            radius=storm['wind_speed'] * 1000,
            popup=f"Wind field: {storm['wind_speed']} mph",
            color=color,
            fill=True,
            fillOpacity=0.2
        ).add_to(m)

    return m

def create_comprehensive_charts(forecast_data: Dict) -> Dict[str, go.Figure]:
    """Create comprehensive weather charts"""
    if not forecast_data:
        return {}

    # Extract data
    dates = [datetime.fromtimestamp(item['dt']) for item in forecast_data['list']]
    temps = [item['main']['temp'] for item in forecast_data['list']]
    feels_like = [item['main']['feels_like'] for item in forecast_data['list']]
    humidity = [item['main']['humidity'] for item in forecast_data['list']]
    pressure = [item['main']['pressure'] for item in forecast_data['list']]
    wind_speed = [item['wind']['speed'] for item in forecast_data['list']]
    wind_gust = [item['wind'].get('gust', item['wind']['speed']) for item in forecast_data['list']]
    wind_dir = [item['wind'].get('deg', 0) for item in forecast_data['list']]
    rain = [item.get('rain', {}).get('3h', 0) for item in forecast_data['list']]
    snow = [item.get('snow', {}).get('3h', 0) for item in forecast_data['list']]
    dew_points = [item.get('dew_point', temps[i] - ((100 - humidity[i]) / 5)) for i, item in enumerate(forecast_data['list'])]

    charts = {}

    # Temperature and related parameters
    fig_temp = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature & Feels Like', 'Dew Point & Wet Bulb',
                       'Humidity', 'Heat Index'),
        vertical_spacing=0.1
    )

    fig_temp.add_trace(go.Scatter(x=dates, y=temps, name='Temperature', line=dict(color='red')), row=1, col=1)
    fig_temp.add_trace(go.Scatter(x=dates, y=feels_like, name='Feels Like', line=dict(color='orange')), row=1, col=1)
    fig_temp.add_trace(go.Scatter(x=dates, y=dew_points, name='Dew Point', line=dict(color='blue')), row=1, col=2)
    fig_temp.add_trace(go.Scatter(x=dates, y=humidity, name='Humidity', line=dict(color='green')), row=2, col=1)

    # Calculate heat index
    heat_index = [t + 0.5 * (h - 50) / 10 for t, h in zip(temps, humidity)]
    fig_temp.add_trace(go.Scatter(x=dates, y=heat_index, name='Heat Index', line=dict(color='purple')), row=2, col=2)

    fig_temp.update_layout(height=600, title='Temperature Analysis', showlegend=True)
    charts['temperature'] = fig_temp

    # Wind analysis
    fig_wind = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Wind Speed & Gusts', 'Wind Direction',
                       'Wind Pressure Effect', 'Wind Accumulation'),
        specs=[[{}, {}], [{}, {"type": "polar"}]]
    )

    fig_wind.add_trace(go.Scatter(x=dates, y=wind_speed, name='Wind Speed', line=dict(color='blue')), row=1, col=1)
    fig_wind.add_trace(go.Scatter(x=dates, y=wind_gust, name='Wind Gusts', line=dict(color='red')), row=1, col=1)
    fig_wind.add_trace(go.Scatter(x=dates, y=wind_dir, name='Wind Direction', line=dict(color='green')), row=1, col=2)

    # Wind pressure effect (simplified calculation)
    wind_pressure = [0.613 * (ws ** 2) for ws in wind_speed]
    fig_wind.add_trace(go.Scatter(x=dates, y=wind_pressure, name='Wind Pressure', line=dict(color='purple')), row=2, col=1)

    # Polar wind plot
    fig_wind.add_trace(go.Scatterpolar(
        r=wind_speed[:24],  # First 24 hours
        theta=wind_dir[:24],
        mode='lines+markers',
        name='Wind Pattern'
    ), row=2, col=2)

    fig_wind.update_layout(height=600, title='Wind Analysis', showlegend=True)
    charts['wind'] = fig_wind

    # Precipitation analysis
    fig_precip = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Rain & Snow', 'Precipitation Accumulation',
                       'Precipitation Type', 'Thunder Probability'),
        vertical_spacing=0.1
    )

    fig_precip.add_trace(go.Bar(x=dates, y=rain, name='Rain', marker_color='blue'), row=1, col=1)
    fig_precip.add_trace(go.Bar(x=dates, y=snow, name='Snow', marker_color='lightblue'), row=1, col=1)

    # Accumulation
    rain_accum = np.cumsum(rain)
    snow_accum = np.cumsum(snow)
    fig_precip.add_trace(go.Scatter(x=dates, y=rain_accum, name='Rain Accumulation', line=dict(color='blue')), row=1, col=2)
    fig_precip.add_trace(go.Scatter(x=dates, y=snow_accum, name='Snow Accumulation', line=dict(color='lightblue')), row=1, col=2)

    # Precipitation type classification
    precip_type = ['Rain' if r > s else 'Snow' if s > 0 else 'None' for r, s in zip(rain, snow)]
    type_counts = pd.Series(precip_type).value_counts()
    fig_precip.add_trace(go.Bar(x=type_counts.index, y=type_counts.values, name='Precip Type'), row=2, col=1)

    # Thunder probability (using precipitation probability as proxy)
    thunder_prob = [item.get('pop', 0) * 100 for item in forecast_data['list']]
    fig_precip.add_trace(go.Scatter(x=dates, y=thunder_prob, name='Thunder Probability', line=dict(color='gold')), row=2, col=2)

    fig_precip.update_layout(height=600, title='Precipitation Analysis', showlegend=True)
    charts['precipitation'] = fig_precip

    return charts

def create_enhanced_metrics(weather_data: Dict) -> None:
    """Create comprehensive weather metrics display"""
    if not weather_data or 'main' not in weather_data:
        st.error("No weather data available")
        return

    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="🌡️ Temperature",
            value=f"{weather_data['main']['temp']:.1f}°C",
            delta=f"Feels like {weather_data['main']['feels_like']:.1f}°C"
        )

    with col2:
        st.metric(
            label="💧 Humidity",
            value=f"{weather_data['main']['humidity']}%",
            delta=f"Dew point: {weather_data.get('dew_point', 0):.1f}°C"
        )

    with col3:
        st.metric(
            label="🌬️ Wind",
            value=f"{weather_data['wind']['speed']} m/s",
            delta=f"Gusts: {weather_data['wind'].get('gust', 'N/A')} m/s"
        )

    with col4:
        st.metric(
            label="🔽 Pressure",
            value=f"{weather_data['main']['pressure']} hPa",
            delta=f"Sea level: {weather_data['main'].get('sea_level', 'N/A')} hPa"
        )

    with col5:
        visibility = weather_data.get('visibility', 0) / 1000  # Convert to km
        st.metric(
            label="👁️ Visibility",
            value=f"{visibility:.1f} km",
            delta=f"Clouds: {weather_data['clouds']['all']}%"
        )

    # Additional detailed metrics
    st.subheader("🔬 Advanced Meteorological Data")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info(f"**Wet Bulb Temperature:** {weather_data.get('wet_bulb_temp', 0):.1f}°C")
        st.info(f"**Wind Direction:** {weather_data['wind'].get('deg', 'N/A')}°")

    with col2:
        rain_1h = weather_data.get('rain', {}).get('1h', 0)
        snow_1h = weather_data.get('snow', {}).get('1h', 0)
        st.info(f"**Rain (1h):** {rain_1h} mm")
        st.info(f"**Snow (1h):** {snow_1h} mm")

    with col3:
        sunrise = datetime.fromtimestamp(weather_data['sys']['sunrise']).strftime('%H:%M')
        sunset = datetime.fromtimestamp(weather_data['sys']['sunset']).strftime('%H:%M')
        st.info(f"**Sunrise:** {sunrise}")
        st.info(f"**Sunset:** {sunset}")

    with col4:
        timezone_offset = weather_data['timezone'] // 3600
        st.info(f"**Timezone:** UTC{timezone_offset:+d}")
        st.info(f"**Coordinates:** {weather_data['coord']['lat']:.2f}, {weather_data['coord']['lon']:.2f}")

def create_city_heatmap(location: str, weather_api) -> go.Figure:
    """Create temperature heatmap for city area"""
    # This is a simplified demonstration - in reality you'd need multiple weather stations
    current_weather = weather_api.get_current_weather(location)

    if not current_weather or 'coord' not in current_weather:
        return go.Figure()

    base_lat, base_lon = current_weather['coord']['lat'], current_weather['coord']['lon']
    base_temp = current_weather['main']['temp']

    # Generate demo heatmap data around the city
    lats = np.linspace(base_lat - 0.1, base_lat + 0.1, 20)
    lons = np.linspace(base_lon - 0.1, base_lon + 0.1, 20)

    # Create temperature variations (demo data)
    temps = np.random.normal(base_temp, 2, (20, 20))

    fig = go.Figure(data=go.Heatmap(
        z=temps,
        x=lons,
        y=lats,
        colorscale='RdYlBu_r',
        colorbar=dict(title="Temperature (°C)"),
        hoverongaps=False
    ))

    fig.update_layout(
        title=f'Temperature Heatmap - {location}',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        height=500
    )

    return fig

def create_solar_power_chart(forecast_data: Dict, uv_data: Dict = None) -> go.Figure:
    """Create solar power potential chart"""
    if not forecast_data:
        return go.Figure()

    dates = [datetime.fromtimestamp(item['dt']) for item in forecast_data['list']]

    # Calculate solar potential based on cloud cover and time of day
    solar_potential = []
    for item in forecast_data['list']:
        dt = datetime.fromtimestamp(item['dt'])
        cloud_cover = item['clouds']['all']

        # Simple solar calculation (hour of day effect)
        hour = dt.hour
        if 6 <= hour <= 18:  # Daylight hours
            base_solar = 100 * (1 - abs(hour - 12) / 6)  # Peak at noon
            cloud_factor = (100 - cloud_cover) / 100
            solar_potential.append(base_solar * cloud_factor)
        else:
            solar_potential.append(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=solar_potential,
        mode='lines+markers',
        name='Solar Power Potential',
        line=dict(color='gold', width=3),
        fill='tonexty'
    ))

    fig.update_layout(
        title='Solar Power Generation Potential',
        xaxis_title='Date/Time',
        yaxis_title='Solar Potential (%)',
        height=400,
        yaxis=dict(range=[0, 100])
    )

    return fig

def main():
    st.title("🌤️ Advanced Weather Radar & Satellite System")
    st.markdown("### Comprehensive Weather Monitoring with AI Assistant")

    # Sidebar for API keys and settings
    with st.sidebar:
        st.header("🔧 Configuration")

        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get your API key from Google AI Studio"
        )

        openweather_api_key = st.text_input(
            "OpenWeather API Key",
            type="password",
            help="Get your API key from OpenWeatherMap"
        )

        st.header("🎯 Quick Actions")
        location_input = st.text_input("Location", placeholder="Enter city name...")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌡️ Get Weather"):
                if location_input and openweather_api_key:
                    st.session_state['quick_location'] = location_input

        with col2:
            if st.button("🌀 Track Storms"):
                st.session_state['show_hurricane_tracker'] = True

        st.header("🗺️ Map Layers")
        map_layer = st.selectbox(
            "Choose radar layer:",
            ["Temperature", "Precipitation", "Wind", "Pressure", "Clouds", "Hurricane Tracker"]
        )
        st.session_state['selected_layer'] = map_layer

    # Store API keys in session state
    if gemini_api_key:
        st.session_state['gemini_api_key'] = gemini_api_key
    if openweather_api_key:
        st.session_state['openweather_api_key'] = openweather_api_key

    # Main content
    if not openweather_api_key:
        st.warning("Please provide your OpenWeather API key in the sidebar to continue.")
        st.info("""
        **To get started:**
        1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey) (optional for AI chat)
        2. Get an OpenWeather API key from [OpenWeatherMap](https://openweathermap.org/api) (required)
        3. Enter the keys in the sidebar
        """)
        return

    # Initialize enhanced weather API
    weather_api = EnhancedWeatherAPI(openweather_api_key)

    # Initialize weather agent if Gemini key is provided
    if gemini_api_key:
        if 'weather_agent' not in st.session_state:
            st.session_state['weather_agent'] = WeatherAgent(gemini_api_key, openweather_api_key)

    # Tabs for different features
    tabs = st.tabs([
        "🗺️ Radar & Maps", "🤖 AI Chat", "🌡️ Current Weather",
        "📊 Detailed Forecast", "🌬️ Air Quality & UV", "🌀 Storm Tracker",
        "📈 Advanced Analytics", "🏙️ City Analysis"
    ])

    with tabs[0]:  # Radar & Maps
        st.header("🗺️ Weather Radar & Satellite Maps")

    location = st.text_input("Location for maps:", value=st.session_state.get('quick_location', ''))

    if location:
        weather_data = weather_api.get_current_weather(location)

        if weather_data:
            selected_layer = st.session_state.get('selected_layer', 'Temperature')

            if selected_layer == "Hurricane Tracker":
                st.subheader("🌀 Hurricane & Storm Tracker")
                hurricane_map = create_hurricane_tracker_map()
                st_folium(hurricane_map, width=700, height=500, key="hurricane_map")
            else:
                st.subheader(f"📡 Weather Radar - {selected_layer}")
                radar_map = create_weather_radar_map(weather_data, selected_layer)
                st_folium(radar_map, width=700, height=500, key=f"radar_map_{selected_layer}")

            # Optional: Map legend
            st.subheader("🎨 Map Legend & Controls")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("**Temperature Scale:**\n🔵 Cold\n🟢 Mild\n🟡 Warm\n🔴 Hot")
            with col2:
                st.info("**Precipitation:**\n🟦 Light Rain\n🟨 Moderate\n🟧 Heavy\n🟥 Extreme")
            with col3:
                st.info("**Wind Speed:**\n🟢 0-10 mph\n🟡 10-25 mph\n🟧 25-40 mph\n🔴 40+ mph")
    else:
        st.info("Enter a location to view weather maps and radar data.")


    with tabs[1]:  # AI Chat
        if not gemini_api_key:
            st.warning("Please provide your Gemini API key to use the AI chat feature.")
        else:
            st.header("🤖 AI Weather Assistant")
            st.markdown("Chat with AI about weather patterns, ask for analysis, and get recommendations!")

            # Chat interface
            if 'chat_messages' not in st.session_state:
                st.session_state['chat_messages'] = []

            # Display chat history
            for message in st.session_state['chat_messages']:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # Chat input
            user_input = st.chat_input("Ask about weather patterns, forecasts, or analysis...")

            if user_input:
                # Add user message
                st.session_state['chat_messages'].append({"role": "user", "content": user_input})

                with st.chat_message("user"):
                    st.write(user_input)

                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing weather data..."):
                        try:
                            response = st.session_state['weather_agent'].get_response(user_input)
                            st.write(response)
                            st.session_state['chat_messages'].append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            st.error(error_msg)
                            st.session_state['chat_messages'].append({"role": "assistant", "content": error_msg})

    with tabs[2]:  # Current Weather
        st.header("🌡️ Current Weather Conditions")

        location = st.text_input("Enter location:", value=st.session_state.get('quick_location', ''), key="current_weather_location")

        if location:
            weather_data = weather_api.get_current_weather(location)

            if weather_data and 'main' in weather_data:
                # Enhanced metrics display
                create_enhanced_metrics(weather_data)

                # Weather summary
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader(f"📍 {weather_data['name']}, {weather_data['sys']['country']}")

                    # Current conditions
                    st.markdown(f"""
                    **Current Conditions:** {weather_data['weather'][0]['description'].title()}

                    **Temperature Details:**
                    - Current: {weather_data['main']['temp']:.1f}°C
                    - Feels like: {weather_data['main']['feels_like']:.1f}°C
                    - Min today: {weather_data['main']['temp_min']:.1f}°C
                    - Max today: {weather_data['main']['temp_max']:.1f}°C

                    **Wind Information:**
                    - Speed: {weather_data['wind']['speed']} m/s
                    - Direction: {weather_data['wind'].get('deg', 'N/A')}°
                    - Gusts: {weather_data['wind'].get('gust', 'None')} m/s
                    """)

                with col2:
                    # Weather icon and current temp
                    icon_code = weather_data['weather'][0]['icon']
                    st.image(f"http://openweathermap.org/img/wn/{icon_code}@4x.png")
                    st.markdown(f"### {weather_data['main']['temp']:.1f}°C")
                    st.markdown(f"*{weather_data['weather'][0]['description'].title()}*")
            else:
                st.error("❌ Location not found or API error. Please check the location name.")

    with tabs[3]:  # Detailed Forecast
        st.header("📊 5-Day Detailed Forecast")

        if location:
            forecast_data = weather_api.get_forecast(location)

            if forecast_data:
                # Comprehensive charts
                charts = create_comprehensive_charts(forecast_data)

                # Display charts
                for chart_name, fig in charts.items():
                    st.plotly_chart(fig, use_container_width=True)

                # Detailed forecast table
                st.subheader("📋 Detailed Forecast Table")
                forecast_df = create_detailed_forecast_table(forecast_data)
                st.dataframe(forecast_df, use_container_width=True)
            else:
                st.error("Error fetching forecast data")

    with tabs[4]:  # Air Quality & UV
        st.header("🌬️ Air Quality & Environmental Data")

        if location:
            current_weather = weather_api.get_current_weather(location)

            if current_weather and 'coord' in current_weather:
                lat, lon = current_weather['coord']['lat'], current_weather['coord']['lon']

                col1, col2 = st.columns(2)

                with col1:
                    # Air Quality
                    air_quality_data = weather_api.get_air_quality(lat, lon)

                    if air_quality_data:
                        st.subheader("💨 Air Quality Index")

                        aqi = air_quality_data['list'][0]['main']['aqi']
                        components = air_quality_data['list'][0]['components']

                        aqi_levels = {
                            1: ("Good", "green", "😊"),
                            2: ("Fair", "lightgreen", "🙂"),
                            3: ("Moderate", "orange", "😐"),
                            4: ("Poor", "red", "😷"),
                            5: ("Very Poor", "darkred", "🤢")
                        }

                        level, color, emoji = aqi_levels.get(aqi, ("Unknown", "gray", "❓"))

                        st.markdown(f"### {emoji} {aqi} - {level}")

                        # AQI components
                        components_df = pd.DataFrame([
                            {"Pollutant": "CO", "Value": components['co'], "Unit": "μg/m³"},
                            {"Pollutant": "NO₂", "Value": components['no2'], "Unit": "μg/m³"},
                            {"Pollutant": "O₃", "Value": components['o3'], "Unit": "μg/m³"},
                            {"Pollutant": "PM2.5", "Value": components['pm2_5'], "Unit": "μg/m³"},
                            {"Pollutant": "PM10", "Value": components['pm10'], "Unit": "μg/m³"},
                            {"Pollutant": "SO₂", "Value": components['so2'], "Unit": "μg/m³"},
                            {"Pollutant": "NH₃", "Value": components['nh3'], "Unit": "μg/m³"},
                        ])

                        fig = px.bar(
                            components_df,
                            x="Pollutant",
                            y="Value",
                            title="Air Quality Components",
                            color="Value",
                            color_continuous_scale="Reds"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # UV Index
                    uv_data = weather_api.get_uv_index(lat, lon)

                    if uv_data and 'value' in uv_data:
                        st.subheader("☀️ UV Index & Solar Data")

                        uv_value = uv_data['value']

                        uv_levels = {
                            (0, 2): ("Low", "green", "😎"),
                            (3, 5): ("Moderate", "yellow", "🕶️"),
                            (6, 7): ("High", "orange", "🧴"),
                            (8, 10): ("Very High", "red", "🏖️"),
                            (11, float('inf')): ("Extreme", "purple", "🚨")
                        }

                        uv_level = "Unknown"
                        uv_color = "gray"
                        uv_emoji = "❓"
                        for (low, high), (level, color, emoji) in uv_levels.items():
                            if low <= uv_value <= high:
                                uv_level = level
                                uv_color = color
                                uv_emoji = emoji
                                break

                        st.markdown(f"### {uv_emoji} {uv_value:.1f} - {uv_level}")

                        # Solar power potential
                        solar_potential = min(uv_value * 10, 100)
                        st.metric(
                            label="🔋 Solar Power Potential",
                            value=f"{solar_potential:.0f}%"
                        )

                        # UV protection recommendations
                        if uv_value <= 2:
                            st.success("✅ Minimal protection needed")
                        elif uv_value <= 5:
                            st.info("🧴 Use sunscreen, wear hat")
                        elif uv_value <= 7:
                            st.warning("🏖️ Seek shade, use SPF 30+")
                        elif uv_value <= 10:
                            st.error("🚨 Avoid sun exposure")
                        else:
                            st.error("☠️ Extreme - stay indoors")

                # Solar power forecast
                if forecast_data:
                    st.subheader("🌞 Solar Power Forecast")
                    solar_chart = create_solar_power_chart(forecast_data, uv_data)
                    st.plotly_chart(solar_chart, use_container_width=True)

    with tabs[5]:  # Storm Tracker
        st.header("🌀 Hurricane & Storm Tracker")

        # Hurricane tracker map
        hurricane_map = create_hurricane_tracker_map()
        st_folium(hurricane_map, width=700, height=600)

        # Storm information
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌊 Active Systems")
            st.markdown("""
            **Current Active Storms** *(Demo Data)*:

            🔴 **Hurricane Demo 1**
            - Category: 2
            - Wind Speed: 96 mph (154 km/h)
            - Pressure: 965 hPa
            - Location: 25.5°N, 80.2°W
            - Movement: NW at 12 mph

            🟡 **Tropical Storm Demo 2**
            - Category: Tropical Storm
            - Wind Speed: 45 mph (72 km/h)
            - Pressure: 1005 hPa
            - Location: 18.2°N, 65.5°W
            - Movement: N at 8 mph
            """)

        with col2:
            st.subheader("📊 Storm Statistics")

            # Demo storm statistics
            storm_stats = {
                "Active Hurricanes": 1,
                "Tropical Storms": 1,
                "Tropical Depressions": 0,
                "Total Systems": 2
            }

            for stat, value in storm_stats.items():
                st.metric(stat, value)

            st.info("""
            **Real-time Data Sources:**
            - National Hurricane Center (NHC)
            - NOAA Hurricane Database
            - Global Disaster Alert System
            - European Weather Satellites
            """)

        # Storm alerts
        st.subheader("🚨 Weather Alerts & Warnings")
        st.warning("⚠️ Hurricane Watch in effect for: Demo Region (Example)")
        st.info("ℹ️ Tropical Storm Warning: Demo Coastal Area")

    with tabs[6]:  # Advanced Analytics
        st.header("📈 Advanced Weather Analytics")

        if location and forecast_data:
            # Create comprehensive analytics dashboard

            # Extract all forecast data
            dates = [datetime.fromtimestamp(item['dt']) for item in forecast_data['list']]
            temps = [item['main']['temp'] for item in forecast_data['list']]
            humidity = [item['main']['humidity'] for item in forecast_data['list']]
            pressure = [item['main']['pressure'] for item in forecast_data['list']]
            wind_speed = [item['wind']['speed'] for item in forecast_data['list']]
            rain = [item.get('rain', {}).get('3h', 0) for item in forecast_data['list']]

            # Statistical Analysis
            st.subheader("📊 Statistical Weather Analysis")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("🌡️ Avg Temperature", f"{np.mean(temps):.1f}°C")
                st.metric("📈 Max Temperature", f"{np.max(temps):.1f}°C")
                st.metric("📉 Min Temperature", f"{np.min(temps):.1f}°C")

            with col2:
                st.metric("💧 Avg Humidity", f"{np.mean(humidity):.0f}%")
                st.metric("🌬️ Avg Wind Speed", f"{np.mean(wind_speed):.1f} m/s")
                st.metric("💨 Max Wind Speed", f"{np.max(wind_speed):.1f} m/s")

            with col3:
                st.metric("🔽 Avg Pressure", f"{np.mean(pressure):.0f} hPa")
                st.metric("🌧️ Total Rain", f"{np.sum(rain):.1f} mm")
                st.metric("☔ Rainy Periods", f"{sum(1 for r in rain if r > 0)}")

            with col4:
                # Weather variability
                temp_std = np.std(temps)
                pressure_std = np.std(pressure)
                st.metric("📊 Temp Variability", f"{temp_std:.1f}°C")
                st.metric("📊 Pressure Variability", f"{pressure_std:.1f} hPa")

            # Correlation Analysis
            st.subheader("🔗 Weather Pattern Correlations")

            # Create correlation matrix
            weather_df = pd.DataFrame({
                'Temperature': temps,
                'Humidity': humidity,
                'Pressure': pressure,
                'Wind Speed': wind_speed,
                'Precipitation': rain
            })

            correlation_matrix = weather_df.corr()

            fig_corr = px.imshow(
                correlation_matrix,
                title="Weather Parameters Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Trend Analysis
            st.subheader("📈 Trend Analysis & Predictions")

            # Simple moving averages
            temp_ma = pd.Series(temps).rolling(window=8).mean()
            pressure_ma = pd.Series(pressure).rolling(window=8).mean()

            fig_trends = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Temperature Trend', 'Pressure Trend'),
                vertical_spacing=0.1
            )

            fig_trends.add_trace(
                go.Scatter(x=dates, y=temps, name='Temperature', line=dict(color='red', width=1)),
                row=1, col=1
            )
            fig_trends.add_trace(
                go.Scatter(x=dates, y=temp_ma, name='Temp Moving Average', line=dict(color='darkred', width=3)),
                row=1, col=1
            )

            fig_trends.add_trace(
                go.Scatter(x=dates, y=pressure, name='Pressure', line=dict(color='blue', width=1)),
                row=2, col=1
            )
            fig_trends.add_trace(
                go.Scatter(x=dates, y=pressure_ma, name='Pressure Moving Average', line=dict(color='darkblue', width=3)),
                row=2, col=1
            )

            fig_trends.update_layout(height=600, title="Weather Trends & Moving Averages")
            st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.info("Select a location to view advanced analytics.")

    with tabs[7]:  # City Analysis
        st.header("🏙️ City Weather Analysis")

        if location:
            # City temperature heatmap
            st.subheader("🌡️ City Temperature Heatmap")
            heatmap_fig = create_city_heatmap(location, weather_api)
            st.plotly_chart(heatmap_fig, use_container_width=True)

            # Urban weather effects
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏢 Urban Heat Island Effect")

                current_weather = weather_api.get_current_weather(location)
                if current_weather:
                    base_temp = current_weather['main']['temp']

                    # Simulate urban heat island data
                    urban_areas = [
                        {"Area": "City Center", "Temperature": base_temp + 2.5, "Type": "Commercial"},
                        {"Area": "Residential", "Temperature": base_temp + 1.0, "Type": "Residential"},
                        {"Area": "Industrial", "Temperature": base_temp + 3.0, "Type": "Industrial"},
                        {"Area": "Parks", "Temperature": base_temp - 1.5, "Type": "Green Space"},
                        {"Area": "Suburbs", "Temperature": base_temp - 0.5, "Type": "Suburban"},
                    ]

                    urban_df = pd.DataFrame(urban_areas)

                    fig_urban = px.bar(
                        urban_df,
                        x="Area",
                        y="Temperature",
                        color="Type",
                        title="Temperature Variations Across City Areas"
                    )
                    st.plotly_chart(fig_urban, use_container_width=True)

            with col2:
                st.subheader("🌱 Environmental Impact")

                if current_weather:
                    # Calculate environmental metrics
                    temp = current_weather['main']['temp']
                    humidity = current_weather['main']['humidity']
                    wind_speed = current_weather['wind']['speed']

                    # Air quality impact estimate
                    if wind_speed < 2:
                        air_dispersion = "Poor"
                    elif wind_speed < 5:
                        air_dispersion = "Moderate"
                    else:
                        air_dispersion = "Good"

                    # Heat stress index
                    heat_index = temp + (humidity / 10)
                    if heat_index < 20:
                        heat_stress = "Low"
                    elif heat_index < 30:
                        heat_stress = "Moderate"
                    else:
                        heat_stress = "High"

                    st.info(f"**Air Dispersion:** {air_dispersion}")
                    st.info(f"**Heat Stress Level:** {heat_stress}")
                    st.info(f"**Cooling Demand:** {max(0, (temp - 20) * 5):.0f}% above normal")

                    # Energy consumption estimate
                    if temp > 25:
                        cooling_demand = (temp - 25) * 10
                        st.warning(f"🔥 High cooling demand: +{cooling_demand:.0f}% energy usage")
                    elif temp < 10:
                        heating_demand = (10 - temp) * 8
                        st.warning(f"❄️ High heating demand: +{heating_demand:.0f}% energy usage")
                    else:
                        st.success("✅ Optimal temperature range - minimal HVAC demand")

        else:
            st.info("Enter a location to analyze city-specific weather patterns.")

def create_detailed_forecast_table(forecast_data: Dict) -> pd.DataFrame:
    """Create comprehensive forecast data table"""
    if not forecast_data:
        return pd.DataFrame()

    forecast_list = []
    for item in forecast_data['list'][:15]:  # Next 5 days
        temp = item['main']['temp']
        humidity = item['main']['humidity']
        dew_point = temp - ((100 - humidity) / 5)

        forecast_list.append({
            'Date/Time': datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d %H:%M'),
            'Temperature (°C)': item['main']['temp'],
            'Feels Like (°C)': item['main']['feels_like'],
            'Dew Point (°C)': dew_point,
            'Humidity (%)': item['main']['humidity'],
            'Pressure (hPa)': item['main']['pressure'],
            'Wind Speed (m/s)': item['wind']['speed'],
            'Wind Direction (°)': item['wind'].get('deg', 0),
            'Wind Gust (m/s)': item['wind'].get('gust', item['wind']['speed']),
            'Weather': item['weather'][0]['description'].title(),
            'Clouds (%)': item['clouds']['all'],
            'Rain (mm)': item.get('rain', {}).get('3h', 0),
            'Snow (mm)': item.get('snow', {}).get('3h', 0),
            'Thunder Prob (%)': item.get('pop', 0) * 100,
            'Visibility (km)': item.get('visibility', 10000) / 1000
        })

    return pd.DataFrame(forecast_list)

# Additional utility functions for enhanced features

def calculate_wind_chill(temp: float, wind_speed: float) -> float:
    """Calculate wind chill temperature"""
    if temp > 10 or wind_speed < 4.8:
        return temp

    wind_kmh = wind_speed * 3.6  # Convert m/s to km/h
    return 13.12 + 0.6215 * temp - 11.37 * (wind_kmh ** 0.16) + 0.3965 * temp * (wind_kmh ** 0.16)

def calculate_heat_index(temp: float, humidity: float) -> float:
    """Calculate heat index (feels like temperature in hot weather)"""
    if temp < 27:
        return temp

    # Heat index formula (Fahrenheit-based, converted)
    temp_f = temp * 9/5 + 32

    hi = (-42.379 + 2.04901523 * temp_f + 10.14333127 * humidity
          - 0.22475541 * temp_f * humidity - 6.83783e-3 * temp_f**2
          - 5.481717e-2 * humidity**2 + 1.22874e-3 * temp_f**2 * humidity
          + 8.5282e-4 * temp_f * humidity**2 - 1.99e-6 * temp_f**2 * humidity**2)

    return (hi - 32) * 5/9  # Convert back to Celsius

def get_precipitation_type(temp: float, humidity: float, pressure: float) -> str:
    """Determine precipitation type based on atmospheric conditions"""
    if humidity < 60:
        return "None"
    elif temp < 0:
        return "Snow"
    elif temp < 2:
        return "Sleet/Freezing Rain"
    elif humidity > 80 and pressure < 1013:
        return "Rain"
    elif humidity > 90:
        return "Heavy Rain"
    else:
        return "Light Rain"

def calculate_snow_accumulation(snow_data: List[float], temp_data: List[float]) -> List[float]:
    """Calculate cumulative snow accumulation considering melting"""
    accumulation = []
    current_snow = 0

    for i, (new_snow, temp) in enumerate(zip(snow_data, temp_data)):
        # Add new snow
        current_snow += new_snow

        # Calculate melting
        if temp > 2:  # Above 2°C, snow starts melting
            melt_rate = (temp - 2) * 0.5  # Simplified melting model
            current_snow = max(0, current_snow - melt_rate)

        accumulation.append(current_snow)

    return accumulation

def create_wind_rose(wind_speed_data: List[float], wind_dir_data: List[int]) -> go.Figure:
    """Create a wind rose chart"""
    # Define direction bins
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    # Calculate wind statistics by direction
    dir_bins = np.linspace(0, 360, 17)
    wind_by_dir = {direction: [] for direction in directions}

    for speed, direction in zip(wind_speed_data, wind_dir_data):
        dir_index = int((direction + 11.25) / 22.5) % 16
        wind_by_dir[directions[dir_index]].append(speed)

    # Calculate average wind speed for each direction
    avg_speeds = [np.mean(wind_by_dir[d]) if wind_by_dir[d] else 0 for d in directions]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=avg_speeds,
        theta=directions,
        fill='toself',
        name='Average Wind Speed'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(avg_speeds) * 1.1] if avg_speeds else [0, 10]
            )),
        showlegend=False,
        title="Wind Rose - Average Wind Speed by Direction"
    )

    return fig

def create_pressure_trend_analysis(pressure_data: List[float], dates: List[datetime]) -> go.Figure:
    """Create pressure trend analysis with weather system indicators"""
    fig = go.Figure()

    # Main pressure line
    fig.add_trace(go.Scatter(
        x=dates,
        y=pressure_data,
        mode='lines+markers',
        name='Pressure',
        line=dict(color='blue', width=2)
    ))

    # Add pressure change indicators
    pressure_changes = np.diff(pressure_data)
    rising_pressure = []
    falling_pressure = []

    for i, change in enumerate(pressure_changes):
        if change > 2:  # Rising pressure
            rising_pressure.append((dates[i+1], pressure_data[i+1]))
        elif change < -2:  # Falling pressure
            falling_pressure.append((dates[i+1], pressure_data[i+1]))

    if rising_pressure:
        rising_dates, rising_values = zip(*rising_pressure)
        fig.add_trace(go.Scatter(
            x=rising_dates,
            y=rising_values,
            mode='markers',
            name='Rising Pressure',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))

    if falling_pressure:
        falling_dates, falling_values = zip(*falling_pressure)
        fig.add_trace(go.Scatter(
            x=falling_dates,
            y=falling_values,
            mode='markers',
            name='Falling Pressure',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))

    # Add weather system indicators
    avg_pressure = np.mean(pressure_data)
    fig.add_hline(y=1013.25, line_dash="dash", line_color="gray",
                  annotation_text="Sea Level Standard")
    fig.add_hline(y=avg_pressure, line_dash="dot", line_color="orange",
                  annotation_text=f"Average: {avg_pressure:.1f} hPa")

    fig.update_layout(
        title='Pressure Trend Analysis with Weather System Indicators',
        xaxis_title='Date/Time',
        yaxis_title='Pressure (hPa)',
        height=400
    )

    return fig

def create_thunderstorm_probability_map(forecast_data: Dict, weather_api) -> folium.Map:
    """Create thunderstorm probability visualization"""
    if not forecast_data or 'city' not in forecast_data:
        return folium.Map(location=[0, 0], zoom_start=2)

    lat = forecast_data['city']['coord']['lat']
    lon = forecast_data['city']['coord']['lon']

    # Create base map
    m = folium.Map(location=[lat, lon], zoom_start=6)

    # Add thunderstorm probability circles
    for item in forecast_data['list'][:8]:  # Next 24 hours
        thunder_prob = item.get('pop', 0)

        if thunder_prob > 0.3:  # Only show significant probabilities
            # Calculate position offset for time progression
            time_offset = forecast_data['list'].index(item) * 0.01

            color = 'red' if thunder_prob > 0.7 else 'orange' if thunder_prob > 0.5 else 'yellow'

            folium.Circle(
                [lat + time_offset, lon + time_offset],
                radius=thunder_prob * 50000,  # Scale radius
                popup=f"Thunder Probability: {thunder_prob*100:.0f}%<br>Time: {datetime.fromtimestamp(item['dt']).strftime('%H:%M')}",
                color=color,
                fill=True,
                fillOpacity=0.3
            ).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 150px; height: 90px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; ">
    <p><b>Thunder Probability</b></p>
    <p><i class="fa fa-circle" style="color:red"></i> High (&gt;70%)</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Moderate (50-70%)</p>
    <p><i class="fa fa-circle" style="color:yellow"></i> Low (30-50%)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

if __name__ == "__main__":
    # Add required imports at the top
    try:
        import streamlit_folium
    except ImportError:
        st.error("Please install streamlit-folium: pip install streamlit-folium")
        st.stop()

    try:
        import folium.plugins
    except ImportError:
        st.error("Please install folium with plugins: pip install folium")
        st.stop()

    main()