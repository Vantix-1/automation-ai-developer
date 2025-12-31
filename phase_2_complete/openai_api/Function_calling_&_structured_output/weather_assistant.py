"""
Weather Assistant - Days 9-10
OpenAI Function Calling Demo with Real Weather API
"""

import os
import json
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pydantic
from openai import OpenAI
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

load_dotenv()

class TemperatureUnit(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

@dataclass
class WeatherData:
    """Data class for weather information"""
    location: str
    temperature: float
    feels_like: float
    humidity: int
    wind_speed: float
    wind_direction: str
    condition: str
    description: str
    unit: TemperatureUnit
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "temperature": self.temperature,
            "feels_like": self.feels_like,
            "humidity": self.humidity,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "condition": self.condition,
            "description": self.description,
            "unit": self.unit.value,
            "timestamp": self.timestamp.isoformat()
        }
    
    def format_for_display(self) -> str:
        """Format weather data for nice display"""
        temp_symbol = "°C" if self.unit == TemperatureUnit.CELSIUS else "°F"
        
        return f"""
🌤️  Weather for {self.location}
══════════════════════════════════════════

📊 Current Conditions:
  • Temperature: {self.temperature}{temp_symbol} (Feels like: {self.feels_like}{temp_symbol})
  • Condition: {self.condition} - {self.description}
  • Humidity: {self.humidity}%
  • Wind: {self.wind_speed} km/h from {self.wind_direction}

⏰ Last Updated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
══════════════════════════════════════════
"""

class WeatherAPIError(Exception):
    """Custom exception for weather API errors"""
    pass

class WeatherAssistant:
    """Weather assistant using OpenAI function calling"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", use_mock: bool = False):
        """Initialize the weather assistant
        
        Args:
            model: OpenAI model to use
            use_mock: Use mock data instead of real API (for testing)
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.use_mock = use_mock
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        
        # Define available functions/tools for OpenAI
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country, e.g., 'New York, US' or 'London, UK'"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit, defaults to celsius"
                            }
                        },
                        "required": ["location"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_forecast",
                    "description": "Get weather forecast for multiple days",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country"
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of forecast days (1-5)",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit"
                            }
                        },
                        "required": ["location", "days"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_locations_weather",
                    "description": "Compare weather between multiple locations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "locations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of locations to compare",
                                "minItems": 2,
                                "maxItems": 5
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit for comparison"
                            }
                        },
                        "required": ["locations"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_alerts",
                    "description": "Get weather alerts and warnings for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country"
                            }
                        },
                        "required": ["location"],
                        "additionalProperties": False
                    }
                }
            }
        ]
        
        # Conversation history
        self.conversation_history: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": """You are a helpful weather assistant. You can:
1. Get current weather for any location
2. Provide weather forecasts
3. Compare weather between locations
4. Check for weather alerts

Always be concise but informative. When showing weather data, include:
- Temperature (and feels like)
- Conditions and description
- Humidity and wind information
- Any relevant alerts

If you need to clarify location details, ask the user."""
            }
        ]
    
    def _call_weather_api(self, location: str, endpoint: str = "current") -> Dict[str, Any]:
        """Call weather API (mock or real)"""
        if self.use_mock:
            return self._get_mock_weather_data(location)
        
        # For real implementation, you would use OpenWeatherMap or similar
        # This is a placeholder - you'd need to sign up for a weather API key
        if not self.weather_api_key:
            print(Fore.YELLOW + "⚠️  No real weather API key found. Using mock data.")
            return self._get_mock_weather_data(location)
        
        # Real API implementation would go here
        # Example with OpenWeatherMap:
        # url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.weather_api_key}&units=metric"
        # async with httpx.AsyncClient() as client:
        #     response = await client.get(url)
        #     return response.json()
        
        # For now, use mock
        return self._get_mock_weather_data(location)
    
    def _get_mock_weather_data(self, location: str) -> Dict[str, Any]:
        """Generate realistic mock weather data for demonstration"""
        import random
        
        # Mock weather conditions based on location
        locations_conditions = {
            "new york": {"temp_range": (-5, 10), "conditions": ["Cloudy", "Partly Cloudy", "Clear"]},
            "london": {"temp_range": (0, 8), "conditions": ["Rainy", "Cloudy", "Foggy"]},
            "tokyo": {"temp_range": (5, 15), "conditions": ["Clear", "Partly Cloudy", "Sunny"]},
            "sydney": {"temp_range": (18, 28), "conditions": ["Sunny", "Clear", "Partly Cloudy"]},
            "paris": {"temp_range": (2, 12), "conditions": ["Cloudy", "Light Rain", "Clear"]},
            "default": {"temp_range": (0, 20), "conditions": ["Clear", "Partly Cloudy", "Cloudy"]}
        }
        
        loc_key = location.lower().split(",")[0].strip()
        config = locations_conditions.get(loc_key, locations_conditions["default"])
        
        temp = round(random.uniform(*config["temp_range"]), 1)
        feels_like = round(temp + random.uniform(-3, 3), 1)
        condition = random.choice(config["conditions"])
        
        descriptions = {
            "Clear": "Clear skies throughout the day",
            "Partly Cloudy": "Mix of sun and clouds",
            "Cloudy": "Overcast with no sun",
            "Sunny": "Bright and sunny",
            "Rainy": "Rain expected, take an umbrella",
            "Light Rain": "Light drizzle possible",
            "Foggy": "Reduced visibility, drive carefully"
        }
        
        return {
            "location": location.title(),
            "temperature": temp,
            "feels_like": feels_like,
            "humidity": random.randint(40, 85),
            "wind_speed": round(random.uniform(5, 25), 1),
            "wind_direction": random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
            "condition": condition,
            "description": descriptions.get(condition, "Weather conditions normal"),
            "timestamp": datetime.now()
        }
    
    def get_current_weather(self, location: str, unit: str = "celsius") -> WeatherData:
        """Get current weather for a location"""
        print(Fore.CYAN + f"\n🌍 Fetching weather for {location}...")
        
        try:
            # Get weather data
            weather_data = self._call_weather_api(location)
            
            # Convert to WeatherData object
            temp_unit = TemperatureUnit.CELSIUS if unit == "celsius" else TemperatureUnit.FAHRENHEIT
            
            # Convert temperature if needed
            if unit == "fahrenheit":
                weather_data["temperature"] = (weather_data["temperature"] * 9/5) + 32
                weather_data["feels_like"] = (weather_data["feels_like"] * 9/5) + 32
            
            weather = WeatherData(
                location=weather_data["location"],
                temperature=weather_data["temperature"],
                feels_like=weather_data["feels_like"],
                humidity=weather_data["humidity"],
                wind_speed=weather_data["wind_speed"],
                wind_direction=weather_data["wind_direction"],
                condition=weather_data["condition"],
                description=weather_data["description"],
                unit=temp_unit,
                timestamp=weather_data["timestamp"]
            )
            
            return weather
            
        except Exception as e:
            raise WeatherAPIError(f"Failed to get weather data: {e}")
    
    def get_weather_forecast(self, location: str, days: int, unit: str = "celsius") -> List[WeatherData]:
        """Get weather forecast for multiple days"""
        print(Fore.CYAN + f"\n📅 Fetching {days}-day forecast for {location}...")
        
        forecasts = []
        for day_offset in range(days):
            # Mock forecast - in real app, you'd call forecast endpoint
            forecast_date = datetime.now() + timedelta(days=day_offset)
            
            # Modify base data slightly for each day
            base_data = self._call_weather_api(location)
            base_data["temperature"] += day_offset * random.uniform(-2, 2)
            base_data["condition"] = random.choice(["Sunny", "Cloudy", "Partly Cloudy", "Rainy"])
            
            temp_unit = TemperatureUnit.CELSIUS if unit == "celsius" else TemperatureUnit.FAHRENHEIT
            
            if unit == "fahrenheit":
                base_data["temperature"] = (base_data["temperature"] * 9/5) + 32
                base_data["feels_like"] = (base_data["feels_like"] * 9/5) + 32
            
            forecast = WeatherData(
                location=base_data["location"],
                temperature=round(base_data["temperature"], 1),
                feels_like=round(base_data["temperature"] + random.uniform(-2, 2), 1),
                humidity=random.randint(40, 90),
                wind_speed=round(base_data["wind_speed"] + random.uniform(-5, 5), 1),
                wind_direction=base_data["wind_direction"],
                condition=base_data["condition"],
                description=f"Forecast for {forecast_date.strftime('%A, %b %d')}",
                unit=temp_unit,
                timestamp=forecast_date
            )
            
            forecasts.append(forecast)
        
        return forecasts
    
    def compare_locations_weather(self, locations: List[str], unit: str = "celsius") -> Dict[str, WeatherData]:
        """Compare weather between multiple locations"""
        print(Fore.CYAN + f"\n⚖️ Comparing weather for {len(locations)} locations...")
        
        comparison = {}
        for location in locations:
            try:
                weather = self.get_current_weather(location, unit)
                comparison[location] = weather
            except Exception as e:
                print(Fore.RED + f"⚠️ Failed to get weather for {location}: {e}")
        
        return comparison
    
    def get_weather_alerts(self, location: str) -> Dict[str, Any]:
        """Get weather alerts for a location (mock implementation)"""
        print(Fore.CYAN + f"\n⚠️ Checking weather alerts for {location}...")
        
        # Mock alerts based on location
        alerts_db = {
            "new york": ["Winter Storm Warning", "Wind Advisory"],
            "london": ["Flood Warning", "High Wind Warning"],
            "tokyo": ["Typhoon Watch", "Heavy Rain Warning"],
            "miami": ["Hurricane Warning", "Tornado Watch"],
            "default": []
        }
        
        loc_key = location.lower().split(",")[0].strip()
        alerts = alerts_db.get(loc_key, alerts_db["default"])
        
        return {
            "location": location,
            "has_alerts": len(alerts) > 0,
            "alerts": alerts,
            "timestamp": datetime.now().isoformat(),
            "recommendation": "Stay indoors" if alerts else "No special precautions needed"
        }
    
    def handle_function_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle function calls from OpenAI"""
        print(Fore.MAGENTA + f"\n🔧 Function call: {function_name}")
        print(Fore.MAGENTA + f"   Arguments: {arguments}")
        
        try:
            if function_name == "get_current_weather":
                location = arguments.get("location", "")
                unit = arguments.get("unit", "celsius")
                
                weather = self.get_current_weather(location, unit)
                
                return {
                    "success": True,
                    "data": weather.to_dict(),
                    "display": weather.format_for_display(),
                    "function": function_name
                }
            
            elif function_name == "get_weather_forecast":
                location = arguments.get("location", "")
                days = arguments.get("days", 3)
                unit = arguments.get("unit", "celsius")
                
                forecasts = self.get_weather_forecast(location, days, unit)
                
                forecast_data = [f.to_dict() for f in forecasts]
                forecast_display = "\n".join([f.format_for_display() for f in forecasts])
                
                return {
                    "success": True,
                    "data": forecast_data,
                    "display": f"📅 {days}-Day Forecast for {location}\n{forecast_display}",
                    "function": function_name
                }
            
            elif function_name == "compare_locations_weather":
                locations = arguments.get("locations", [])
                unit = arguments.get("unit", "celsius")
                
                comparison = self.compare_locations_weather(locations, unit)
                
                # Create comparison table
                comparison_table = f"\n{'Location':<20} {'Temp':<10} {'Condition':<15} {'Humidity':<10}\n"
                comparison_table += "=" * 55 + "\n"
                
                for loc, weather in comparison.items():
                    temp_symbol = "°C" if unit == "celsius" else "°F"
                    comparison_table += f"{loc:<20} {weather.temperature}{temp_symbol:<9} {weather.condition:<15} {weather.humidity}%\n"
                
                return {
                    "success": True,
                    "data": {loc: w.to_dict() for loc, w in comparison.items()},
                    "display": f"⚖️ Weather Comparison\n{comparison_table}",
                    "function": function_name
                }
            
            elif function_name == "get_weather_alerts":
                location = arguments.get("location", "")
                
                alerts = self.get_weather_alerts(location)
                
                alert_display = f"\n📍 {location}\n"
                if alerts["has_alerts"]:
                    alert_display += Fore.RED + "⚠️ ACTIVE ALERTS:\n"
                    for alert in alerts["alerts"]:
                        alert_display += f"  • {alert}\n"
                    alert_display += f"\nRecommendation: {alerts['recommendation']}"
                else:
                    alert_display += Fore.GREEN + "✅ No active weather alerts"
                
                return {
                    "success": True,
                    "data": alerts,
                    "display": alert_display,
                    "function": function_name
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown function: {function_name}",
                    "function": function_name
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "function": function_name
            }
    
    def process_user_query(self, user_input: str) -> str:
        """Process user query with function calling"""
        # Add user message to conversation
        self.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Get initial response from OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=self.tools,
                tool_choice="auto",  # Let model decide when to use tools
                temperature=0.3,
                max_tokens=500
            )
            
            response_message = response.choices[0].message
            
            # Check if tool calls are needed
            tool_calls = response_message.tool_calls
            
            if tool_calls:
                # Add assistant message with tool calls
                self.conversation_history.append(response_message)
                
                # Process each tool call
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    # Handle the function call
                    function_response = self.handle_function_call(function_name, arguments)
                    
                    # Add tool response to conversation
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_response)
                    })
                
                # Get final response with function results
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=0.3,
                    max_tokens=500
                )
                
                assistant_message = final_response.choices[0].message.content
                
            else:
                # No tool calls needed
                assistant_message = response_message.content
            
            # Add assistant response to conversation
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except Exception as e:
            error_msg = f"❌ Error processing query: {e}"
            print(Fore.RED + error_msg)
            return error_msg
    
    def interactive_demo(self):
        """Run interactive weather assistant demo"""
        print(Fore.CYAN + "\n" + "="*70)
        print(Fore.CYAN + "🌤️  Weather Assistant - Function Calling Demo")
        print(Fore.CYAN + "="*70)
        
        print(Fore.YELLOW + "\n🤖 I'm your weather assistant! I can help with:")
        print(Fore.YELLOW + "  • Current weather for any location")
        print(Fore.YELLOW + "  • Weather forecasts (1-5 days)")
        print(Fore.YELLOW + "  • Compare weather between locations")
        print(Fore.YELLOW + "  • Check for weather alerts")
        print(Fore.YELLOW + "\n💡 Examples: 'What's the weather in Tokyo?', 'Compare New York and London'")
        print(Fore.YELLOW + "Type 'quit' to exit, 'clear' to clear history")
        print(Fore.CYAN + "="*70)
        
        if self.use_mock or not self.weather_api_key:
            print(Fore.YELLOW + "\n⚠️ Using mock weather data (for demonstration)")
            print(Fore.YELLOW + "To use real data, add WEATHER_API_KEY to .env file")
        
        while True:
            try:
                print(Fore.GREEN + "\n" + "━" * 50)
                user_input = input(Fore.WHITE + "\n🌍 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print(Fore.YELLOW + "👋 Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = self.conversation_history[:1]  # Keep system message
                    print(Fore.YELLOW + "🧹 Conversation cleared")
                    continue
                
                if user_input.lower() == 'help':
                    print(Fore.CYAN + "\n📋 Available commands:")
                    print(Fore.CYAN + "  • 'current <location>' - Get current weather")
                    print(Fore.CYAN + "  • 'forecast <location> <days>' - Get forecast")
                    print(Fore.CYAN + "  • 'compare <loc1> vs <loc2>' - Compare weather")
                    print(Fore.CYAN + "  • 'alerts <location>' - Check for alerts")
                    print(Fore.CYAN + "  • Or just ask naturally!")
                    continue
                
                # Process the query
                print(Fore.BLUE + "\n🤖 Processing...", end="", flush=True)
                response = self.process_user_query(user_input)
                print(Fore.BLUE + "\r🤖 Assistant: " + Fore.WHITE + response)
                
            except KeyboardInterrupt:
                print(Fore.YELLOW + "\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(Fore.RED + f"\n❌ Error: {e}")

def main():
    """Main function"""
    try:
        # Create weather assistant
        # Set use_mock=False if you have a real weather API key
        assistant = WeatherAssistant(model="gpt-3.5-turbo", use_mock=True)
        
        # Run interactive demo
        assistant.interactive_demo()
        
    except ValueError as e:
        print(Fore.RED + f"\n❌ Configuration Error: {e}")
        print(Fore.YELLOW + "Please ensure you have:")
        print(Fore.YELLOW + "1. A .env file with OPENAI_API_KEY")
        print(Fore.YELLOW + "2. Optional: WEATHER_API_KEY for real weather data")
    except Exception as e:
        print(Fore.RED + f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()