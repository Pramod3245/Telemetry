import os
import requests
import asyncio
import json

from serpapi import GoogleSearch

from autogen_agentchat.agents import AssistantAgent

from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def weather_tool(city: str) -> str:
    with tracer.start_as_current_span("tool.weather_tool", kind=trace.SpanKind.CLIENT) as span: # Set span kind
        span.set_attribute("weather.city", city)
        # config_key = os.getenv("WEATHER_API_APIKEY")
        config_key= "45408d486ba24acb91c44416253004"
        if not config_key:
             span.set_attribute("otel.status_code", "ERROR")
             span.set_attribute("error.message", "WEATHER_API_APIKEY not set")
             return "Error: WEATHER_API_APIKEY not set"

        url = "http://api.weatherapi.com/v1/current.json"
        params = {"key": config_key, "q": city, "aqi": "no"}

        try:
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            temp = data["current"]["temp_c"]
            condition = data["current"]["condition"]["text"]
            result = f"The current weather in {city} is {condition} with a temperature of {temp}°C."
            span.set_attribute("otel.status_code", "OK")
            span.set_attribute("weather.temperature_c", temp)
            span.set_attribute("weather.condition", condition)
            span.set_attribute("tool.output", result) # Keep tool specific output attribute
            return result
        except requests.exceptions.RequestException as e:
            error_msg = f"Weather tool request error: {e}"
            span.set_attribute("otel.status_code", "ERROR")
            span.set_attribute("exception.type", type(e).__name__)
            span.set_attribute("error.message", error_msg)
            span.record_exception(e)
            return error_msg
        except Exception as e:
            error_msg = f"Weather tool error: {e}"
            span.set_attribute("otel.status_code", "ERROR")
            span.set_attribute("exception.type", type(e).__name__)
            span.set_attribute("error.message", error_msg)
            span.record_exception(e)
            return error_msg

async def web_search(query: str) -> str:
    with tracer.start_as_current_span("tool.web_search", kind=trace.SpanKind.CLIENT) as span: # Set span kind
        span.set_attribute("search.query", query)
        serp_api_key = os.getenv("SERP_API_KEY")
        if not serp_api_key:
            error_msg = "Error: SERP_API_KEY not configured."
            span.set_attribute("otel.status_code", "ERROR")
            span.set_attribute("error.message", error_msg)
            return error_msg

        params = {
            "q": query,
            "api_key": serp_api_key,
            "location": "Earth",
            "hl": "en",
            "gl": "in",
            "num": 3,
        }

        try:
            search = GoogleSearch(params)
            loop = asyncio.get_running_loop()
            # Internal span for the blocking call within the tool
            with tracer.start_as_current_span("GoogleSearch.get_dict", kind=trace.SpanKind.INTERNAL):
                 results = await loop.run_in_executor(None, search.get_dict)

            output = f"Search results for '{query}':\n"
            organic_results = results.get("organic_results", [])
            if not organic_results:
                 output = "No results found."
                 span.set_attribute("search.results_count", 0)
            else:
                for i, r in enumerate(organic_results[:params["num"]], 1):
                    title = r.get("title", "No Title")
                    snippet = r.get("snippet", "")
                    link = r.get("link", "No Link")
                    output += f"{i}. {title}: {snippet}\n"
                    span.add_event(f"result_{i}", attributes={"title": title, "snippet": snippet[:100] + '...' if len(snippet) > 100 else snippet, "link": link})

                span.set_attribute("search.results_count", len(organic_results))

            span.set_attribute("otel.status_code", "OK")
            span.set_attribute("tool.output_summary", output[:500] + '...' if len(output) > 500 else output) # Keep tool specific output attribute
            return output

        except Exception as e:
            error_msg = f"Web search error: {e}"
            span.set_attribute("otel.status_code", "ERROR")
            span.set_attribute("exception.type", type(e).__name__)
            span.set_attribute("exception.message", error_msg)
            span.record_exception(e)
            return error_msg

def create_agents(model_client1, model_client2):
     planning_agent = AssistantAgent(
        name="PlanningAgent",
        description="Plans tasks and delegates.",
        model_client=model_client1,
        system_message="""
        You are a planner. Delegate tasks to agents:
        - WeatherAgent: For weather queries.
        - WebSearchAgent: For general information lookups.
        After receiving results from other agents, synthesize the information, summarize the overall outcome, and end your final message with 'TERMINATE'.
        If a tool call fails or an agent reports an error, acknowledge it and try to proceed if possible, or report the failure and summarize before terminating.
        """,
     )

     weather_agent = AssistantAgent(
        name="WeatherAgent",
        description="Fetches current weather using weather_tool.",
        tools=[weather_tool],
        model_client=model_client2,
        system_message="""
        You are the WeatherAgent. Use the 'weather_tool' ONLY to get weather information for the requested city.
        Once you get the result from the tool, return it directly and clearly state the information.
        If the tool reports an error, return the exact error message received from the tool.
        Do not try to answer weather questions directly or use other tools.
        """,
        )

     web_search_agent = AssistantAgent(
        name="WebSearchAgent",
        description="Searches the web using web_search.",
        tools=[web_search],
        model_client=model_client1,
        system_message="""
        You are the WebSearchAgent. Use the 'web_search' tool ONLY for all information lookups requested.
        Once you get the search results from the tool, summarize the key findings concisely and return the summary.
        If the tool reports an error, return the exact error message received from the tool.
        Do not speculate or use other tools.
        """,
        )

     return planning_agent, weather_agent, web_search_agent