import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console

from autogen_ext.models.openai import OpenAIChatCompletionClient

from telemetry_wrapper import initialize_telemetry, wrap_agent_with_telemetry

from agents import create_agents

from opentelemetry import trace

load_dotenv()

tracer = initialize_telemetry(service_name="AutoGen_Example_Team", output_file="telemetry_output_V2.json")

async def main():
    print("--- Initializing AutoGen Team with Telemetry ---")

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    placeholder_key = "AIzaSyC8Q7GccF-P1b_AIAcwuZNHsA0Psou5mXs"

    if not gemini_api_key:
        print("Warning: GEMINI_API_KEY not set from .env. Using placeholder for model_client1. REPLACE THIS.")
        gemini_api_key = placeholder_key

    model_client1 = OpenAIChatCompletionClient(
        model="gemini-1.5-flash",
        api_key=gemini_api_key,
        api_type="google"
    )

    model_client2_key = os.getenv("GEMINI_API_KEY_2")
    if not model_client2_key:
         print("Warning: GEMINI_API_KEY_2 not set from .env. Using placeholder for model_client2. REPLACE THIS.")
         model_client2_key = placeholder_key


    model_client2 = OpenAIChatCompletionClient(
        model="gemini-1.5-flash",
        api_key=model_client2_key,
        api_type="google"
    )

    planning_agent, weather_agent, web_search_agent = create_agents(model_client1, model_client2)

    print("--- Wrapping Agents with Telemetry ---")
    try:
        wrap_agent_with_telemetry(planning_agent, tracer)
        wrap_agent_with_telemetry(weather_agent, tracer)
        wrap_agent_with_telemetry(web_search_agent, tracer)
        print("--- Agents Wrapped Successfully ---")
    except TypeError as e:
        print(f"Error wrapping agents: {e}")
        print("Please ensure the objects created by create_agents() have the expected async message processing method (currently targeting 'on_messages').")
        return


    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=25)

    selector_prompt = """Select an agent to perform the task.

Available agents and their descriptions:
{roles}

Current conversation history:
{history}

Based on the latest message, carefully choose ONE agent from {participants} whose description best matches the required task.
Start the conversation by selecting the PlanningAgent to interpret the initial request.
"""

    task_weather = "What is the current weather at Hyderabad now?"
    task_search = "Who is the current President of USA?"
    task_combined = "Tell me the weather in London and search for the current Prime Minister of the UK."

    chosen_task = task_search

    print(f"\n--- Running Task: {chosen_task} ---")

    with tracer.start_as_current_span("AgentTeamRuntime", kind=trace.SpanKind.INTERNAL) as runtime_span: # Set span kind
         runtime_span.set_attribute("task", chosen_task)

         team = SelectorGroupChat(
            [planning_agent, weather_agent, web_search_agent],
            model_client=model_client1,
            termination_condition=termination,
            selector_prompt=selector_prompt,
            allow_repeated_speaker=True
         )
         await Console(team.run_stream(task=chosen_task))
         
    print("\n--- Task Finished ---")

    print("--- Shutting down ---")
    try:
        await model_client1.close()
        await model_client2.close()
    except Exception as e:
        print(f"Error closing model clients: {e}")

    try:
        trace.get_tracer_provider().shutdown()
        print(f"Telemetry data exported to telemetry_output_V2.json")
    except Exception as e:
        print(f"Error shutting down telemetry: {e}")

if __name__ == "__main__":
    asyncio.run(main())