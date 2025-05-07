import json
import datetime
import asyncio
import inspect
from typing import Any, Dict, Optional, List

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.util.instrumentation import InstrumentationScope

class FileSpanExporter(SpanExporter):
    def __init__(self, file_path):
        self.file_path = file_path
        self.spans = []

    def export(self, spans):
        for span_data in spans:
            span_dict = {
                "name": span_data.name,
                "trace_id": format(span_data.context.trace_id, '032x'),
                "span_id": format(span_data.context.span_id, '016x'),
                "parent_id": format(span_data.parent.span_id, '016x') if span_data.parent else None,
                "start_time": self._format_timestamp(span_data.start_time),
                "end_time": self._format_timestamp(span_data.end_time),
                "duration_ns": span_data.end_time - span_data.start_time,
                "attributes": dict(span_data.attributes or {}),
                "events": [
                    {
                        "name": event.name,
                        "timestamp": self._format_timestamp(event.timestamp),
                        "attributes": dict(event.attributes or {})
                    } for event in span_data.events
                ],
                "status": {
                    "status_code": span_data.status.status_code.name,
                    "description": span_data.status.description
                },
                "kind": span_data.kind.name,
                "resource": {key: value for key, value in span_data.resource.attributes.items()} if span_data.resource and hasattr(span_data.resource, 'attributes') and isinstance(span_data.resource.attributes, dict) else {},
                 "instrumentation_scope": {
                    "name": span_data.instrumentation_scope.name,
                    "version": span_data.instrumentation_scope.version,
                    "schema_url": span_data.instrumentation_scope.schema_url,
                 } if isinstance(span_data.instrumentation_scope, InstrumentationScope) else {},
            }
            self.spans.append(span_dict)
        return True

    def shutdown(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.spans, f, indent=2)
        self.spans.clear()

    def _format_timestamp(self, timestamp_ns):
        dt = datetime.datetime.fromtimestamp(timestamp_ns / 1e9, tz=datetime.timezone.utc)
        return dt.isoformat().replace('+00:00', 'Z')

def initialize_telemetry(service_name="AutoGen_Team", output_file="telemetry_output.json"):
    resource = Resource(attributes={
        "service.name": service_name,
        "service.version": "1.0",
        "deployment.environment": "development"
    })
    provider = TracerProvider(resource=resource)
    file_exporter = FileSpanExporter(output_file)
    processor = BatchSpanProcessor(file_exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)

def get_agent_role(agent_name: str) -> str:
    # Simple mapping based on agent names defined in agents.py
    # You might need to customize this or add a 'role' attribute to your agents
    if agent_name == "PlanningAgent":
        return "orchestrator" # or 'planner'
    elif agent_name in ["WeatherAgent", "WebSearchAgent"]:
        return "executor" # or 'responder'
    else:
        return "unknown"

def wrap_agent_with_telemetry(agent, tracer: trace.Tracer):
    method_name = 'on_messages'

    if not hasattr(agent, method_name) or not inspect.iscoroutinefunction(getattr(agent, method_name)):
        raise TypeError(f"Object {type(agent).__name__} does not have an async '{method_name}' method and cannot be wrapped.")

    original_method = getattr(agent, method_name)

    async def wrapped_method(*args, **kwargs):
        # Extract relevant info from arguments
        sender = kwargs.get('sender') or (args[2] if len(args) > 2 else None)
        messages = kwargs.get('messages') or (args[1] if len(args) > 1 else [])
        latest_message = messages[-1] if messages and isinstance(messages[-1], dict) else {}
        latest_message_content = latest_message.get('content', '')

        agent_name = getattr(agent, 'name', type(agent).__name__)
        span_name = f"messaging.process {agent_name}" # Use messaging convention for span name

        with tracer.start_as_current_span(span_name, kind=trace.SpanKind.CONSUMER) as span: # Set span kind
            # Add Semantic Attributes
            span.set_attribute("messaging.operation", "process")
            span.set_attribute("messaging.destination", agent_name) # Agent name as destination
            span.set_attribute("ai.agent.name", agent_name)
            span.set_attribute("ai.agent.role", get_agent_role(agent_name))

            # Add input message details as attributes
            # Guess message type based on content keywords or structure
            message_type = "unknown"
            if 'tool_code' in latest_message_content:
                message_type = "tool_code"
            elif 'tool_response' in latest_message_content:
                 message_type = "tool_response"
            elif messages and len(messages) == 1: # Assuming the first message is the initial request
                 message_type = "initial_request"
            else:
                 message_type = "plain_text"

            span.set_attribute("messaging.message.type", message_type)
            span.set_attribute("message.input.content", latest_message_content[:1000] + '...' if len(latest_message_content) > 1000 else latest_message_content) # Capture input content

            # Optional: Add sender info
            span.set_attribute("messaging.sender.name", getattr(sender, 'name', str(sender) if sender else 'Unknown'))

            # ai.input.hash and ai.session_id are harder without more context, skipping for now

            try:
                result = await original_method(*args, **kwargs)

                # Add Output Details and Token Usage (Attempt to extract)
                output_content = None
                token_usage = None # Initialize token usage

                if isinstance(result, str):
                     output_content = result
                elif isinstance(result, dict):
                     # Try to extract content from dict result
                     output_content = result.get('content')

                     # Attempt to extract token usage from dict result (common structure)
                     usage_data = result.get('usage', {})
                     if isinstance(usage_data, dict):
                         prompt_tokens = usage_data.get('prompt_tokens')
                         completion_tokens = usage_data.get('completion_tokens')
                         total_tokens = usage_data.get('total_tokens')

                         if prompt_tokens is not None:
                             span.set_attribute("token.usage.prompt", prompt_tokens)
                         if completion_tokens is not None:
                             span.set_attribute("token.usage.completion", completion_tokens)
                         if total_tokens is not None:
                             span.set_attribute("token.usage.total", total_tokens)
                             token_usage = total_tokens # Indicate usage was found

                     # Check for tool calls in output
                     tool_calls = result.get('tool_calls')
                     if isinstance(tool_calls, list):
                          span.set_attribute("tool_calls_count", len(tool_calls))
                          # Add summary of tool calls if needed in output_content
                          if not output_content: # If no content was generated, show tool calls
                             output_content = f"Tool calls: {json.dumps(tool_calls)[:500]}..."


                elif isinstance(result, list) and result and isinstance(result[-1], dict):
                     # If result is a list of messages, get last message content
                     output_content = result[-1].get('content')
                     span.set_attribute("messages.returned_count", len(result))
                     # Token usage might be on individual message dicts or not present here


                if output_content is not None:
                    span.set_attribute("message.output.content", output_content[:1000] + '...' if len(output_content) > 1000 else output_content)
 

                # Set Status Code based on success and potential token usage
                if token_usage is not None: # Assume success if tokens were used
                     span.set_attribute("otel.status_code", "OK")
                elif result is not None: # If result is not None, assume some processing happened
                     span.set_attribute("otel.status_code", "OK")
                else: # If result is None, might be UNSET or OK depending on desired convention
                     span.set_attribute("otel.status_code", "OK") # Default to OK if no explicit error

                return result

            except Exception as e:
                span.set_attribute("otel.status_code", "ERROR")
                span.set_attribute("exception.type", type(e).__name__)
                span.set_attribute("exception.message", str(e))
                span.record_exception(e)
                raise

        setattr(agent, method_name, wrapped_method)