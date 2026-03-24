# KISS Framework API Reference

> **Auto-generated** — run `uv run generate-api-docs` to regenerate.

<details><summary><b>Table of Contents</b></summary>

- [`kiss`](#kiss)
  - [`kiss.core`](#kisscore)
    - [`kiss.core.kiss_agent`](#kisscorekiss_agent)
    - [`kiss.core.config`](#kisscoreconfig)
    - [`kiss.core.config_builder`](#kisscoreconfig_builder)
    - [`kiss.core.models`](#kisscoremodels)
      - [`kiss.core.models.model_info`](#kisscoremodelsmodel_info)
      - [`kiss.core.models.openai_compatible_model`](#kisscoremodelsopenai_compatible_model)
      - [`kiss.core.models.anthropic_model`](#kisscoremodelsanthropic_model)
      - [`kiss.core.models.gemini_model`](#kisscoremodelsgemini_model)
    - [`kiss.core.printer`](#kisscoreprinter)
    - [`kiss.core.print_to_console`](#kisscoreprint_to_console)
      - [`kiss.agents.sorcar.browser_ui`](#kissagentssorcarbrowser_ui)
      - [`kiss.agents.sorcar.useful_tools`](#kissagentssorcaruseful_tools)
      - [`kiss.agents.sorcar.web_use_tool`](#kissagentssorcarweb_use_tool)
    - [`kiss.core.utils`](#kisscoreutils)
  - [`kiss.agents`](#kissagents)
    - [`kiss.agents.coding_agents`](#kissagentscoding_agents)
      - [`kiss.agents.coding_agents.config`](#kissagentscoding_agentsconfig)
    - [`kiss.agents.sorcar`](#kissagentssorcar)
    - [`kiss.core.relentless_agent`](#kisscorerelentless_agent)
      - [`kiss.agents.sorcar.sorcar_agent`](#kissagentssorcarsorcar_agent)
      - [`kiss.agents.sorcar.config`](#kissagentssorcarconfig)
    - [`kiss.agents.gepa`](#kissagentsgepa)
      - [`kiss.agents.gepa.config`](#kissagentsgepaconfig)
    - [`kiss.agents.kiss_evolve`](#kissagentskiss_evolve)
      - [`kiss.agents.kiss_evolve.config`](#kissagentskiss_evolveconfig)
  - [`kiss.docker`](#kissdocker)
    - [`kiss.agents.sorcar.shared_utils`](#kissagentssorcarshared_utils)
    - [`kiss.agents.sorcar.stateful_sorcar_agent`](#kissagentssorcarstateful_sorcar_agent)
    - [`kiss.agents.vscode`](#kissagentsvscode)
      - [`kiss.agents.vscode.server`](#kissagentsvscodeserver)
  - [`kiss.channels`](#kisschannels)
    - [`kiss.channels.gmail_agent`](#kisschannelsgmail_agent)
    - [`kiss.channels.slack_agent`](#kisschannelsslack_agent)
    - [`kiss.channels.whatsapp_agent`](#kisschannelswhatsapp_agent)
      - [`kiss.core.models.novita_model`](#kisscoremodelsnovita_model)

</details>

______________________________________________________________________

## `kiss` — *Top-level Kiss module for the project.*

```python
from kiss import __version__
```

______________________________________________________________________

### `kiss.core` — *Core module for the KISS agent framework.*

```python
from kiss.core import AgentConfig, AnthropicModel, Config, DEFAULT_CONFIG, GeminiModel, KISSError, Model, OpenAICompatibleModel
```

#### `class AgentConfig(BaseModel)`

#### `class Config(BaseModel)`

#### `class KISSError(ValueError)` — Custom exception class for KISS framework errors.

______________________________________________________________________

#### `kiss.core.kiss_agent` — *Core KISS agent implementation with native function calling support.*

##### `class KISSAgent(Base)` — A KISS agent using native function calling.

**Constructor:** `KISSAgent(name: str) -> None`

- **run** — Runs the agent's main ReAct loop to solve the task.<br/>`run(model_name: str, prompt_template: str, arguments: dict[str, str] | None = None, system_prompt: str = '', tools: list[Callable[..., Any]] | None = None, is_agentic: bool = True, max_steps: int | None = None, max_budget: float | None = None, model_config: dict[str, Any] | None = None, printer: Printer | None = None, verbose: bool | None = None, attachments: list[Attachment] | None = None, session_info: str = '') -> str`

  - `model_name`: The name of the model to use for the agent.
  - `prompt_template`: The prompt template for the agent.
  - `arguments`: The arguments to be substituted into the prompt template. Default is None.
  - `system_prompt`: Optional system prompt to provide to the model. Default is empty string (no system prompt).
  - `tools`: The tools to use for the agent. If None, no tools are provided (only the built-in finish tool is added).
  - `is_agentic`: Whether the agent is agentic. Default is True.
  - `max_steps`: The maximum number of steps to take. Default is DEFAULT_CONFIG.agent.max_steps.
  - `max_budget`: The maximum budget to spend. Default is DEFAULT_CONFIG.agent.max_agent_budget.
  - `model_config`: The model configuration to use for the agent. Default is None.
  - `printer`: Optional printer for streaming output. Default is None.
  - `verbose`: Whether to print output to console. Default is None (uses config verbose setting).
  - `attachments`: Optional file attachments (images, PDFs) to include in the initial prompt. Default is None.
  - `session_info`: Sub-session label string (e.g. "Session: 1/5") to include in usage info output. Default is empty string.
  - **Returns:** str: The result of the agent's task.

- **finish** — The agent must call this function with the final answer to the task.<br/>`finish(result: str) -> str`

  - `result`: The result generated by the agent.
  - **Returns:** Returns the result of the agent's task.

______________________________________________________________________

#### `kiss.core.config` — *Configuration Pydantic models for KISS agent settings with CLI support.*

##### `class APIKeysConfig(BaseModel)`

##### `class RelentlessAgentConfig(BaseModel)`

##### `class DockerConfig(BaseModel)`

______________________________________________________________________

#### `kiss.core.config_builder` — *Configuration builder for KISS agent settings with CLI support.*

**`add_config`** — Build the KISS config, optionally overriding with command-line arguments. This function accumulates configs - each call adds a new config field while preserving existing fields from previous calls.<br/>`def add_config(name: str, config_class: type[BaseModel]) -> None`

- `name`: Name of the config class.
- `config_class`: Class of the config.

______________________________________________________________________

#### `kiss.core.models` — *Model implementations for different LLM providers.*

```python
from kiss.core.models import Attachment, Model, AnthropicModel, OpenAICompatibleModel, GeminiModel, NovitaModel
```

##### `class Attachment` — A file attachment (image or document) to include in a prompt.

- **from_file** — Create an Attachment from a file path.<br/>`from_file(path: str) -> 'Attachment'`

  - `path`: Path to the file to attach.
  - **Returns:** An Attachment with the file's bytes and detected MIME type.

- **to_base64** — Return the file data as a base64-encoded string.<br/>`to_base64() -> str`

- **to_data_url** — Return a data: URL suitable for OpenAI image_url fields.<br/>`to_data_url() -> str`

##### `class Model(ABC)` — Abstract base class for LLM provider implementations.

**Constructor:** `Model(model_name: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)`

- `model_name`: The name/identifier of the model.

- `model_config`: Optional dictionary of model configuration parameters.

- `token_callback`: Optional async callback invoked with each streamed text token.

- **close_callback_loop** — Close the per-instance event loop used for synchronous token callback invocation. Safe to call multiple times; subsequent calls are no-ops.<br/>`close_callback_loop() -> None`

- **initialize** — Initializes the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt to start the conversation.
  - `attachments`: Optional list of file attachments (images, PDFs) to include.

- **generate** — Generates content from prompt.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** tuple\[str, Any\]: A tuple of (generated_text, raw_response).

- **generate_and_process_with_tools** — Generates content with tools, processes the response, and adds it to conversation.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]]) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - **Returns:** tuple\[list\[dict[str, Any]\], str, Any\]: A tuple of (function_calls, response_text, raw_response).

- **add_function_results_to_conversation_and_return** — Adds function results to the conversation state. Matches results to tool calls by index from the last assistant message.<br/>`add_function_results_to_conversation_and_return(function_results: list[tuple[str, dict[str, Any]]]) -> None`

  - `function_results`: List of tuples containing (function_name, result_dict).

- **add_message_to_conversation** — Adds a message to the conversation state.<br/>`add_message_to_conversation(role: str, content: str) -> None`

  - `role`: The role of the message sender (e.g., 'user', 'assistant').
  - `content`: The message content.

- **extract_input_output_token_counts_from_response** — Extracts token counts from an API response.<br/>`extract_input_output_token_counts_from_response(response: Any) -> tuple[int, int, int, int]`

  - `response`: The raw API response object.
  - **Returns:** tuple\[int, int, int, int\]: (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).

- **get_embedding** — Generates an embedding vector for the given text.<br/>`get_embedding(text: str, embedding_model: str | None = None) -> list[float]`

  - `text`: The text to generate an embedding for.
  - `embedding_model`: Optional model name to use for embedding generation.
  - **Returns:** list\[float\]: The embedding vector as a list of floats.

- **set_usage_info_for_messages** — Sets token information to append to messages sent to the LLM.<br/>`set_usage_info_for_messages(usage_info: str) -> None`

  - `usage_info`: The usage information string to append.

______________________________________________________________________

#### `kiss.core.models.model_info` — *Model information: pricing and context lengths for supported LLM providers.*

##### `class ModelInfo` — Container for model metadata including pricing and capabilities.

**Constructor:** `ModelInfo(context_length: int, input_price_per_million: float, output_price_per_million: float, is_function_calling_supported: bool, is_embedding_supported: bool, is_generation_supported: bool, cache_read_price_per_million: float | None = None, cache_write_price_per_million: float | None = None)`

**`is_model_flaky`** — Check if a model is known to be flaky.<br/>`def is_model_flaky(model_name: str) -> bool`

- `model_name`: The name of the model to check.
- **Returns:** bool: True if the model is known to have reliability issues.

**`get_flaky_reason`** — Get the reason why a model is flaky.<br/>`def get_flaky_reason(model_name: str) -> str`

- `model_name`: The name of the model to check.
- **Returns:** str: The reason for flakiness, or empty string if not flaky.

**`model`** — Get a model instance based on model name prefix.<br/>`def model(model_name: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None) -> Model`

- `model_name`: The name of the model (with provider prefix if applicable).
- `model_config`: Optional dictionary of model configuration parameters. If it contains "base_url", routing is bypassed and an OpenAICompatibleModel is built with that base_url and optional "api_key".
- `token_callback`: Optional async callback invoked with each streamed text token.
- **Returns:** Model: An appropriate Model instance for the specified model.

**`get_available_models`** — Return model names for which an API key is configured and generation is supported.<br/>`def get_available_models() -> list[str]`

- **Returns:** list\[str\]: Sorted list of model name strings that have a configured API key and support text generation.

**`get_most_expensive_model`**<br/>`def get_most_expensive_model(fc_only: bool = True) -> str`

**`calculate_cost`** — Calculates the cost in USD for the given token counts.<br/>`def calculate_cost(model_name: str, num_input_tokens: int, num_output_tokens: int, num_cache_read_tokens: int = 0, num_cache_write_tokens: int = 0) -> float`

- `model_name`: Name of the model (with or without provider prefix).
- `num_input_tokens`: Number of non-cached input tokens.
- `num_output_tokens`: Number of output tokens.
- `num_cache_read_tokens`: Number of tokens read from cache.
- `num_cache_write_tokens`: Number of tokens written to cache.
- **Returns:** float: Cost in USD, or 0.0 if pricing is not available for the model.

**`get_max_context_length`** — Returns the maximum context length supported by the model.<br/>`def get_max_context_length(model_name: str) -> int`

- `model_name`: Name of the model (with or without provider prefix).
- **Returns:** int: Maximum context length in tokens.

______________________________________________________________________

#### `kiss.core.models.openai_compatible_model` — *OpenAI-compatible model implementation for custom endpoints.*

##### `class OpenAICompatibleModel(Model)` — A model that uses an OpenAI-compatible API with a custom base URL.

**Constructor:** `OpenAICompatibleModel(model_name: str, base_url: str, api_key: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)`

- `model_name`: The name/identifier of the model to use.

- `base_url`: The base URL for the API endpoint (e.g., "http://localhost:11434/v1").

- `api_key`: API key for authentication.

- `model_config`: Optional dictionary of model configuration parameters.

- `token_callback`: Optional async callback invoked with each streamed text token.

- **initialize** — Initialize the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt to start the conversation.
  - `attachments`: Optional list of file attachments (images, PDFs) to include.

- **generate** — Generate content from prompt without tools.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** A tuple of (content, response) where content is the generated text and response is the raw API response object.

- **generate_and_process_with_tools** — Generate content with tools, process the response, and add it to conversation.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]]) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - **Returns:** A tuple of (function_calls, content, response) where function_calls is a list of dictionaries containing tool call information, content is the text response, and response is the raw API response object.

- **extract_input_output_token_counts_from_response** — Extract token counts from an API response.<br/>`extract_input_output_token_counts_from_response(response: Any) -> tuple[int, int, int, int]`

  - **Returns:** (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens). For OpenAI, cached_tokens is a subset of prompt_tokens; input_tokens is reported as (prompt_tokens - cached_tokens) so costs apply correctly. OpenRouter returns cache_write_tokens in prompt_tokens_details. OpenAI reasoning models may report reasoning tokens in completion_tokens_details.reasoning_tokens; those are counted as output tokens so Sorcar shows thinking-token usage.

- **get_embedding** — Generate an embedding vector for the given text.<br/>`get_embedding(text: str, embedding_model: str | None = None) -> list[float]`

  - `text`: The text to generate an embedding for.
  - `embedding_model`: Optional model name for embedding generation. Uses the model's name if not specified.
  - **Returns:** A list of floating point numbers representing the embedding vector.

______________________________________________________________________

#### `kiss.core.models.anthropic_model` — *Anthropic model implementation for Claude models.*

##### `class AnthropicModel(Model)` — A model that uses Anthropic's Messages API (Claude).

**Constructor:** `AnthropicModel(model_name: str, api_key: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)`

- `model_name`: The name of the Claude model to use.

- `api_key`: The Anthropic API key for authentication.

- `model_config`: Optional dictionary of model configuration parameters.

- `token_callback`: Optional async callback invoked with each streamed text token.

- **initialize** — Initializes the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt to start the conversation.
  - `attachments`: Optional list of file attachments (images, PDFs) to include.

- **generate** — Generates content from the current conversation.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** tuple\[str, Any\]: A tuple of (generated_text, raw_response).

- **generate_and_process_with_tools** — Generates content with tools and processes the response.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]]) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - **Returns:** tuple\[list\[dict[str, Any]\], str, Any\]: A tuple of (function_calls, response_text, raw_response).

- **add_function_results_to_conversation_and_return** — Add tool results to the conversation.<br/>`add_function_results_to_conversation_and_return(function_results: list[tuple[str, dict[str, Any]]]) -> None`

  - `function_results`: List of (func_name, result_dict) tuples. result_dict can contain: - "result": The result content string - "tool_use_id": Optional explicit tool_use_id to use

- **extract_input_output_token_counts_from_response** — Extracts token counts from an Anthropic API response.<br/>`extract_input_output_token_counts_from_response(response: Any) -> tuple[int, int, int, int]`

  - **Returns:** (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).

- **get_embedding** — Generates an embedding vector for the given text.<br/>`get_embedding(text: str, embedding_model: str | None = None) -> list[float]`

  - `text`: The text to generate an embedding for.
  - `embedding_model`: Optional model name (not used by Anthropic).

______________________________________________________________________

#### `kiss.core.models.gemini_model` — *Gemini model implementation for Google's GenAI models.*

##### `class GeminiModel(Model)` — A model that uses Google's GenAI API (Gemini).

**Constructor:** `GeminiModel(model_name: str, api_key: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)`

- `model_name`: The name of the Gemini model to use.

- `api_key`: The Google API key for authentication.

- `model_config`: Optional dictionary of model configuration parameters.

- `token_callback`: Optional async callback invoked with each streamed text token.

- **initialize** — Initializes the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt to start the conversation.
  - `attachments`: Optional list of file attachments (images, PDFs) to include.

- **generate** — Generates content from prompt without tools.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** tuple\[str, Any\]: A tuple of (generated_text, raw_response).

- **generate_and_process_with_tools** — Generates content with tools, processes the response, and adds it to conversation.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]]) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - **Returns:** tuple\[list\[dict[str, Any]\], str, Any\]: A tuple of (function_calls, response_text, raw_response).

- **extract_input_output_token_counts_from_response** — Extracts token counts from a Gemini API response.<br/>`extract_input_output_token_counts_from_response(response: Any) -> tuple[int, int, int, int]`

  - **Returns:** (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).

- **get_embedding** — Generates an embedding vector for the given text.<br/>`get_embedding(text: str, embedding_model: str | None = None) -> list[float]`

  - `text`: The text to generate an embedding for.
  - `embedding_model`: Optional model name. Defaults to "text-embedding-004".
  - **Returns:** list\[float\]: The embedding vector as a list of floats.

______________________________________________________________________

#### `kiss.core.printer` — *Abstract base class and shared utilities for KISS agent printers.*

##### `class StreamEventParser` — Shared parser for LLM stream events used by both console and browser printers.

**Constructor:** `StreamEventParser() -> None`

- **reset_stream_state** — Reset block type and tool buffer state.<br/>`reset_stream_state() -> None`
- **parse_stream_event** — Parse a stream event, dispatch to on\_\* callbacks, return extracted text.<br/>`parse_stream_event(event: Any) -> str`
  - `event`: An event object with an `event` dict attribute.
  - **Returns:** str: Any text content extracted from text or thinking deltas.

##### `class Printer(ABC)`

- **print** — Render content to the output destination.<br/>`print(content: Any, type: str = 'text', **kwargs: Any) -> str`

  - `content`: The content to display.
  - `type`: Content type (e.g. "text", "prompt", "stream_event", "tool_call", "tool_result", "result", "usage_info", "message").
  - `**kwargs`: Additional type-specific options (e.g. tool_input, is_error).
  - **Returns:** str: Any extracted text (e.g. streamed text deltas), or empty string.

- **token_callback** — Handle a single streamed token from the LLM.<br/>`async token_callback(token: str) -> None`

  - `token`: The text token to process.

- **reset** — Reset the printer's internal streaming state between messages.<br/>`reset() -> None`

##### `class MultiPrinter(Printer)`

**Constructor:** `MultiPrinter(printers: list[Printer]) -> None`

- **print** — Dispatch a print call to all child printers.<br/>`print(content: Any, type: str = 'text', **kwargs: Any) -> str`

  - `content`: The content to display.
  - `type`: Content type forwarded to each child printer.
  - `**kwargs`: Additional options forwarded to each child printer.
  - **Returns:** str: The result from the last child printer.

- **token_callback** — Forward a streamed token to all child printers.<br/>`async token_callback(token: str) -> None`

  - `token`: The text token to forward.

- **reset** — Reset streaming state on all child printers.<br/>`reset() -> None`
  **`parse_result_yaml`** — Parse a YAML result string and return the dict if it has a 'summary' key. Used by both console and browser printers to extract structured result data from agent finish() output.<br/>`def parse_result_yaml(raw: str) -> dict[str, Any] | None`

- `raw`: Raw result string, potentially YAML-formatted.

- **Returns:** The parsed dict if valid YAML with a 'summary' key, else None.

**`lang_for_path`** — Map a file path to its syntax-highlighting language name.<br/>`def lang_for_path(path: str) -> str`

- `path`: File path whose extension determines the language.
- **Returns:** str: Language name (e.g. "python", "javascript"), or the raw extension, or "text" if no extension is present.

**`truncate_result`** — Truncate long content to MAX_RESULT_LEN, keeping the first and last halves.<br/>`def truncate_result(content: str) -> str`

- `content`: The string to truncate.
- **Returns:** str: The original string if short enough, otherwise the first and last halves joined by a truncation marker.

**`extract_path_and_lang`** — Extract the file path and inferred language from a tool input dict.<br/>`def extract_path_and_lang(tool_input: dict) -> tuple[str, str]`

- `tool_input`: Dictionary of tool call arguments, checked for "file_path" or "path" keys.
- **Returns:** tuple\[str, str\]: A (file_path, language) pair. Language defaults to "text" if no path is found.

**`extract_extras`** — Extract non-standard keys from a tool input dict for display.<br/>`def extract_extras(tool_input: dict) -> dict[str, str]`

- `tool_input`: Dictionary of tool call arguments.
- **Returns:** dict\[str, str\]: Keys not in KNOWN_KEYS mapped to their string values (truncated to 200 chars).

______________________________________________________________________

#### `kiss.core.print_to_console` — *Console output formatting for KISS agents.*

##### `class ConsolePrinter(StreamEventParser, Printer)`

**Constructor:** `ConsolePrinter(file: Any = None) -> None`

- **reset** — Reset internal streaming and tool-parsing state for a new turn.<br/>`reset() -> None`

- **print** — Render content to the console using Rich formatting.<br/>`print(content: Any, type: str = 'text', **kwargs: Any) -> str`

  - `content`: The content to display.
  - `type`: Content type (e.g. "text", "prompt", "stream_event", "tool_call", "tool_result", "result", "usage_info", "message").
  - `**kwargs`: Additional options such as tool_input, is_error, cost, total_tokens.
  - **Returns:** str: Extracted text from stream events, or empty string.

- **token_callback** — Stream a single token to the console, styled by current block type.<br/>`async token_callback(token: str) -> None`

  - `token`: The text token to display.

______________________________________________________________________

#### `kiss.agents.sorcar.browser_ui` — *Shared browser UI components for KISS agent viewers.*

##### `class BaseBrowserPrinter(StreamEventParser, Printer)`

**Constructor:** `BaseBrowserPrinter() -> None`

- **reset** — Reset internal streaming and tool-parsing state for a new turn.<br/>`reset() -> None`

- **start_recording** — Start recording broadcast events for the calling thread. Each thread gets its own independent recording buffer, so concurrent agent threads do not interfere with each other's recordings.<br/>`start_recording() -> None`

- **stop_recording** — Stop recording for the calling thread and return its display events.<br/>`stop_recording() -> list[dict[str, Any]]`

  - **Returns:** List of display-relevant events with consecutive deltas merged.

- **broadcast** — Send an SSE event dict to the connected client. The event is also appended to every active per-thread recording.<br/>`broadcast(event: dict[str, Any]) -> None`

  - `event`: The event dictionary to broadcast.

- **add_client** — Register the SSE client and return its event queue. Only one client is supported. A new connection replaces any previous one.<br/>`add_client() -> queue.Queue[dict[str, Any]]`

  - **Returns:** queue.Queue\[dict[str, Any]\]: A queue that will receive broadcast events.

- **remove_client** — Unregister the SSE client's event queue. Only clears the queue if *cq* is the current client (handles reconnection races where the old connection tears down after a new one has already connected).<br/>`remove_client(cq: queue.Queue[dict[str, Any]]) -> None`

  - `cq`: The client queue to remove.

- **has_clients** — Return True if a client is currently connected.<br/>`has_clients() -> bool`

- **print** — Render content by broadcasting SSE events to connected browser clients.<br/>`print(content: Any, type: str = 'text', **kwargs: Any) -> str`

  - `content`: The content to display.
  - `type`: Content type (e.g. "text", "prompt", "stream_event", "tool_call", "tool_result", "result", "usage_info", "message").
  - `**kwargs`: Additional options such as tool_input, is_error, cost, total_tokens.
  - **Returns:** str: Extracted text from stream events, or empty string.

- **token_callback** — Broadcast a streamed token as an SSE delta event to browser clients.<br/>`async token_callback(token: str) -> None`

  - `token`: The text token to broadcast.

______________________________________________________________________

#### `kiss.agents.sorcar.useful_tools` — *Useful tools for agents: file editing and bash execution.*

##### `class UsefulTools` — A hardened collection of useful tools with improved security.

**Constructor:** `UsefulTools(stream_callback: Callable[[str], None] | None = None, stop_event: threading.Event | None = None) -> None`

- **Read** — Read file contents.<br/>`Read(file_path: str, max_lines: int = 2000) -> str`

  - `file_path`: Absolute path to file.
  - `max_lines`: Maximum number of lines to return.

- **Write** — Write content to a file, creating it if it doesn't exist or overwriting if it does.<br/>`Write(file_path: str, content: str) -> str`

  - `file_path`: Path to the file to write.
  - `content`: The full content to write to the file.

- **Edit** — Performs precise string replacements in files with exact matching.<br/>`Edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str`

  - `file_path`: Absolute path to the file to modify.
  - `old_string`: Exact text to find and replace.
  - `new_string`: Replacement text, must differ from old_string.
  - `replace_all`: If True, replace all occurrences.
  - **Returns:** The output of the edit operation.

- **Bash** — Runs a bash command and returns its output.<br/>`Bash(command: str, description: str, timeout_seconds: float = 30, max_output_chars: int = 50000) -> str`

  - `command`: The bash command to run.
  - `description`: A brief description of the command.
  - `timeout_seconds`: Timeout in seconds for the command.
  - `max_output_chars`: Maximum characters in output before truncation.
  - **Returns:** The output of the command.

______________________________________________________________________

#### `kiss.agents.sorcar.web_use_tool` — *Browser automation tool for LLM agents using Playwright.*

##### `class WebUseTool` — Browser automation tool using Playwright + default OS browser.

**Constructor:** `WebUseTool(viewport: tuple[int, int] = (1280, 900), user_data_dir: str | None = None, wait_for_user_callback: Callable[[str, str], None] | None = None, **_kwargs: Any) -> None`

- **go_to_url** — Navigate the browser to a URL and return the page accessibility tree. Use when you need to open a new page or switch pages. Special values: "tab:list" returns a list of open tabs; "tab:N" switches to tab N (0-based).<br/>`go_to_url(url: str) -> str`

  - `url`: Full URL to open, or "tab:list" for tab list, or "tab:N" to switch to tab N.
  - **Returns:** On success: page title, URL, and accessibility tree with [N] IDs. For "tab:list": list of open tabs with indices. On error: "Error navigating to <url>: <message>".

- **click** — Click or hover on an interactive element by its [N] ID from the accessibility tree. Use after get_page_content or go_to_url to interact with links, buttons, tabs, etc.<br/>`click(element_id: int, action: str = 'click') -> str`

  - `element_id`: Numeric ID shown in brackets [N] next to the element in the tree.
  - `action`: "click" (default) to click the element, "hover" to only move focus.
  - **Returns:** Updated accessibility tree (title, URL, numbered elements), or on error "Error clicking element <id>: <message>".

- **type_text** — Type text into a textbox, searchbox, or other editable element by its [N] ID. Clears existing content then types the given text. Use for forms, search boxes, etc.<br/>`type_text(element_id: int, text: str, press_enter: bool = False) -> str`

  - `element_id`: Numeric ID from the accessibility tree (brackets [N]).
  - `text`: String to type into the element.
  - `press_enter`: If True, press Enter after typing (e.g. to submit a search).
  - **Returns:** Updated accessibility tree, or "Error typing into element <id>: <message>" on error.

- **press_key** — Press a single key or key combination. Use for navigation, closing dialogs, shortcuts.<br/>`press_key(key: str) -> str`

  - `key`: Key name, e.g. "Enter", "Escape", "Tab", "ArrowDown", "PageDown", "Backspace", or combination like "Control+a", "Shift+Tab".
  - **Returns:** Updated accessibility tree, or "Error pressing key '<key>': <message>" on error.

- **scroll** — Scroll the current page to reveal more content. Use when needed elements are off-screen.<br/>`scroll(direction: str = 'down', amount: int = 3) -> str`

  - `direction`: "down", "up", "left", or "right".
  - `amount`: Number of scroll steps (default 3).
  - **Returns:** Updated accessibility tree after scrolling, or "Error scrolling <direction>: <message>" on error.

- **screenshot** — Capture the visible viewport of the Chromium browser as an image. Use to verify layout, captchas, or visual state of a web page currently open in the browser. This does NOT capture or display local files, attached images, or PDFs — it only screenshots the browser window.<br/>`screenshot(file_path: str = 'screenshot.png') -> str`

  - `file_path`: Path where the PNG will be saved (default "screenshot.png"). Parent directories are created if needed.
  - **Returns:** "Screenshot saved to \<resolved_path>", or "Error taking screenshot: <message>" on error.

- **get_page_content** — Get the current page content. Use to decide what to click or type next.<br/>`get_page_content(text_only: bool = False) -> str`

  - `text_only`: If False (default), return accessibility tree with [N] IDs for interactive elements. If True, return plain text only (title, URL, body text).
  - **Returns:** Accessibility tree or plain text as described above, or "Error getting page content: <message>" on error.

- **close** — Close the browser and release resources. Call when done with the session or before exit.<br/>`close() -> str`

  - **Returns:** "Browser closed." (always, even if nothing was open).

- **ask_user_browser_action** — Open URL in user's default browser for interaction, wait for completion. Use when the agent needs the human to interact with a webpage directly — CAPTCHAs, 2FA/MFA, OAuth flows, cookie consent, or any complex interaction the agent cannot automate. Opens the URL in the user's default OS browser (Chrome, Safari, Firefox, etc.) rather than a Playwright-controlled window.<br/>`ask_user_browser_action(instruction: str, url: str = '') -> str`

  - `instruction`: What the user should do (e.g. "Please solve the CAPTCHA").
  - `url`: Optional URL to navigate to before handing control to the user.
  - **Returns:** Updated accessibility tree after the user signals they are done.

- **get_tools** — Return callable web tools for registration with an agent.<br/>`get_tools() -> list[Callable[..., str]]`

  - **Returns:** List of callables: go_to_url, click, type_text, press_key, scroll, screenshot, get_page_content, ask_user_browser_action. Does not include close.

______________________________________________________________________

#### `kiss.core.utils` — *Utility functions for the KISS core module.*

**`get_config_value`** — Get a config value, preferring explicit value over config default. This eliminates the repetitive pattern: value if value is not None else config.attr_name<br/>`def get_config_value(value: T | None, config_obj: Any, attr_name: str, default: T | None = None) -> T`

- `value`: The explicitly provided value (may be None)
- `config_obj`: The config object to read from if value is None
- `attr_name`: The attribute name to read from config_obj
- `default`: Fallback default if both value and config attribute are None
- **Returns:** The resolved value (explicit value > config value > default)

**`get_template_field_names`** — Get the field names from the text.<br/>`def get_template_field_names(text: str) -> list[str]`

- `text`: The text containing template field placeholders.
- **Returns:** list\[str\]: A list of field names found in the text.

**`escape_invalid_template_field_names`** — Escape invalid field names from the text.<br/>`def escape_invalid_template_field_names(text: str, valid_field_names: set[str]) -> str`

- `text`: The text containing template field placeholders.
- `valid_field_names`: A list of valid field names.
- **Returns:** An escaped string with invalid field placeholders escaped

**`add_prefix_to_each_line`** — Adds a prefix to each line of the text.<br/>`def add_prefix_to_each_line(text: str, prefix: str) -> str`

- `text`: The text to add prefix to.
- `prefix`: The prefix to add to each line.
- **Returns:** str: The text with prefix added to each line.

**`config_to_dict`** — Convert the config to a dictionary.<br/>`def config_to_dict() -> dict[Any, Any]`

- **Returns:** dict\[Any, Any\]: A dictionary representation of the default config.

**`fc`** — Reads a file and returns the content.<br/>`def fc(file_path: str) -> str`

- `file_path`: The path to the file to read.
- **Returns:** str: The content of the file.

**`finish`** — The agent must call this function with the final status, analysis, and result when it has solved the given task. Status **MUST** be 'success' or 'failure'.<br/>`def finish(status: str = 'success', analysis: str = '', result: str = '') -> str`

- `status`: The status of the agent's task ('success' or 'failure'). Defaults to 'success'.
- `analysis`: The analysis of the agent's trajectory.
- `result`: The result generated by the agent.
- **Returns:** A YAML string containing the status, analysis, and result of the agent's task.

**`read_project_file`** — Read a file from the project root. Compatible with installations packaged as .whl (zip) or source.<br/>`def read_project_file(file_path_relative_to_project_root: str) -> str`

- `file_path_relative_to_project_root`: Path relative to the project root.
- **Returns:** str: The file's contents.

**`read_project_file_from_package`** — Read a file from the project root.<br/>`def read_project_file_from_package(file_name_as_python_package: str) -> str`

- `file_name_as_python_package`: File name as a Python package.
- **Returns:** str: The file's contents.

**`resolve_path`** — Resolve a path relative to base_dir if not absolute.<br/>`def resolve_path(p: str, base_dir: str) -> Path`

- `p`: The path string to resolve.
- `base_dir`: The base directory for relative path resolution.
- **Returns:** Path: The resolved absolute path.

**`is_subpath`** — Check if target has any prefix in whitelist.<br/>`def is_subpath(target: Path, whitelist: list[Path]) -> bool`

- `target`: The path to check.
- `whitelist`: List of allowed path prefixes.
- **Returns:** bool: True if target is under any path in whitelist, False otherwise.

______________________________________________________________________

### `kiss.agents` — *KISS agents package with pre-built agent implementations.*

```python
from kiss.agents import prompt_refiner_agent, get_run_simple_coding_agent, run_bash_task_in_sandboxed_ubuntu_latest
```

**`prompt_refiner_agent`** — Refines the prompt template based on the agent's trajectory summary.<br/>`def prompt_refiner_agent(original_prompt_template: str, previous_prompt_template: str, agent_trajectory_summary: str, model_name: str) -> str`

- `original_prompt_template`: The original prompt template.
- `previous_prompt_template`: The previous version of the prompt template that led to the given trajectory.
- `agent_trajectory_summary`: The agent's trajectory summary as a string.
- `model_name`: The name of the model to use for the agent.
- **Returns:** str: The refined prompt template.

**`get_run_simple_coding_agent`** — Return a function that runs a simple coding agent with a test function.<br/>`def get_run_simple_coding_agent(test_fn: Callable[[str], bool]) -> Callable[..., str]`

- `test_fn`: The test function to use for the agent.
- **Returns:** Callable\[..., str\]: A function that runs a simple coding agent with a test function. Accepts keyword arguments: model_name (str), prompt_template (str), and arguments (dict[str, str]).

**`run_bash_task_in_sandboxed_ubuntu_latest`** — Run a bash task in a sandboxed Ubuntu latest container.<br/>`def run_bash_task_in_sandboxed_ubuntu_latest(task: str, model_name: str) -> str`

- `task`: The task to run.
- `model_name`: The name of the model to use for the agent.
- **Returns:** str: The result of the task.

______________________________________________________________________

#### `kiss.agents.coding_agents` — *Coding agents for KISS framework.*

```python
from kiss.agents.coding_agents import Base, SYSTEM_PROMPT
```

##### `class Base` — Base class for all KISS agents with common state management and persistence.

**Constructor:** `Base(name: str) -> None`

- `name`: The name identifier for the agent.

- **set_printer** — Configure the output printer for this agent. If an explicit *printer* is provided, it is always used regardless of the verbose setting. Otherwise a `ConsolePrinter` is created when verbose output is enabled (either explicitly or via config).<br/>`set_printer(printer: Printer | None = None, verbose: bool | None = None) -> None`

  - `printer`: An existing Printer instance to use directly. If provided, verbose is ignored.
  - `verbose`: Whether to print to the console. If None, uses the verbose config value.

- **get_trajectory** — Return the trajectory as JSON for visualization.<br/>`get_trajectory() -> str`

  - **Returns:** str: A JSON-formatted string of all messages in the agent's history.

______________________________________________________________________

#### `kiss.agents.coding_agents.config` — *Configuration Pydantic models for coding agent settings.*

##### `class RelentlessCodingAgentConfig(BaseModel)`

##### `class CodingAgentConfig(BaseModel)`

______________________________________________________________________

#### `kiss.agents.sorcar` — *Sorcar agent with coding tools and browser automation.*

______________________________________________________________________

#### `kiss.core.relentless_agent` — *Base relentless agent with smart continuation for long tasks.*

##### `class RelentlessAgent(Base)` — Base agent with auto-continuation for long tasks.

- **perform_task** — Execute the task with auto-continuation across multiple sub-sessions.<br/>`perform_task(tools: list[Callable[..., Any]], attachments: list[Attachment] | None = None) -> str`

  - `tools`: List of callable tools available to the agent during execution.
  - `attachments`: Optional file attachments (images, PDFs) for the initial prompt.
  - **Returns:** YAML string with 'success' and 'summary' keys on successful completion.

- **run** — Run the agent with the provided tools.<br/>`run(model_name: str | None = None, prompt_template: str = '', arguments: dict[str, str] | None = None, system_prompt: str = '', max_steps: int | None = None, max_budget: float | None = None, model_config: dict[str, Any] | None = None, work_dir: str | None = None, printer: Printer | None = None, max_sub_sessions: int | None = None, docker_image: str | None = None, verbose: bool | None = None, tools: list[Callable[..., Any]] | None = None, attachments: list[Attachment] | None = None) -> str`

  - `model_name`: LLM model to use. Defaults to config value.
  - `prompt_template`: Task prompt template with format placeholders.
  - `arguments`: Dictionary of values to fill prompt_template placeholders.
  - `system_prompt`: System-level instructions passed to the underlying LLM via model_config. Defaults to empty string (no system instructions).
  - `max_steps`: Maximum steps per sub-session. Defaults to config value.
  - `max_budget`: Maximum budget in USD. Defaults to config value.
  - `model_config`: Optional dictionary of additional model configuration parameters (e.g. temperature, top_p). Defaults to None.
  - `work_dir`: Working directory for the agent. Defaults to artifact_dir/kiss_workdir.
  - `printer`: Printer instance for output display.
  - `max_sub_sessions`: Maximum continuation sub-sessions. Defaults to config value.
  - `docker_image`: Docker image name to run tools inside a container.
  - `verbose`: Whether to print output to console. Defaults to config verbose setting.
  - `tools`: List of callable tools available to the agent during execution.
  - `attachments`: Optional file attachments (images, PDFs) for the initial prompt.
  - **Returns:** YAML string with 'success' and 'summary' keys.

**`finish`** — Finish execution with status and summary.<br/>`def finish(success: bool, is_continue: bool, summary: str) -> str`

- `success`: True if the agent has successfully completed the task, False otherwise
- `is_continue`: True if the task is incomplete and should continue, False otherwise
- `summary`: precise chronologically-ordered list of things the agent did with the reason for doing that along with relevant code snippets

______________________________________________________________________

#### `kiss.agents.sorcar.sorcar_agent` — *Sorcar agent with both coding tools and browser automation.*

##### `class SorcarAgent(RelentlessAgent)` — Agent with both coding tools and browser automation for web + code tasks.

**Constructor:** `SorcarAgent(name: str) -> None`

- **run** — Run the assistant agent with coding tools and browser automation.<br/>`run(model_name: str | None = None, prompt_template: str = '', arguments: dict[str, str] | None = None, system_prompt: str | None = None, tools: list[Callable[..., Any]] | None = None, max_steps: int | None = None, max_budget: float | None = None, model_config: dict[str, Any] | None = None, work_dir: str | None = None, printer: Printer | None = None, max_sub_sessions: int | None = None, docker_image: str | None = None, headless: bool | None = None, verbose: bool | None = None, current_editor_file: str | None = None, attachments: list[Attachment] | None = None, wait_for_user_callback: Callable[[str, str], None] | None = None, ask_user_question_callback: Callable[[str], str] | None = None) -> str`
  - `model_name`: LLM model to use. Defaults to config value.
  - `prompt_template`: Task prompt template with format placeholders.
  - `arguments`: Dictionary of values to fill prompt_template placeholders.
  - `system_prompt`: system prompt to be appended to the actual system prompt
  - `tools`: List of tools to be added in addition to bash and web tools.
  - `max_steps`: Maximum steps per sub-session. Defaults to config value.
  - `max_budget`: Maximum budget in USD. Defaults to config value.
  - `work_dir`: Working directory for the agent. Defaults to artifact_dir/kiss_workdir.
  - `printer`: Printer instance for output display.
  - `max_sub_sessions`: Maximum continuation sub-sessions. Defaults to config value.
  - `docker_image`: Docker image name to run tools inside a container.
  - `headless`: Deprecated, ignored. Browser always runs headless.
  - `verbose`: Whether to print output to console. Defaults to config verbose setting.
  - `current_editor_file`: Path to the currently active editor file, appended to prompt.
  - `attachments`: Optional file attachments (images, PDFs) for the initial prompt.
  - `wait_for_user_callback`: Optional callback used by browser tools when user action is required.
  - `ask_user_question_callback`: Optional callback used by the ask_user_question tool to collect a text response from the user.
  - **Returns:** YAML string with 'success' and 'summary' keys.

**`cli_wait_for_user`** — CLI callback for browser-action prompts (prints and waits for Enter).<br/>`def cli_wait_for_user(instruction: str, url: str) -> None`

- `instruction`: What the user should do.
- `url`: Current browser URL (printed if non-empty).

**`cli_ask_user_question`** — CLI callback for agent questions (prints and reads from stdin).<br/>`def cli_ask_user_question(question: str) -> str`

- `question`: The question to display to the user.
- **Returns:** The user's typed response text.

______________________________________________________________________

#### `kiss.agents.sorcar.config` — *Configuration for the Assistant Agent.*

##### `class AgentConfig(BaseModel)`

##### `class SorcarConfig(BaseModel)`

______________________________________________________________________

#### `kiss.agents.gepa` — *GEPA (Genetic-Pareto) prompt optimization package.*

```python
from kiss.agents.gepa import GEPA, GEPAPhase, GEPAProgress, PromptCandidate, create_progress_callback
```

##### `class GEPA` — GEPA (Genetic-Pareto) prompt optimizer.

**Constructor:** `GEPA(agent_wrapper: Callable[[str, dict[str, str]], tuple[str, list[Any]]], initial_prompt_template: str, evaluation_fn: Callable[[str], dict[str, float]] | None = None, max_generations: int | None = None, population_size: int | None = None, pareto_size: int | None = None, mutation_rate: float | None = None, reflection_model: str | None = None, dev_val_split: float | None = None, perfect_score: float = 1.0, use_merge: bool = True, max_merge_invocations: int = 5, merge_val_overlap_floor: int = 2, progress_callback: Callable[[GEPAProgress], None] | None = None, batched_agent_wrapper: Callable[[str, list[dict[str, str]]], list[tuple[str, list[Any]]]] | None = None)`

- `agent_wrapper`: Function (prompt_template, arguments) -> (result, trajectory). Used when batched_agent_wrapper is not provided, or as fallback.

- `initial_prompt_template`: The initial prompt template to optimize

- `evaluation_fn`: Function to evaluate result -> {metric: score}

- `max_generations`: Maximum evolutionary generations

- `population_size`: Number of candidates per generation

- `pareto_size`: Maximum Pareto frontier size

- `mutation_rate`: Probability of mutation (default: 0.5)

- `reflection_model`: Model for reflection

- `dev_val_split`: Fraction for dev set (default: 0.5)

- `perfect_score`: Score threshold to skip mutation (default: 1.0)

- `use_merge`: Whether to enable structural merge (default: True)

- `max_merge_invocations`: Maximum merge operations to attempt (default: 5)

- `merge_val_overlap_floor`: Minimum validation overlap for merge (default: 2)

- `progress_callback`: Optional callback function called with GEPAProgress during optimization. Use this to track progress, display progress bars, or log intermediate results.

- `batched_agent_wrapper`: Optional batched version of agent_wrapper. Function (prompt_template, [arguments]) -> [(result, trajectory)]. When provided, GEPA calls this with all examples in a minibatch at once instead of calling agent_wrapper one at a time. This enables prompt merging (combining multiple prompts into a single API call) for significantly higher throughput.

- **optimize** — Run GEPA optimization.<br/>`optimize(train_examples: list[dict[str, str]], dev_minibatch_size: int | None = None) -> PromptCandidate`

  - `train_examples`: Training examples (will be split into dev/val)
  - `dev_minibatch_size`: Dev examples per evaluation (default: all)
  - **Returns:** Best PromptCandidate found

- **get_pareto_frontier** — Get a copy of the current Pareto frontier.<br/>`get_pareto_frontier() -> list[PromptCandidate]`

  - **Returns:** A list of PromptCandidate instances representing the current Pareto frontier (best candidates per validation instance).

- **get_best_prompt** — Get the best prompt template found during optimization.<br/>`get_best_prompt() -> str`

  - **Returns:** The prompt template string from the best candidate.

##### `class GEPAPhase(Enum)` — Enum representing the current phase of GEPA optimization.

##### `class GEPAProgress` — Progress information for GEPA optimization callbacks.

##### `class PromptCandidate` — Represents a prompt candidate with its performance metrics.

**`create_progress_callback`** — Create a standard progress callback for GEPA optimization.<br/>`def create_progress_callback(verbose: bool = False) -> 'Callable[[GEPAProgress], None]'`

- `verbose`: If True, prints all phases. If False, only prints val evaluation completion messages (when a candidate has been fully evaluated).
- **Returns:** A callback function that prints progress updates during optimization.

______________________________________________________________________

#### `kiss.agents.gepa.config` — *GEPA-specific configuration that extends the main KISS config.*

##### `class GEPAConfig(BaseModel)` — GEPA-specific configuration settings.

______________________________________________________________________

#### `kiss.agents.kiss_evolve` — *KISSEvolve: Evolutionary Algorithm Discovery using LLMs.*

```python
from kiss.agents.kiss_evolve import CodeVariant, KISSEvolve, SimpleRAG
```

##### `class CodeVariant` — Represents a code variant in the evolutionary population.

##### `class KISSEvolve` — KISSEvolve: Evolutionary algorithm discovery using LLMs.

**Constructor:** `KISSEvolve(code_agent_wrapper: Callable[..., str], initial_code: str, evaluation_fn: Callable[[str], dict[str, Any]], model_names: list[tuple[str, float]], extra_coding_instructions: str = '', population_size: int | None = None, max_generations: int | None = None, mutation_rate: float | None = None, elite_size: int | None = None, num_islands: int | None = None, migration_frequency: int | None = None, migration_size: int | None = None, migration_topology: str | None = None, enable_novelty_rejection: bool | None = None, novelty_threshold: float | None = None, max_rejection_attempts: int | None = None, novelty_rag_model: Model | None = None, parent_sampling_method: str | None = None, power_law_alpha: float | None = None, performance_novelty_lambda: float | None = None)`

- `code_agent_wrapper`: The code generation agent wrapper. Should accept keyword arguments: model_name (str), prompt_template (str), and arguments (dict[str, str]).

- `initial_code`: The initial code to evolve.

- `evaluation_fn`: Function that takes code string and returns dict with: - 'fitness': float (higher is better) - 'metrics': dict[str, float] (optional additional metrics) - 'artifacts': dict[str, Any] (optional execution artifacts) - 'error': str (optional error message if evaluation failed)

- `model_names`: List of tuples containing (model_name, probability). Probabilities will be normalized to sum to 1.0.

- `extra_coding_instructions`: Extra instructions to add to the code generation prompt.

- `population_size`: Number of variants to maintain in population. If None, uses value from DEFAULT_CONFIG.kiss_evolve.population_size.

- `max_generations`: Maximum number of evolutionary generations. If None, uses value from DEFAULT_CONFIG.kiss_evolve.max_generations.

- `mutation_rate`: Probability of mutating a variant. If None, uses value from DEFAULT_CONFIG.kiss_evolve.mutation_rate.

- `elite_size`: Number of best variants to preserve each generation. If None, uses value from DEFAULT_CONFIG.kiss_evolve.elite_size.

- `num_islands`: Number of islands for island-based evolution. If None, uses value from DEFAULT_CONFIG.kiss_evolve.num_islands.

- `migration_frequency`: Number of generations between migrations. If None, uses value from DEFAULT_CONFIG.kiss_evolve.migration_frequency.

- `migration_size`: Number of individuals to migrate between islands. If None, uses value from DEFAULT_CONFIG.kiss_evolve.migration_size.

- `migration_topology`: Migration topology ('ring', 'fully_connected', 'random'). If None, uses value from DEFAULT_CONFIG.kiss_evolve.migration_topology.

- `enable_novelty_rejection`: Enable code novelty rejection sampling. If None, uses value from DEFAULT_CONFIG.kiss_evolve.enable_novelty_rejection.

- `novelty_threshold`: Cosine similarity threshold for rejecting code (0.0-1.0, higher = more strict). If None, uses value from DEFAULT_CONFIG.kiss_evolve.novelty_threshold.

- `max_rejection_attempts`: Maximum number of rejection attempts before accepting a variant anyway. If None, uses value from DEFAULT_CONFIG.kiss_evolve.max_rejection_attempts.

- `novelty_rag_model`: Model to use for generating code embeddings. If None and novelty rejection is enabled, uses the first model from models list.

- `parent_sampling_method`: Parent sampling method ('tournament', 'power_law', or 'performance_novelty'). If None, uses value from DEFAULT_CONFIG.kiss_evolve.parent_sampling_method.

- `power_law_alpha`: Power-law sampling parameter (α) for rank-based sampling. Lower = more exploration, higher = more exploitation. If None, uses value from DEFAULT_CONFIG.kiss_evolve.power_law_alpha.

- `performance_novelty_lambda`: Performance-novelty sampling parameter (λ) controlling selection pressure. If None, uses value from DEFAULT_CONFIG.kiss_evolve.performance_novelty_lambda.

- **evolve** — Run the evolutionary algorithm.<br/>`evolve() -> CodeVariant`

  - **Returns:** CodeVariant: The best code variant found during evolution.

- **get_best_variant** — Get the best variant from the current population or islands.<br/>`get_best_variant() -> CodeVariant`

  - **Returns:** The CodeVariant with the highest fitness from the current population or all islands. Returns a default variant with initial_code if no population exists.

- **get_population_stats** — Get statistics about the current population.<br/>`get_population_stats() -> dict[str, Any]`

  - **Returns:** Dictionary containing: - size: Total population size - avg_fitness: Average fitness across all variants - best_fitness: Maximum fitness value - worst_fitness: Minimum fitness value

##### `class SimpleRAG` — Simple and elegant RAG system for document storage and retrieval.

**Constructor:** `SimpleRAG(model_name: str, metric: str = 'cosine', embedding_model_name: str | None = None)`

- `model_name`: Model name to use for the LLM provider.

- `metric`: Distance metric to use - "cosine" or "l2" (default: "cosine").

- `embedding_model_name`: Optional specific model name for embeddings. If None, uses model_name or provider default.

- **add_documents** — Add documents to the vector store.<br/>`add_documents(documents: list[dict[str, Any]], batch_size: int = 100) -> None`

  - `documents`: List of document dictionaries. Each document should have: - "id": Unique identifier (str) - "text": Document text content (str) - "metadata": Optional metadata dictionary (dict)
  - `batch_size`: Number of documents to process in each batch (default: 100).
  - **Returns:** None.

- **query** — Query similar documents from the collection.<br/>`query(query_text: str, top_k: int = 5, filter_fn: Callable[[dict[str, Any]], bool] | None = None) -> list[dict[str, Any]]`

  - `query_text`: Query text to search for.
  - `top_k`: Number of top results to return (default: 5).
  - `filter_fn`: Optional filter function that takes a document dict and returns bool.
  - **Returns:** List of dictionaries containing: - "id": Document ID - "text": Document text - "metadata": Document metadata - "score": Similarity score (higher is better for cosine, lower for L2)

- **delete_documents** — Delete documents from the collection by their IDs.<br/>`delete_documents(document_ids: list[str]) -> None`

  - `document_ids`: List of document IDs to delete.
  - **Returns:** None.

- **get_collection_stats** — Get statistics about the collection.<br/>`get_collection_stats() -> dict[str, Any]`

  - **Returns:** Dictionary containing collection statistics.

- **clear_collection** — Clear all documents from the collection.<br/>`clear_collection() -> None`

  - **Returns:** None.

- **get_document** — Get a document by its ID.<br/>`get_document(document_id: str) -> dict[str, Any] | None`

  - `document_id`: Document ID to retrieve.
  - **Returns:** Document dictionary or None if not found.

______________________________________________________________________

#### `kiss.agents.kiss_evolve.config` — *KISSEvolve-specific configuration that extends the main KISS config.*

##### `class KISSEvolveConfig(BaseModel)` — KISSEvolve-specific configuration settings.

______________________________________________________________________

### `kiss.docker` — *Docker wrapper module for the KISS agent framework.*

```python
from kiss.docker import DockerManager
```

#### `class DockerManager` — Manages Docker container lifecycle and command execution.

**Constructor:** `DockerManager(image_name: str, tag: str = 'latest', workdir: str = '/', mount_shared_volume: bool = True, ports: dict[int, int] | None = None) -> None`

- `image_name`: The name of the Docker image (e.g., 'ubuntu', 'python')

- `tag`: The tag/version of the image (default: 'latest')

- `workdir`: The working directory inside the container

- `mount_shared_volume`: Whether to mount a shared volume. Set to False for images that already have content in the workdir (e.g., SWE-bench).

- `ports`: Port mapping from container port to host port. Example: {8080: 8080} maps container port 8080 to host port 8080. Example: {80: 8000, 443: 8443} maps multiple ports.

- **open** — Pull and load a Docker image, then create and start a container.<br/>`open() -> None`

  - `image_name`: The name of the Docker image (e.g., 'ubuntu', 'python')
  - `tag`: The tag/version of the image (default: 'latest')

- **Bash** — Execute a bash command in the running Docker container.<br/>`Bash(command: str, description: str) -> str`

  - `command`: The bash command to execute
  - `description`: A short description of the command in natural language
  - **Returns:** The output of the command, including stdout, stderr, and exit code

- **get_host_port** — Get the host port mapped to a container port.<br/>`get_host_port(container_port: int) -> int | None`

  - `container_port`: The container port to look up.
  - **Returns:** The host port mapped to the container port, or None if not mapped.

- **close** — Stop and remove the Docker container. Handles cleanup of both the container and any temporary directories created for shared volumes.<br/>`close() -> None`

______________________________________________________________________

#### `kiss.agents.sorcar.shared_utils` — *Shared utilities for Sorcar agent backends (chatbot UI and VS Code).*

**`clean_llm_output`** — Strip whitespace and surrounding quotes from LLM output.<br/>`def clean_llm_output(text: str) -> str`

**`clip_autocomplete_suggestion`** — Return the autocomplete continuation, stripped of the query prefix. Removes the query prefix if the LLM echoed it, strips surrounding whitespace, and stops at newlines.<br/>`def clip_autocomplete_suggestion(query: str, suggestion: str) -> str`

**`model_vendor`** — Return (vendor_display_name, sort_order) for a model name.<br/>`def model_vendor(name: str) -> tuple[str, int]`

- `name`: The model name string.
- **Returns:** Tuple of (display name, numeric sort order).

**`generate_followup_text`** — Generate a follow-up task suggestion via LLM.<br/>`def generate_followup_text(task: str, result: str, model: str) -> str`

- `task`: The completed task description.
- `result`: The task result summary (truncated to 500 chars internally).
- `model`: The model to use for generation.
- **Returns:** Suggestion text, or empty string on failure.

**`rank_file_suggestions`** — Rank and filter file paths by query match, recency, and usage.<br/>`def rank_file_suggestions(file_cache: list[str], query: str, usage: dict[str, int], limit: int = 20) -> list[dict[str, str]]`

- `file_cache`: List of file paths to search.
- `query`: Case-insensitive substring to match against paths.
- `usage`: File usage counts keyed by path (insertion order encodes recency, last key = most recently used).
- `limit`: Maximum number of results to return.
- **Returns:** Sorted list of dicts with `type` (`"frequent"` or `"file"`) and `text` keys.

______________________________________________________________________

#### `kiss.agents.sorcar.stateful_sorcar_agent` — *Stateful Sorcar agent with chat-session persistence.*

##### `class StatefulSorcarAgent(SorcarAgent)` — SorcarAgent with chat-session state management.

**Constructor:** `StatefulSorcarAgent(name: str) -> None`

- **chat_id** — Return the current chat session ID.<br/>`chat_id() -> str` *(property)*

- **new_chat** — Reset to a new chat session (equivalent to VS Code 'Clear').<br/>`new_chat() -> None`

- **resume_chat** — Resume a previous chat session by looking up the task's chat_id. If the task has an associated `chat_id` in history, subsequent `run()` calls will continue that session.<br/>`resume_chat(task: str) -> None`

  - `task`: The task description string to look up.

- **build_chat_prompt** — Load chat context and augment prompt with previous tasks/results.<br/>`build_chat_prompt(prompt: str) -> str`

  - `prompt`: The original task prompt.
  - **Returns:** The augmented prompt with chat history prepended, or the original prompt if no prior context exists.

- **run** — Run the agent with chat-session context management. Loads prior chat context, persists the new task, augments the prompt with previous tasks/results, runs the underlying agent, and saves the result back to history.<br/>`run(prompt_template: str = '', **kwargs: Any) -> str`

  - `prompt_template`: The task prompt.
  - `**kwargs`: All other arguments forwarded to `SorcarAgent.run()`.
  - **Returns:** YAML string with 'success' and 'summary' keys.

______________________________________________________________________

#### `kiss.agents.vscode` — *KISS Sorcar VS Code Extension backend.*

______________________________________________________________________

#### `kiss.agents.vscode.server` — *VS Code extension backend server for Sorcar agent.*

##### `class VSCodePrinter(BaseBrowserPrinter)` — Printer that outputs JSON events to stdout for VS Code extension.

**Constructor:** `VSCodePrinter() -> None`

- **broadcast** — Write event as a JSON line to stdout and record it.<br/>`broadcast(event: dict[str, Any]) -> None`
  - `event`: The event dictionary to emit.

##### `class VSCodeServer` — Backend server for VS Code extension.

**Constructor:** `VSCodeServer() -> None`

- **run** — Main loop: read commands from stdin, execute them.<br/>`run() -> None`

______________________________________________________________________

### `kiss.channels` — *Channel integrations for KISS agents.*

______________________________________________________________________

#### `kiss.channels.gmail_agent` — *Gmail Agent — SorcarAgent extension with Gmail API tools.*

##### `class GmailAgent(SorcarAgent)`

**Constructor:** `GmailAgent() -> None`

- **run** — Run the Gmail agent with optional user-interaction callbacks.<br/>`run(model_name: str | None = None, prompt_template: str = '', arguments: dict[str, str] | None = None, max_steps: int | None = None, max_budget: float | None = None, work_dir: str | None = None, printer: Any = None, max_sub_sessions: int | None = None, docker_image: str | None = None, headless: bool | None = None, verbose: bool | None = None, current_editor_file: str | None = None, attachments: list | None = None, wait_for_user_callback: Callable[[str, str], None] | None = None, ask_user_question_callback: Callable[[str], str] | None = None) -> str`

______________________________________________________________________

#### `kiss.channels.slack_agent` — *Slack Agent — SorcarAgent extension with Slack API tools.*

##### `class SlackChannelBackend` — ChannelBackend implementation for Slack.

**Constructor:** `SlackChannelBackend() -> None`

- **connect** — Authenticate with Slack using the stored bot token.<br/>`connect() -> bool`

  - **Returns:** True on success, False on failure.

- **connection_info** — Human-readable connection status string.<br/>`connection_info() -> str` *(property)*

- **find_channel** — Find a Slack channel ID by name.<br/>`find_channel(name: str) -> str | None`

  - `name`: Channel name without '#'.
  - **Returns:** Channel ID string, or None if not found.

- **find_user** — Find a Slack user ID by display name or username.<br/>`find_user(username: str) -> str | None`

  - `username`: Slack username (without @).
  - **Returns:** User ID string, or None if not found.

- **join_channel** — Join a Slack channel (bot needs to be a member to read/post).<br/>`join_channel(channel_id: str) -> None`

  - `channel_id`: Channel ID to join.

- **poll_messages** — Poll a Slack channel for new messages. Retries up to 3 times on transient network errors (e.g. SSL handshake timeouts, connection resets) with exponential backoff.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

  - `channel_id`: Channel ID to poll.
  - `oldest`: Only return messages newer than this timestamp.
  - `limit`: Maximum number of messages to return.
  - **Returns:** Tuple of (messages sorted oldest-first, updated oldest timestamp).

- **send_message** — Send a message to a Slack channel, optionally in a thread.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

  - `channel_id`: Channel ID to post to.
  - `text`: Message text (supports Slack mrkdwn formatting).
  - `thread_ts`: If non-empty, reply in this thread.

- **wait_for_reply** — Poll a Slack thread for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str) -> str`

  - `channel_id`: Channel ID containing the thread.
  - `thread_ts`: Timestamp of the parent message (thread root).
  - `user_id`: User ID to wait for a reply from.
  - **Returns:** The text of the user's reply message.

- **is_from_bot** — Check if a message was sent by the bot itself.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

  - `msg`: Message dict from poll_messages.
  - **Returns:** True if the message is from the bot.

- **strip_bot_mention** — Remove bot mention markers from message text.<br/>`strip_bot_mention(text: str) -> str`

  - `text`: Raw message text.
  - **Returns:** Cleaned text with bot mentions removed.

- **list_channels** — List channels in the Slack workspace.<br/>`list_channels(types: str = 'public_channel', limit: int = 200, cursor: str = '') -> str`

  - `types`: Comma-separated channel types. Options: public_channel, private_channel, mpim, im. Default: "public_channel".
  - `limit`: Maximum number of channels to return (1-1000). Default: 200.
  - `cursor`: Pagination cursor for next page of results. Pass the value from the previous response's response_metadata.next_cursor.
  - **Returns:** JSON string with channel list (id, name, purpose, num_members) and pagination cursor.

- **read_messages** — Read messages from a Slack channel.<br/>`read_messages(channel: str, limit: int = 20, cursor: str = '', oldest: str = '', newest: str = '') -> str`

  - `channel`: Channel ID (e.g. "C01234567").
  - `limit`: Number of messages to return (1-1000). Default: 20.
  - `cursor`: Pagination cursor for next page.
  - `oldest`: Only messages after this Unix timestamp.
  - `newest`: Only messages before this Unix timestamp.
  - **Returns:** JSON string with messages (user, text, ts, thread_ts) and pagination cursor.

- **read_thread** — Read replies in a message thread.<br/>`read_thread(channel: str, thread_ts: str, limit: int = 50, cursor: str = '') -> str`

  - `channel`: Channel ID where the thread lives.
  - `thread_ts`: Timestamp of the parent message.
  - `limit`: Number of replies to return (1-1000). Default: 50.
  - `cursor`: Pagination cursor for next page.
  - **Returns:** JSON string with thread messages and pagination cursor.

- **post_message** — Send a message to a Slack channel.<br/>`post_message(channel: str, text: str, thread_ts: str = '', blocks: str = '') -> str`

  - `channel`: Channel ID or name (e.g. "C01234567" or "#general").
  - `text`: Message text (supports Slack mrkdwn formatting).
  - `thread_ts`: Optional parent message timestamp to reply in a thread.
  - `blocks`: Optional JSON string of Block Kit blocks for rich formatting. If provided, text becomes the fallback.
  - **Returns:** JSON string with ok status and the message timestamp (ts).

- **update_message** — Update an existing message in a Slack channel.<br/>`update_message(channel: str, ts: str, text: str, blocks: str = '') -> str`

  - `channel`: Channel ID where the message is.
  - `ts`: Timestamp of the message to update.
  - `text`: New message text.
  - `blocks`: Optional JSON string of Block Kit blocks.
  - **Returns:** JSON string with ok status and updated timestamp.

- **delete_message** — Delete a message from a Slack channel.<br/>`delete_message(channel: str, ts: str) -> str`

  - `channel`: Channel ID where the message is.
  - `ts`: Timestamp of the message to delete.
  - **Returns:** JSON string with ok status.

- **list_users** — List users in the Slack workspace.<br/>`list_users(limit: int = 200, cursor: str = '') -> str`

  - `limit`: Maximum number of users to return (1-1000). Default: 200.
  - `cursor`: Pagination cursor for next page.
  - **Returns:** JSON string with user list (id, name, real_name, is_bot) and pagination cursor.

- **get_user_info** — Get detailed information about a Slack user.<br/>`get_user_info(user: str) -> str`

  - `user`: User ID (e.g. "U01234567").
  - **Returns:** JSON string with user profile details.

- **create_channel** — Create a new Slack channel.<br/>`create_channel(name: str, is_private: bool = False) -> str`

  - `name`: Channel name (lowercase, no spaces, max 80 chars). Use hyphens instead of spaces.
  - `is_private`: If True, create a private channel. Default: False.
  - **Returns:** JSON string with the new channel's id and name.

- **invite_to_channel** — Invite users to a Slack channel.<br/>`invite_to_channel(channel: str, users: str) -> str`

  - `channel`: Channel ID to invite users to.
  - `users`: Comma-separated list of user IDs to invite.
  - **Returns:** JSON string with ok status.

- **add_reaction** — Add an emoji reaction to a message.<br/>`add_reaction(channel: str, timestamp: str, name: str) -> str`

  - `channel`: Channel ID where the message is.
  - `timestamp`: Timestamp of the message to react to.
  - `name`: Emoji name without colons (e.g. "thumbsup", "heart").
  - **Returns:** JSON string with ok status.

- **search_messages** — Search for messages across the workspace.<br/>`search_messages(query: str, count: int = 20, sort: str = 'timestamp') -> str`

  - `query`: Search query string (supports Slack search modifiers like "in:#channel", "from:@user", "has:link").
  - `count`: Number of results to return (1-100). Default: 20.
  - `sort`: Sort order — "timestamp" (default) or "score".
  - **Returns:** JSON string with matching messages.

- **set_channel_topic** — Set the topic for a Slack channel.<br/>`set_channel_topic(channel: str, topic: str) -> str`

  - `channel`: Channel ID.
  - `topic`: New topic text.
  - **Returns:** JSON string with ok status.

- **upload_file** — Upload text content as a file to Slack channels.<br/>`upload_file(channels: str, content: str, filename: str, title: str = '') -> str`

  - `channels`: Comma-separated channel IDs to share the file in.
  - `content`: Text content of the file.
  - `filename`: Name for the file (e.g. "report.txt").
  - `title`: Optional title for the file.
  - **Returns:** JSON string with ok status and file id.

- **get_channel_info** — Get detailed information about a Slack channel.<br/>`get_channel_info(channel: str) -> str`

  - `channel`: Channel ID (e.g. "C01234567").
  - **Returns:** JSON string with channel details (name, topic, purpose, num_members, created, creator).

- **get_tool_methods** — Return list of bound tool methods for use by the LLM agent. Automatically discovers all public methods of this class, excluding ChannelBackend protocol/infrastructure methods.<br/>`get_tool_methods() -> list`

  - **Returns:** List of callable tool methods for Slack API operations.

##### `class SlackAgent(SorcarAgent)`

**Constructor:** `SlackAgent() -> None`

- **run** — Run the Slack agent with optional user-interaction callbacks.<br/>`run(model_name: str | None = None, prompt_template: str = '', arguments: dict[str, str] | None = None, max_steps: int | None = None, max_budget: float | None = None, work_dir: str | None = None, printer: Any = None, max_sub_sessions: int | None = None, docker_image: str | None = None, headless: bool | None = None, verbose: bool | None = None, current_editor_file: str | None = None, attachments: list | None = None, wait_for_user_callback: Callable[[str, str], None] | None = None, ask_user_question_callback: Callable[[str], str] | None = None) -> str`

______________________________________________________________________

#### `kiss.channels.whatsapp_agent` — *WhatsApp Agent — SorcarAgent extension with WhatsApp Business Cloud API tools.*

##### `class WhatsAppAgent(SorcarAgent)`

**Constructor:** `WhatsAppAgent() -> None`

- **run** — Run the WhatsApp agent with optional user-interaction callbacks.<br/>`run(model_name: str | None = None, prompt_template: str = '', arguments: dict[str, str] | None = None, max_steps: int | None = None, max_budget: float | None = None, work_dir: str | None = None, printer: Any = None, max_sub_sessions: int | None = None, docker_image: str | None = None, headless: bool | None = None, verbose: bool | None = None, current_editor_file: str | None = None, attachments: list | None = None, wait_for_user_callback: Callable[[str, str], None] | None = None, ask_user_question_callback: Callable[[str], str] | None = None) -> str`

______________________________________________________________________

#### `kiss.core.models.novita_model` — *Novita model implementation using OpenAI-compatible API.*

##### `class NovitaModel(OpenAICompatibleModel)` — A model that uses Novita's OpenAI-compatible API.

**Constructor:** `NovitaModel(model_name: str, api_key: str, model_config: dict | None = None, token_callback: TokenCallback | None = None)`

- `model_name`: The name of the Novita model to use.
- `api_key`: The Novita API key for authentication.
- `model_config`: Optional dictionary of model configuration parameters.
- `token_callback`: Optional async callback invoked with each streamed text token.

______________________________________________________________________
