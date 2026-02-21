# KISS Framework API Reference

> **Auto-generated** from source code by `generate_api_docs.py`.
> Run `uv run generate-api-docs` to regenerate.

______________________________________________________________________

## Table of Contents

- [`kiss`](#kiss)
  - [`kiss.core`](#kisscore)
    - [`kiss.core.kiss_agent`](#kisscorekiss-agent)
    - [`kiss.core.config`](#kisscoreconfig)
    - [`kiss.core.config_builder`](#kisscoreconfig-builder)
    - [`kiss.core.models`](#kisscoremodels)
      - [`kiss.core.models.model_info`](#kisscoremodelsmodel-info)
      - [`kiss.core.models.openai_compatible_model`](#kisscoremodelsopenai-compatible-model)
      - [`kiss.core.models.anthropic_model`](#kisscoremodelsanthropic-model)
      - [`kiss.core.models.gemini_model`](#kisscoremodelsgemini-model)
    - [`kiss.core.printer`](#kisscoreprinter)
    - [`kiss.core.print_to_console`](#kisscoreprint-to-console)
    - [`kiss.core.print_to_browser`](#kisscoreprint-to-browser)
    - [`kiss.core.browser_ui`](#kisscorebrowser-ui)
    - [`kiss.core.useful_tools`](#kisscoreuseful-tools)
    - [`kiss.core.web_use_tool`](#kisscoreweb-use-tool)
    - [`kiss.core.utils`](#kisscoreutils)
  - [`kiss.agents`](#kissagents)
    - [`kiss.agents.coding_agents`](#kissagentscoding-agents)
      - [`kiss.agents.coding_agents.claude_coding_agent`](#kissagentscoding-agentsclaude-coding-agent)
      - [`kiss.agents.coding_agents.relentless_coding_agent`](#kissagentscoding-agentsrelentless-coding-agent)
      - [`kiss.agents.coding_agents.config`](#kissagentscoding-agentsconfig)
    - [`kiss.agents.assistant`](#kissagentsassistant)
      - [`kiss.agents.assistant.relentless_agent`](#kissagentsassistantrelentless-agent)
      - [`kiss.agents.assistant.assistant_agent`](#kissagentsassistantassistant-agent)
      - [`kiss.agents.assistant.assistant`](#kissagentsassistantassistant)
      - [`kiss.agents.assistant.config`](#kissagentsassistantconfig)
    - [`kiss.agents.gepa`](#kissagentsgepa)
      - [`kiss.agents.gepa.config`](#kissagentsgepaconfig)
    - [`kiss.agents.kiss_evolve`](#kissagentskiss-evolve)
      - [`kiss.agents.kiss_evolve.config`](#kissagentskiss-evolveconfig)
  - [`kiss.docker`](#kissdocker)
  - [`kiss.multiprocessing`](#kissmultiprocessing)
  - [`kiss.rag`](#kissrag)

______________________________________________________________________

## `kiss`

*Top-level Kiss module for the project.*

```python
from kiss import __version__
```

______________________________________________________________________

### `kiss.core`

*Core module for the KISS agent framework.*

```python
from kiss.core import AgentConfig, AnthropicModel, Config, DEFAULT_CONFIG, GeminiModel, KISSError, Model, OpenAICompatibleModel
```

#### `AgentConfig`

```python
class AgentConfig(BaseModel)
```

#### `Config`

```python
class Config(BaseModel)
```

#### `KISSError`

```python
class KISSError(ValueError)
```

Custom exception class for KISS framework errors.

______________________________________________________________________

#### `kiss.core.kiss_agent`

*Core KISS agent implementation with native function calling support.*

##### `KISSAgent`

```python
class KISSAgent(Base)
```

A KISS agent using native function calling.

**Constructor:**

```python
KISSAgent(name: str) -> None
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `run(model_name: str, prompt_template: str, arguments: dict[str, str] \| None = None, tools: list[Callable[..., Any]] \| None = None, is_agentic: bool = True, max_steps: int \| None = None, max_budget: float \| None = None, model_config: dict[str, Any] \| None = None, printer: Printer \| None = None, print_to_console: bool \| None = None, print_to_browser: bool \| None = None) -> str` | Runs the agent's main ReAct loop to solve the task. |
| `finish` | `finish(result: str) -> str` | The agent must call this function with the final answer to the task. |

______________________________________________________________________

#### `kiss.core.config`

*Configuration Pydantic models for KISS agent settings with CLI support.*

##### `APIKeysConfig`

```python
class APIKeysConfig(BaseModel)
```

##### `DockerConfig`

```python
class DockerConfig(BaseModel)
```

______________________________________________________________________

#### `kiss.core.config_builder`

*Configuration builder for KISS agent settings with CLI support.*

**`add_config`**

```python
def add_config(name: str, config_class: type[BaseModel]) -> None
```

Build the KISS config, optionally overriding with command-line arguments.

______________________________________________________________________

#### `kiss.core.models`

*Model implementations for different LLM providers.*

```python
from kiss.core.models import Model, AnthropicModel, OpenAICompatibleModel, GeminiModel
```

##### `Model`

```python
class Model(ABC)
```

Abstract base class for LLM provider implementations.

**Constructor:**

```python
Model(model_name: str, model_description: str = '', model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `close_callback_loop` | `close_callback_loop() -> None` | Close the per-instance event loop used for synchronous token callback invocation. |
| `initialize` | `initialize(prompt: str) -> None` | Initializes the conversation with an initial user prompt. |
| `generate` | `generate() -> tuple[str, Any]` | Generates content from prompt. |
| `generate_and_process_with_tools` | `generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]]) -> tuple[list[dict[str, Any]], str, Any]` | Generates content with tools, processes the response, and adds it to conversation. |
| `add_function_results_to_conversation_and_return` | `add_function_results_to_conversation_and_return(function_results: list[tuple[str, dict[str, Any]]]) -> None` | Adds function results to the conversation state. |
| `add_message_to_conversation` | `add_message_to_conversation(role: str, content: str) -> None` | Adds a message to the conversation state. |
| `extract_input_output_token_counts_from_response` | `extract_input_output_token_counts_from_response(response: Any) -> tuple[int, int]` | Extracts input and output token counts from an API response. |
| `get_embedding` | `get_embedding(text: str, embedding_model: str \| None = None) -> list[float]` | Generates an embedding vector for the given text. |
| `set_usage_info_for_messages` | `set_usage_info_for_messages(usage_info: str) -> None` | Sets token information to append to messages sent to the LLM. |

______________________________________________________________________

#### `kiss.core.models.model_info`

*Model information: pricing and context lengths for supported LLM providers.*

##### `ModelInfo`

```python
class ModelInfo
```

Container for model metadata including pricing and capabilities.

**Constructor:**

```python
ModelInfo(context_length: int, input_price_per_million: float, output_price_per_million: float, is_function_calling_supported: bool, is_embedding_supported: bool, is_generation_supported: bool)
```

**`is_model_flaky`**

```python
def is_model_flaky(model_name: str) -> bool
```

Check if a model is known to be flaky.

**`get_flaky_reason`**

```python
def get_flaky_reason(model_name: str) -> str
```

Get the reason why a model is flaky.

**`model`**

```python
def model(model_name: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None) -> Model
```

Get a model instance based on model name prefix.

**`get_available_models`**

```python
def get_available_models() -> list[str]
```

Return model names for which an API key is configured and generation is supported.

**`calculate_cost`**

```python
def calculate_cost(model_name: str, num_input_tokens: int, num_output_tokens: int) -> float
```

Calculates the cost in USD for the given token counts.

**`get_max_context_length`**

```python
def get_max_context_length(model_name: str) -> int
```

Returns the maximum context length supported by the model.

______________________________________________________________________

#### `kiss.core.models.openai_compatible_model`

*OpenAI-compatible model implementation for custom endpoints.*

##### `OpenAICompatibleModel`

```python
class OpenAICompatibleModel(Model)
```

A model that uses an OpenAI-compatible API with a custom base URL.

**Constructor:**

```python
OpenAICompatibleModel(model_name: str, base_url: str, api_key: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `initialize` | `initialize(prompt: str) -> None` | Initialize the conversation with an initial user prompt. |
| `generate` | `generate() -> tuple[str, Any]` | Generate content from prompt without tools. |
| `generate_and_process_with_tools` | `generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]]) -> tuple[list[dict[str, Any]], str, Any]` | Generate content with tools, process the response, and add it to conversation. |
| `add_function_results_to_conversation_and_return` | `add_function_results_to_conversation_and_return(function_results: list[tuple[str, dict[str, Any]]]) -> None` | Add function results to the conversation state. |
| `add_message_to_conversation` | `add_message_to_conversation(role: str, content: str) -> None` | Add a message to the conversation state. |
| `extract_input_output_token_counts_from_response` | `extract_input_output_token_counts_from_response(response: Any) -> tuple[int, int]` | Extract input and output token counts from an API response. |
| `get_embedding` | `get_embedding(text: str, embedding_model: str \| None = None) -> list[float]` | Generate an embedding vector for the given text. |

______________________________________________________________________

#### `kiss.core.models.anthropic_model`

*Anthropic model implementation for Claude models.*

##### `AnthropicModel`

```python
class AnthropicModel(Model)
```

A model that uses Anthropic's Messages API (Claude).

**Constructor:**

```python
AnthropicModel(model_name: str, api_key: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `initialize` | `initialize(prompt: str) -> None` | Initializes the conversation with an initial user prompt. |
| `generate` | `generate() -> tuple[str, Any]` | Generates content from the current conversation. |
| `generate_and_process_with_tools` | `generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]]) -> tuple[list[dict[str, Any]], str, Any]` | Generates content with tools and processes the response. |
| `add_function_results_to_conversation_and_return` | `add_function_results_to_conversation_and_return(function_results: list[tuple[str, dict[str, Any]]]) -> None` | Add tool results to the conversation. |
| `add_message_to_conversation` | `add_message_to_conversation(role: str, content: str) -> None` | Adds a message to the conversation state. |
| `extract_input_output_token_counts_from_response` | `extract_input_output_token_counts_from_response(response: Any) -> tuple[int, int]` | Extracts input and output token counts from an API response. |
| `get_embedding` | `get_embedding(text: str, embedding_model: str \| None = None) -> list[float]` | Generates an embedding vector for the given text. |

______________________________________________________________________

#### `kiss.core.models.gemini_model`

*Gemini model implementation for Google's GenAI models.*

##### `GeminiModel`

```python
class GeminiModel(Model)
```

A model that uses Google's GenAI API (Gemini).

**Constructor:**

```python
GeminiModel(model_name: str, api_key: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `initialize` | `initialize(prompt: str) -> None` | Initializes the conversation with an initial user prompt. |
| `generate` | `generate() -> tuple[str, Any]` | Generates content from prompt without tools. |
| `generate_and_process_with_tools` | `generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]]) -> tuple[list[dict[str, Any]], str, Any]` | Generates content with tools, processes the response, and adds it to conversation. |
| `add_function_results_to_conversation_and_return` | `add_function_results_to_conversation_and_return(function_results: list[tuple[str, dict[str, Any]]]) -> None` | Adds function results to the conversation state. |
| `add_message_to_conversation` | `add_message_to_conversation(role: str, content: str) -> None` | Adds a message to the conversation state. |
| `extract_input_output_token_counts_from_response` | `extract_input_output_token_counts_from_response(response: Any) -> tuple[int, int]` | Extracts input and output token counts from an API response. |
| `get_embedding` | `get_embedding(text: str, embedding_model: str \| None = None) -> list[float]` | Generates an embedding vector for the given text. |

______________________________________________________________________

#### `kiss.core.printer`

*Abstract base class and shared utilities for KISS agent printers.*

##### `Printer`

```python
class Printer(ABC)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `print` | `print(content: Any, type: str = 'text', **kwargs: Any) -> str` | Render content to the output destination. |
| `token_callback` | `async token_callback(token: str) -> None` | Handle a single streamed token from the LLM. |
| `reset` | `reset() -> None` | Reset the printer's internal streaming state between messages. |

##### `MultiPrinter`

```python
class MultiPrinter(Printer)
```

**Constructor:**

```python
MultiPrinter(printers: list[Printer]) -> None
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `print` | `print(content: Any, type: str = 'text', **kwargs: Any) -> str` | Dispatch a print call to all child printers. |
| `token_callback` | `async token_callback(token: str) -> None` | Forward a streamed token to all child printers. |
| `reset` | `reset() -> None` | Reset streaming state on all child printers. |

**`lang_for_path`**

```python
def lang_for_path(path: str) -> str
```

Map a file path to its syntax-highlighting language name.

**`truncate_result`**

```python
def truncate_result(content: str) -> str
```

Truncate long content to MAX_RESULT_LEN, keeping the first and last halves.

**`extract_path_and_lang`**

```python
def extract_path_and_lang(tool_input: dict) -> tuple[str, str]
```

Extract the file path and inferred language from a tool input dict.

**`extract_extras`**

```python
def extract_extras(tool_input: dict) -> dict[str, str]
```

Extract non-standard keys from a tool input dict for display.

______________________________________________________________________

#### `kiss.core.print_to_console`

*Console output formatting for KISS agents.*

##### `ConsolePrinter`

```python
class ConsolePrinter(Printer)
```

**Constructor:**

```python
ConsolePrinter(file: Any = None) -> None
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset` | `reset() -> None` | Reset internal streaming and tool-parsing state for a new turn. |
| `print` | `print(content: Any, type: str = 'text', **kwargs: Any) -> str` | Render content to the console using Rich formatting. |
| `token_callback` | `async token_callback(token: str) -> None` | Stream a single token to the console, styled by current block type. |

______________________________________________________________________

#### `kiss.core.print_to_browser`

*Browser output streaming for KISS agents via SSE.*

##### `BrowserPrinter`

```python
class BrowserPrinter(BaseBrowserPrinter)
```

**Constructor:**

```python
BrowserPrinter() -> None
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `start` | `start(open_browser: bool = True) -> None` | Launch a local SSE server and optionally open the browser viewer. |
| `stop` | `stop() -> None` | Broadcast a done event to all clients and shut down the SSE server. |

______________________________________________________________________

#### `kiss.core.browser_ui`

*Shared browser UI components for KISS agent viewers.*

##### `BaseBrowserPrinter`

```python
class BaseBrowserPrinter(Printer)
```

**Constructor:**

```python
BaseBrowserPrinter() -> None
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset` | `reset() -> None` | Reset internal streaming and tool-parsing state for a new turn. |
| `broadcast` | `broadcast(event: dict[str, Any]) -> None` | Send an SSE event dict to all connected clients. |
| `add_client` | `add_client() -> queue.Queue[dict[str, Any]]` | Register a new SSE client and return its event queue. |
| `remove_client` | `remove_client(cq: queue.Queue[dict[str, Any]]) -> None` | Unregister an SSE client's event queue. |
| `print` | `print(content: Any, type: str = 'text', **kwargs: Any) -> str` | Render content by broadcasting SSE events to connected browser clients. |
| `token_callback` | `async token_callback(token: str) -> None` | Broadcast a streamed token as an SSE delta event to browser clients. |

**`find_free_port`**

```python
def find_free_port() -> int
```

Find and return an available TCP port on localhost.

**`build_stream_viewer_html`**

```python
def build_stream_viewer_html(title: str = 'KISS Agent', subtitle: str = 'Live Stream') -> str
```

Build a self-contained HTML page for the SSE-based stream viewer.

______________________________________________________________________

#### `kiss.core.useful_tools`

*Useful tools for agents: file editing, bash execution, web search, and URL fetching.*

##### `UsefulTools`

```python
class UsefulTools
```

A hardened collection of useful tools with improved security.

**Constructor:**

```python
UsefulTools(base_dir: str, readable_paths: list[str] | None = None, writable_paths: list[str] | None = None) -> None
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `Read` | `Read(file_path: str, max_lines: int = 2000) -> str` | Read file contents. |
| `Write` | `Write(file_path: str, content: str) -> str` | Write content to a file, creating it if it doesn't exist or overwriting if it does. |
| `Edit` | `Edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False, timeout_seconds: float = 30) -> str` | Performs precise string replacements in files with exact matching. |
| `MultiEdit` | `MultiEdit(file_path: str, old_string: str, new_string: str, replace_all: bool = False, timeout_seconds: float = 30) -> str` | Performs precise string replacements in files with exact matching. |
| `Bash` | `Bash(command: str, description: str, timeout_seconds: float = 30, max_output_chars: int = 50000) -> str` | Runs a bash command and returns its output. |

**`fetch_url`**

```python
def fetch_url(url: str, headers: dict[str, str], max_characters: int = 10000, timeout_seconds: float = 10.0) -> str
```

Fetch and extract text content from a URL using BeautifulSoup.

**`search_web`**

```python
def search_web(query: str, max_results: int = 10) -> str
```

Perform a web search and return the top search results with page contents.

**`parse_bash_command_paths`**

```python
def parse_bash_command_paths(command: str) -> tuple[list[str], list[str]]
```

Parse a bash command to extract readable and writable directory paths.

______________________________________________________________________

#### `kiss.core.web_use_tool`

*Browser automation tool for LLM agents using Playwright.*

##### `WebUseTool`

```python
class WebUseTool
```

Browser automation tool using Playwright with zero JS injection.

**Constructor:**

```python
WebUseTool(browser_type: str = 'chromium', headless: bool = False, viewport: tuple[int, int] = (1280, 900), user_data_dir: str | None = _AUTO_DETECT) -> None
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `go_to_url` | `go_to_url(url: str) -> str` | Navigate the browser to a URL and return the page accessibility tree. |
| `click` | `click(element_id: int, action: str = 'click') -> str` | Click or hover on an interactive element by its [N] ID from the accessibility tree. |
| `type_text` | `type_text(element_id: int, text: str, press_enter: bool = False) -> str` | Type text into a textbox, searchbox, or other editable element by its [N] ID. |
| `press_key` | `press_key(key: str) -> str` | Press a single key or key combination. Use for navigation, closing dialogs, shortcuts. |
| `scroll` | `scroll(direction: str = 'down', amount: int = 3) -> str` | Scroll the current page to reveal more content. Use when needed elements are off-screen. |
| `screenshot` | `screenshot(file_path: str = 'screenshot.png') -> str` | Capture the visible viewport as an image. Use to verify layout, captchas, or |
| `get_page_content` | `get_page_content(text_only: bool = False) -> str` | Get the current page content. Use to decide what to click or type next. |
| `close` | `close() -> str` | Close the browser and release resources. Call when done with the session or before exit. |
| `get_tools` | `get_tools() -> list[Callable[..., str]]` | Return callable web tools for registration with an agent. |

______________________________________________________________________

#### `kiss.core.utils`

*Utility functions for the KISS core module.*

**`get_config_value`**

```python
def get_config_value(value: T | None, config_obj: Any, attr_name: str, default: T | None = None) -> T
```

Get a config value, preferring explicit value over config default.

**`get_template_field_names`**

```python
def get_template_field_names(text: str) -> list[str]
```

Get the field names from the text.

**`add_prefix_to_each_line`**

```python
def add_prefix_to_each_line(text: str, prefix: str) -> str
```

Adds a prefix to each line of the text.

**`config_to_dict`**

```python
def config_to_dict() -> dict[Any, Any]
```

Convert the config to a dictionary.

**`fc`**

```python
def fc(file_path: str) -> str
```

Reads a file and returns the content.

**`finish`**

```python
def finish(status: str = 'success', analysis: str = '', result: str = '') -> str
```

The agent must call this function with the final status, analysis, and result

**`read_project_file`**

```python
def read_project_file(file_path_relative_to_project_root: str) -> str
```

Read a file from the project root.

**`read_project_file_from_package`**

```python
def read_project_file_from_package(file_name_as_python_package: str) -> str
```

Read a file from the project root.

**`resolve_path`**

```python
def resolve_path(p: str, base_dir: str) -> Path
```

Resolve a path relative to base_dir if not absolute.

**`is_subpath`**

```python
def is_subpath(target: Path, whitelist: list[Path]) -> bool
```

Check if target has any prefix in whitelist.

______________________________________________________________________

### `kiss.agents`

*KISS agents package with pre-built agent implementations.*

```python
from kiss.agents import ClaudeCodingAgent, prompt_refiner_agent, get_run_simple_coding_agent, run_bash_task_in_sandboxed_ubuntu_latest
```

**`prompt_refiner_agent`**

```python
def prompt_refiner_agent(original_prompt_template: str, previous_prompt_template: str, agent_trajectory_summary: str, model_name: str) -> str
```

Refines the prompt template based on the agent's trajectory summary.

**`get_run_simple_coding_agent`**

```python
def get_run_simple_coding_agent(test_fn: Callable[[str], bool]) -> Callable[..., str]
```

Return a function that runs a simple coding agent with a test function.

**`run_bash_task_in_sandboxed_ubuntu_latest`**

```python
def run_bash_task_in_sandboxed_ubuntu_latest(task: str, model_name: str) -> str
```

Run a bash task in a sandboxed Ubuntu latest container.

______________________________________________________________________

#### `kiss.agents.coding_agents`

*Coding agents for KISS framework.*

```python
from kiss.agents.coding_agents import Base, CODING_INSTRUCTIONS, ClaudeCodingAgent
```

##### `Base`

```python
class Base
```

Base class for all KISS agents with common state management and persistence.

**Constructor:**

```python
Base(name: str) -> None
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `set_printer` | `set_printer(printer: Printer \| None = None, print_to_console: bool \| None = None, print_to_browser: bool \| None = None) -> None` | Configure the output printer(s) for this agent. |
| `get_trajectory` | `get_trajectory() -> str` | Return the trajectory as JSON for visualization. |

______________________________________________________________________

#### `kiss.agents.coding_agents.claude_coding_agent`

*Claude Coding Agent using the Claude Agent SDK.*

##### `ClaudeCodingAgent`

```python
class ClaudeCodingAgent(Base)
```

**Constructor:**

```python
ClaudeCodingAgent(name: str) -> None
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `permission_handler` | `async permission_handler(tool_name: str, tool_input: dict[str, Any], context: ToolPermissionContext) -> PermissionResultAllow \| PermissionResultDeny` | Check whether a tool call is allowed based on path permissions. |
| `run` | `run(model_name: str \| None = None, prompt_template: str = '', arguments: dict[str, str] \| None = None, max_steps: int \| None = None, max_budget: float \| None = None, work_dir: str \| None = None, base_dir: str \| None = None, readable_paths: list[str] \| None = None, writable_paths: list[str] \| None = None, printer: Printer \| None = None, max_thinking_tokens: int = 1024, print_to_console: bool \| None = None, print_to_browser: bool \| None = None) -> str` | Run the Claude Coding Agent on a task using the Claude Agent SDK. |

______________________________________________________________________

#### `kiss.agents.coding_agents.relentless_coding_agent`

*Single-agent coding system with smart continuation for long tasks.*

##### `RelentlessCodingAgent`

```python
class RelentlessCodingAgent(RelentlessAgent)
```

Single-agent coding system with auto-continuation for infinite tasks.

**Constructor:**

```python
RelentlessCodingAgent(name: str) -> None
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `run(model_name: str \| None = None, prompt_template: str = '', arguments: dict[str, str] \| None = None, max_steps: int \| None = None, max_budget: float \| None = None, work_dir: str \| None = None, base_dir: str \| None = None, readable_paths: list[str] \| None = None, writable_paths: list[str] \| None = None, printer: Printer \| None = None, max_sub_sessions: int \| None = None, docker_image: str \| None = None, print_to_console: bool \| None = None, print_to_browser: bool \| None = None) -> str` | Run the coding agent with file and bash tools. |

______________________________________________________________________

#### `kiss.agents.coding_agents.config`

*Configuration Pydantic models for coding agent settings.*

##### `RelentlessCodingAgentConfig`

```python
class RelentlessCodingAgentConfig(BaseModel)
```

##### `CodingAgentConfig`

```python
class CodingAgentConfig(BaseModel)
```

______________________________________________________________________

#### `kiss.agents.assistant`

*Assistant agent with coding tools and browser automation.*

______________________________________________________________________

#### `kiss.agents.assistant.relentless_agent`

*Base relentless agent with smart continuation for long tasks.*

##### `RelentlessAgent`

```python
class RelentlessAgent(Base)
```

Base agent with auto-continuation for long tasks.

**Constructor:**

```python
RelentlessAgent(name: str) -> None
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `perform_task` | `perform_task(tools: list[Callable[..., Any]]) -> str` | Execute the task with auto-continuation across multiple sub-sessions. |
| `run` | `run(model_name: str \| None = None, prompt_template: str = '', arguments: dict[str, str] \| None = None, max_steps: int \| None = None, max_budget: float \| None = None, work_dir: str \| None = None, base_dir: str \| None = None, readable_paths: list[str] \| None = None, writable_paths: list[str] \| None = None, printer: Printer \| None = None, max_sub_sessions: int \| None = None, docker_image: str \| None = None, print_to_console: bool \| None = None, print_to_browser: bool \| None = None, tools_factory: Callable[[], list[Callable[..., Any]]] \| None = None, config_path: str = 'agent') -> str` | Run the agent with tools created by tools_factory (called after \_reset). |

**`finish`**

```python
def finish(success: bool, summary: str) -> str
```

Finish execution with status and summary.

______________________________________________________________________

#### `kiss.agents.assistant.assistant_agent`

*Assistant agent with both coding tools and browser automation.*

##### `AssistantAgent`

```python
class AssistantAgent(RelentlessAgent)
```

Agent with both coding tools and browser automation for web + code tasks.

**Constructor:**

```python
AssistantAgent(name: str) -> None
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `run(model_name: str \| None = None, prompt_template: str = '', arguments: dict[str, str] \| None = None, max_steps: int \| None = None, max_budget: float \| None = None, work_dir: str \| None = None, base_dir: str \| None = None, readable_paths: list[str] \| None = None, writable_paths: list[str] \| None = None, printer: Printer \| None = None, max_sub_sessions: int \| None = None, docker_image: str \| None = None, headless: bool \| None = None, print_to_console: bool \| None = None, print_to_browser: bool \| None = None) -> str` | Run the assistant agent with coding tools and browser automation. |

______________________________________________________________________

#### `kiss.agents.assistant.assistant`

*Browser-based chatbot for RelentlessAgent-based agents.*

**`run_chatbot`**

```python
def run_chatbot(agent_factory: Callable[[str], RelentlessAgent], title: str = 'KISS Assistant', subtitle: str = 'Interactive Agent', work_dir: str | None = None, default_model: str = 'claude-sonnet-4-5', agent_kwargs: dict[str, Any] | None = None) -> None
```

Run a browser-based chatbot UI for any RelentlessAgent-based agent.

______________________________________________________________________

#### `kiss.agents.assistant.config`

*Configuration for the Assistant Agent.*

##### `AssistantAgentConfig`

```python
class AssistantAgentConfig(BaseModel)
```

##### `AssistantConfig`

```python
class AssistantConfig(BaseModel)
```

______________________________________________________________________

#### `kiss.agents.gepa`

*GEPA (Genetic-Pareto) prompt optimization package.*

```python
from kiss.agents.gepa import GEPA, GEPAPhase, GEPAProgress, PromptCandidate, create_progress_callback
```

##### `GEPA`

```python
class GEPA
```

GEPA (Genetic-Pareto) prompt optimizer.

**Constructor:**

```python
GEPA(agent_wrapper: Callable[[str, dict[str, str]], tuple[str, list[Any]]], initial_prompt_template: str, evaluation_fn: Callable[[str], dict[str, float]] | None = None, max_generations: int | None = None, population_size: int | None = None, pareto_size: int | None = None, mutation_rate: float | None = None, reflection_model: str | None = None, dev_val_split: float | None = None, perfect_score: float = 1.0, use_merge: bool = True, max_merge_invocations: int = 5, merge_val_overlap_floor: int = 2, progress_callback: Callable[[GEPAProgress], None] | None = None, batched_agent_wrapper: Callable[[str, list[dict[str, str]]], list[tuple[str, list[Any]]]] | None = None)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `optimize` | `optimize(train_examples: list[dict[str, str]], dev_minibatch_size: int \| None = None) -> PromptCandidate` | Run GEPA optimization. |
| `get_pareto_frontier` | `get_pareto_frontier() -> list[PromptCandidate]` | Get a copy of the current Pareto frontier. |
| `get_best_prompt` | `get_best_prompt() -> str` | Get the best prompt template found during optimization. |

##### `GEPAPhase`

```python
class GEPAPhase(Enum)
```

Enum representing the current phase of GEPA optimization.

##### `GEPAProgress`

```python
class GEPAProgress
```

Progress information for GEPA optimization callbacks.

##### `PromptCandidate`

```python
class PromptCandidate
```

Represents a prompt candidate with its performance metrics.

**`create_progress_callback`**

```python
def create_progress_callback(verbose: bool = False) -> 'Callable[[GEPAProgress], None]'
```

Create a standard progress callback for GEPA optimization.

______________________________________________________________________

#### `kiss.agents.gepa.config`

*GEPA-specific configuration that extends the main KISS config.*

##### `GEPAConfig`

```python
class GEPAConfig(BaseModel)
```

GEPA-specific configuration settings.

______________________________________________________________________

#### `kiss.agents.kiss_evolve`

*KISSEvolve: Evolutionary Algorithm Discovery using LLMs.*

```python
from kiss.agents.kiss_evolve import CodeVariant, KISSEvolve
```

##### `CodeVariant`

```python
class CodeVariant
```

Represents a code variant in the evolutionary population.

##### `KISSEvolve`

```python
class KISSEvolve
```

KISSEvolve: Evolutionary algorithm discovery using LLMs.

**Constructor:**

```python
KISSEvolve(code_agent_wrapper: Callable[..., str], initial_code: str, evaluation_fn: Callable[[str], dict[str, Any]], model_names: list[tuple[str, float]], extra_coding_instructions: str = '', population_size: int | None = None, max_generations: int | None = None, mutation_rate: float | None = None, elite_size: int | None = None, num_islands: int | None = None, migration_frequency: int | None = None, migration_size: int | None = None, migration_topology: str | None = None, enable_novelty_rejection: bool | None = None, novelty_threshold: float | None = None, max_rejection_attempts: int | None = None, novelty_rag_model: Model | None = None, parent_sampling_method: str | None = None, power_law_alpha: float | None = None, performance_novelty_lambda: float | None = None)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `evolve` | `evolve() -> CodeVariant` | Run the evolutionary algorithm. |
| `get_best_variant` | `get_best_variant() -> CodeVariant` | Get the best variant from the current population or islands. |
| `get_population_stats` | `get_population_stats() -> dict[str, Any]` | Get statistics about the current population. |

______________________________________________________________________

#### `kiss.agents.kiss_evolve.config`

*KISSEvolve-specific configuration that extends the main KISS config.*

##### `KISSEvolveConfig`

```python
class KISSEvolveConfig(BaseModel)
```

KISSEvolve-specific configuration settings.

______________________________________________________________________

### `kiss.docker`

*Docker wrapper module for the KISS agent framework.*

```python
from kiss.docker import DockerManager
```

#### `DockerManager`

```python
class DockerManager
```

Manages Docker container lifecycle and command execution.

**Constructor:**

```python
DockerManager(image_name: str, tag: str = 'latest', workdir: str = '/', mount_shared_volume: bool = True, ports: dict[int, int] | None = None) -> None
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `open` | `open() -> None` | Pull and load a Docker image, then create and start a container. |
| `run_bash_command` | `run_bash_command(command: str, description: str) -> str` | Execute a bash command in the running Docker container. |
| `get_host_port` | `get_host_port(container_port: int) -> int \| None` | Get the host port mapped to a container port. |
| `close` | `close() -> None` | Stop and remove the Docker container. |

______________________________________________________________________

### `kiss.multiprocessing`

*Parallel execution utilities using multiprocessing.*

```python
from kiss.multiprocessing import get_available_cores, run_functions_in_parallel, run_functions_in_parallel_with_kwargs
```

**`get_available_cores`**

```python
def get_available_cores() -> int
```

Get the number of available CPU cores.

**`run_functions_in_parallel`**

```python
def run_functions_in_parallel(tasks: list[tuple[Callable[..., Any], list[Any]]]) -> list[Any]
```

Run a list of functions in parallel using multiprocessing.

**`run_functions_in_parallel_with_kwargs`**

```python
def run_functions_in_parallel_with_kwargs(functions: list[Callable[..., Any]], args_list: list[list[Any]] | None = None, kwargs_list: list[dict[str, Any]] | None = None) -> list[Any]
```

Run a list of functions in parallel using multiprocessing with support for kwargs.

______________________________________________________________________

### `kiss.rag`

*Simple RAG system for retrieval-augmented generation.*

```python
from kiss.rag import SimpleRAG
```

#### `SimpleRAG`

```python
class SimpleRAG
```

Simple and elegant RAG system for document storage and retrieval.

**Constructor:**

```python
SimpleRAG(model_name: str, metric: str = 'cosine', embedding_model_name: str | None = None)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_documents` | `add_documents(documents: list[dict[str, Any]], batch_size: int = 100) -> None` | Add documents to the vector store. |
| `query` | `query(query_text: str, top_k: int = 5, filter_fn: Callable[[dict[str, Any]], bool] \| None = None) -> list[dict[str, Any]]` | Query similar documents from the collection. |
| `delete_documents` | `delete_documents(document_ids: list[str]) -> None` | Delete documents from the collection by their IDs. |
| `get_collection_stats` | `get_collection_stats() -> dict[str, Any]` | Get statistics about the collection. |
| `clear_collection` | `clear_collection() -> None` | Clear all documents from the collection. |
| `get_document` | `get_document(document_id: str) -> dict[str, Any] \| None` | Get a document by its ID. |

______________________________________________________________________
