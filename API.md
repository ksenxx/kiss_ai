# KISS Framework API Reference

> **Auto-generated** — run `uv run generate-api-docs` to regenerate.

<details><summary><b>Table of Contents</b></summary>

- [`kiss`](#kiss)
  - [`kiss.core`](#kisscore)
    - [`kiss.core.kiss_agent`](#kisscorekiss_agent)
    - [`kiss.core.base`](#kisscorebase)
    - [`kiss.core.config`](#kisscoreconfig)
    - [`kiss.core.config_builder`](#kisscoreconfig_builder)
    - [`kiss.core.models`](#kisscoremodels)
      - [`kiss.core.models.model`](#kisscoremodelsmodel)
      - [`kiss.core.models.model_info`](#kisscoremodelsmodel_info)
      - [`kiss.core.models.openai_compatible_model`](#kisscoremodelsopenai_compatible_model)
      - [`kiss.core.models.anthropic_model`](#kisscoremodelsanthropic_model)
      - [`kiss.core.models.gemini_model`](#kisscoremodelsgemini_model)
    - [`kiss.core.printer`](#kisscoreprinter)
    - [`kiss.core.print_to_console`](#kisscoreprint_to_console)
      - [`kiss.agents.vscode.browser_ui`](#kissagentsvscodebrowser_ui)
      - [`kiss.agents.sorcar.useful_tools`](#kissagentssorcaruseful_tools)
      - [`kiss.agents.sorcar.web_use_tool`](#kissagentssorcarweb_use_tool)
    - [`kiss.core.utils`](#kisscoreutils)
  - [`kiss.agents`](#kissagents)
    - [`kiss.agents.kiss`](#kissagentskiss)
    - [`kiss.agents.sorcar`](#kissagentssorcar)
    - [`kiss.core.relentless_agent`](#kisscorerelentless_agent)
      - [`kiss.agents.sorcar.sorcar_agent`](#kissagentssorcarsorcar_agent)
    - [`kiss.agents.gepa`](#kissagentsgepa)
    - [`kiss.agents.kiss_evolve`](#kissagentskiss_evolve)
  - [`kiss.docker`](#kissdocker)
    - [`kiss.docker.docker_manager`](#kissdockerdocker_manager)
      - [`kiss.agents.sorcar.git_worktree`](#kissagentssorcargit_worktree)
      - [`kiss.agents.sorcar.stateful_sorcar_agent`](#kissagentssorcarstateful_sorcar_agent)
      - [`kiss.agents.sorcar.worktree_sorcar_agent`](#kissagentssorcarworktree_sorcar_agent)
    - [`kiss.agents.vscode`](#kissagentsvscode)
      - [`kiss.agents.vscode.helpers`](#kissagentsvscodehelpers)
        - [`kiss.agents.vscode.kiss_project.src.kiss`](#kissagentsvscodekiss_projectsrckiss)
          - [`kiss.agents.vscode.kiss_project.src.kiss.agents`](#kissagentsvscodekiss_projectsrckissagents)
            - [`kiss.agents.vscode.kiss_project.src.kiss.agents.gepa`](#kissagentsvscodekiss_projectsrckissagentsgepa)
              - [`kiss.agents.vscode.kiss_project.src.kiss.agents.gepa.gepa`](#kissagentsvscodekiss_projectsrckissagentsgepagepa)
            - [`kiss.agents.vscode.kiss_project.src.kiss.agents.kiss`](#kissagentsvscodekiss_projectsrckissagentskiss)
            - [`kiss.agents.vscode.kiss_project.src.kiss.agents.kiss_evolve`](#kissagentsvscodekiss_projectsrckissagentskiss_evolve)
              - [`kiss.agents.vscode.kiss_project.src.kiss.agents.kiss_evolve.kiss_evolve`](#kissagentsvscodekiss_projectsrckissagentskiss_evolvekiss_evolve)
              - [`kiss.agents.vscode.kiss_project.src.kiss.agents.kiss_evolve.simple_rag`](#kissagentsvscodekiss_projectsrckissagentskiss_evolvesimple_rag)
            - [`kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar`](#kissagentsvscodekiss_projectsrckissagentssorcar)
              - [`kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar.git_worktree`](#kissagentsvscodekiss_projectsrckissagentssorcargit_worktree)
              - [`kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar.sorcar_agent`](#kissagentsvscodekiss_projectsrckissagentssorcarsorcar_agent)
              - [`kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar.stateful_sorcar_agent`](#kissagentsvscodekiss_projectsrckissagentssorcarstateful_sorcar_agent)
              - [`kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar.useful_tools`](#kissagentsvscodekiss_projectsrckissagentssorcaruseful_tools)
              - [`kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar.web_use_tool`](#kissagentsvscodekiss_projectsrckissagentssorcarweb_use_tool)
              - [`kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar.worktree_sorcar_agent`](#kissagentsvscodekiss_projectsrckissagentssorcarworktree_sorcar_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.agents.vscode`](#kissagentsvscodekiss_projectsrckissagentsvscode)
              - [`kiss.agents.vscode.kiss_project.src.kiss.agents.vscode.browser_ui`](#kissagentsvscodekiss_projectsrckissagentsvscodebrowser_ui)
              - [`kiss.agents.vscode.kiss_project.src.kiss.agents.vscode.helpers`](#kissagentsvscodekiss_projectsrckissagentsvscodehelpers)
              - [`kiss.agents.vscode.kiss_project.src.kiss.agents.vscode.server`](#kissagentsvscodekiss_projectsrckissagentsvscodeserver)
          - [`kiss.agents.vscode.kiss_project.src.kiss.benchmarks`](#kissagentsvscodekiss_projectsrckissbenchmarks)
            - [`kiss.agents.vscode.kiss_project.src.kiss.benchmarks.generate_dashboard`](#kissagentsvscodekiss_projectsrckissbenchmarksgenerate_dashboard)
            - [`kiss.agents.vscode.kiss_project.src.kiss.benchmarks.swebench_pro`](#kissagentsvscodekiss_projectsrckissbenchmarksswebench_pro)
              - [`kiss.agents.vscode.kiss_project.src.kiss.benchmarks.swebench_pro.adapter`](#kissagentsvscodekiss_projectsrckissbenchmarksswebench_proadapter)
              - [`kiss.agents.vscode.kiss_project.src.kiss.benchmarks.swebench_pro.eval`](#kissagentsvscodekiss_projectsrckissbenchmarksswebench_proeval)
              - [`kiss.agents.vscode.kiss_project.src.kiss.benchmarks.swebench_pro.run`](#kissagentsvscodekiss_projectsrckissbenchmarksswebench_prorun)
            - [`kiss.agents.vscode.kiss_project.src.kiss.benchmarks.terminal_bench`](#kissagentsvscodekiss_projectsrckissbenchmarksterminal_bench)
              - [`kiss.agents.vscode.kiss_project.src.kiss.benchmarks.terminal_bench.agent`](#kissagentsvscodekiss_projectsrckissbenchmarksterminal_benchagent)
              - [`kiss.agents.vscode.kiss_project.src.kiss.benchmarks.terminal_bench.run`](#kissagentsvscodekiss_projectsrckissbenchmarksterminal_benchrun)
              - [`kiss.agents.vscode.kiss_project.src.kiss.benchmarks.terminal_bench.test_agent`](#kissagentsvscodekiss_projectsrckissbenchmarksterminal_benchtest_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.benchmarks.webarena`](#kissagentsvscodekiss_projectsrckissbenchmarkswebarena)
              - [`kiss.agents.vscode.kiss_project.src.kiss.benchmarks.webarena.agent`](#kissagentsvscodekiss_projectsrckissbenchmarkswebarenaagent)
              - [`kiss.agents.vscode.kiss_project.src.kiss.benchmarks.webarena.run`](#kissagentsvscodekiss_projectsrckissbenchmarkswebarenarun)
          - [`kiss.agents.vscode.kiss_project.src.kiss.channels`](#kissagentsvscodekiss_projectsrckisschannels)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels._backend_utils`](#kissagentsvscodekiss_projectsrckisschannels_backend_utils)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels._channel_agent_utils`](#kissagentsvscodekiss_projectsrckisschannels_channel_agent_utils)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.bluebubbles_agent`](#kissagentsvscodekiss_projectsrckisschannelsbluebubbles_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.discord_agent`](#kissagentsvscodekiss_projectsrckisschannelsdiscord_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.feishu_agent`](#kissagentsvscodekiss_projectsrckisschannelsfeishu_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.gmail_agent`](#kissagentsvscodekiss_projectsrckisschannelsgmail_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.googlechat_agent`](#kissagentsvscodekiss_projectsrckisschannelsgooglechat_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.imessage_agent`](#kissagentsvscodekiss_projectsrckisschannelsimessage_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.irc_agent`](#kissagentsvscodekiss_projectsrckisschannelsirc_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.line_agent`](#kissagentsvscodekiss_projectsrckisschannelsline_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.matrix_agent`](#kissagentsvscodekiss_projectsrckisschannelsmatrix_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.mattermost_agent`](#kissagentsvscodekiss_projectsrckisschannelsmattermost_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.msteams_agent`](#kissagentsvscodekiss_projectsrckisschannelsmsteams_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.nextcloud_talk_agent`](#kissagentsvscodekiss_projectsrckisschannelsnextcloud_talk_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.nostr_agent`](#kissagentsvscodekiss_projectsrckisschannelsnostr_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.phone_control_agent`](#kissagentsvscodekiss_projectsrckisschannelsphone_control_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.signal_agent`](#kissagentsvscodekiss_projectsrckisschannelssignal_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.slack_agent`](#kissagentsvscodekiss_projectsrckisschannelsslack_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.sms_agent`](#kissagentsvscodekiss_projectsrckisschannelssms_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.synology_chat_agent`](#kissagentsvscodekiss_projectsrckisschannelssynology_chat_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.telegram_agent`](#kissagentsvscodekiss_projectsrckisschannelstelegram_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.tlon_agent`](#kissagentsvscodekiss_projectsrckisschannelstlon_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.twitch_agent`](#kissagentsvscodekiss_projectsrckisschannelstwitch_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.whatsapp_agent`](#kissagentsvscodekiss_projectsrckisschannelswhatsapp_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.channels.zalo_agent`](#kissagentsvscodekiss_projectsrckisschannelszalo_agent)
          - [`kiss.agents.vscode.kiss_project.src.kiss.core`](#kissagentsvscodekiss_projectsrckisscore)
            - [`kiss.agents.vscode.kiss_project.src.kiss.core.base`](#kissagentsvscodekiss_projectsrckisscorebase)
            - [`kiss.agents.vscode.kiss_project.src.kiss.core.config`](#kissagentsvscodekiss_projectsrckisscoreconfig)
            - [`kiss.agents.vscode.kiss_project.src.kiss.core.config_builder`](#kissagentsvscodekiss_projectsrckisscoreconfig_builder)
            - [`kiss.agents.vscode.kiss_project.src.kiss.core.kiss_agent`](#kissagentsvscodekiss_projectsrckisscorekiss_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.core.kiss_error`](#kissagentsvscodekiss_projectsrckisscorekiss_error)
            - [`kiss.agents.vscode.kiss_project.src.kiss.core.models`](#kissagentsvscodekiss_projectsrckisscoremodels)
              - [`kiss.agents.vscode.kiss_project.src.kiss.core.models.anthropic_model`](#kissagentsvscodekiss_projectsrckisscoremodelsanthropic_model)
              - [`kiss.agents.vscode.kiss_project.src.kiss.core.models.claude_code_model`](#kissagentsvscodekiss_projectsrckisscoremodelsclaude_code_model)
              - [`kiss.agents.vscode.kiss_project.src.kiss.core.models.gemini_model`](#kissagentsvscodekiss_projectsrckisscoremodelsgemini_model)
              - [`kiss.agents.vscode.kiss_project.src.kiss.core.models.model`](#kissagentsvscodekiss_projectsrckisscoremodelsmodel)
              - [`kiss.agents.vscode.kiss_project.src.kiss.core.models.model_info`](#kissagentsvscodekiss_projectsrckisscoremodelsmodel_info)
              - [`kiss.agents.vscode.kiss_project.src.kiss.core.models.openai_compatible_model`](#kissagentsvscodekiss_projectsrckisscoremodelsopenai_compatible_model)
            - [`kiss.agents.vscode.kiss_project.src.kiss.core.print_to_console`](#kissagentsvscodekiss_projectsrckisscoreprint_to_console)
            - [`kiss.agents.vscode.kiss_project.src.kiss.core.printer`](#kissagentsvscodekiss_projectsrckisscoreprinter)
            - [`kiss.agents.vscode.kiss_project.src.kiss.core.relentless_agent`](#kissagentsvscodekiss_projectsrckisscorerelentless_agent)
            - [`kiss.agents.vscode.kiss_project.src.kiss.core.utils`](#kissagentsvscodekiss_projectsrckisscoreutils)
          - [`kiss.agents.vscode.kiss_project.src.kiss.docker`](#kissagentsvscodekiss_projectsrckissdocker)
            - [`kiss.agents.vscode.kiss_project.src.kiss.docker.docker_manager`](#kissagentsvscodekiss_projectsrckissdockerdocker_manager)
            - [`kiss.agents.vscode.kiss_project.src.kiss.docker.docker_tools`](#kissagentsvscodekiss_projectsrckissdockerdocker_tools)
      - [`kiss.agents.vscode.server`](#kissagentsvscodeserver)
  - [`kiss.benchmarks`](#kissbenchmarks)
    - [`kiss.benchmarks.generate_dashboard`](#kissbenchmarksgenerate_dashboard)
    - [`kiss.benchmarks.swebench_pro`](#kissbenchmarksswebench_pro)
      - [`kiss.benchmarks.swebench_pro.adapter`](#kissbenchmarksswebench_proadapter)
      - [`kiss.benchmarks.swebench_pro.eval`](#kissbenchmarksswebench_proeval)
      - [`kiss.benchmarks.swebench_pro.run`](#kissbenchmarksswebench_prorun)
    - [`kiss.benchmarks.terminal_bench`](#kissbenchmarksterminal_bench)
      - [`kiss.benchmarks.terminal_bench.agent`](#kissbenchmarksterminal_benchagent)
      - [`kiss.benchmarks.terminal_bench.run`](#kissbenchmarksterminal_benchrun)
      - [`kiss.benchmarks.terminal_bench.test_agent`](#kissbenchmarksterminal_benchtest_agent)
    - [`kiss.benchmarks.webarena`](#kissbenchmarkswebarena)
      - [`kiss.benchmarks.webarena.agent`](#kissbenchmarkswebarenaagent)
      - [`kiss.benchmarks.webarena.run`](#kissbenchmarkswebarenarun)
  - [`kiss.channels`](#kisschannels)
    - [`kiss.channels._backend_utils`](#kisschannels_backend_utils)
    - [`kiss.channels._channel_agent_utils`](#kisschannels_channel_agent_utils)
    - [`kiss.channels.bluebubbles_agent`](#kisschannelsbluebubbles_agent)
    - [`kiss.channels.discord_agent`](#kisschannelsdiscord_agent)
    - [`kiss.channels.feishu_agent`](#kisschannelsfeishu_agent)
    - [`kiss.channels.gmail_agent`](#kisschannelsgmail_agent)
    - [`kiss.channels.googlechat_agent`](#kisschannelsgooglechat_agent)
    - [`kiss.channels.imessage_agent`](#kisschannelsimessage_agent)
    - [`kiss.channels.irc_agent`](#kisschannelsirc_agent)
    - [`kiss.channels.line_agent`](#kisschannelsline_agent)
    - [`kiss.channels.matrix_agent`](#kisschannelsmatrix_agent)
    - [`kiss.channels.mattermost_agent`](#kisschannelsmattermost_agent)
    - [`kiss.channels.msteams_agent`](#kisschannelsmsteams_agent)
    - [`kiss.channels.nextcloud_talk_agent`](#kisschannelsnextcloud_talk_agent)
    - [`kiss.channels.nostr_agent`](#kisschannelsnostr_agent)
    - [`kiss.channels.phone_control_agent`](#kisschannelsphone_control_agent)
    - [`kiss.channels.signal_agent`](#kisschannelssignal_agent)
    - [`kiss.channels.slack_agent`](#kisschannelsslack_agent)
    - [`kiss.channels.sms_agent`](#kisschannelssms_agent)
    - [`kiss.channels.synology_chat_agent`](#kisschannelssynology_chat_agent)
    - [`kiss.channels.telegram_agent`](#kisschannelstelegram_agent)
    - [`kiss.channels.tlon_agent`](#kisschannelstlon_agent)
    - [`kiss.channels.twitch_agent`](#kisschannelstwitch_agent)
    - [`kiss.channels.whatsapp_agent`](#kisschannelswhatsapp_agent)
    - [`kiss.channels.zalo_agent`](#kisschannelszalo_agent)
      - [`kiss.core.models.claude_code_model`](#kisscoremodelsclaude_code_model)
    - [`kiss.docker.docker_tools`](#kissdockerdocker_tools)

</details>

______________________________________________________________________

## `kiss` — *Top-level Kiss module for the project.*

```python
from kiss import __version__
```

______________________________________________________________________

### `kiss.core` — *Core module for the KISS agent framework.*

```python
from kiss.core import Config, DEFAULT_CONFIG, KISSError
```

#### `class Config(BaseModel)`

#### `class KISSError(ValueError)` — Custom exception class for KISS framework errors.

______________________________________________________________________

#### `kiss.core.kiss_agent` — *Core KISS agent implementation with native function calling support.*

##### `class KISSAgent(Base)` — A KISS agent using native function calling.

**Constructor:** `KISSAgent(name: str) -> None`

- **run** — Runs the agent's main ReAct loop to solve the task.<br/>`run(model_name: str, prompt_template: str, arguments: dict[str, str] | None = None, system_prompt: str = '', tools: list[Callable[..., Any]] | None = None, is_agentic: bool = True, max_steps: int | None = None, max_budget: float | None = None, model_config: dict[str, Any] | None = None, printer: Printer | None = None, verbose: bool | None = None, attachments: list[Attachment] | None = None) -> str`

  - `model_name`: The name of the model to use for the agent.
  - `prompt_template`: The prompt template for the agent.
  - `arguments`: The arguments to be substituted into the prompt template. Default is None.
  - `system_prompt`: Optional system prompt to provide to the model. Default is empty string (no system prompt).
  - `tools`: The tools to use for the agent. If None, no tools are provided (only the built-in finish tool is added).
  - `is_agentic`: Whether the agent is agentic. Default is True.
  - `max_steps`: The maximum number of steps to take. Default is 100.
  - `max_budget`: The maximum budget to spend. Default is 10.0.
  - `model_config`: The model configuration to use for the agent. Default is None.
  - `printer`: Optional printer for streaming output. Default is None.
  - `verbose`: Whether to print output to console. Default is None (verbose enabled).
  - `attachments`: Optional file attachments (images, PDFs) to include in the initial prompt. Default is None.
  - **Returns:** str: The result of the agent's task.

- **finish** — The agent must call this function with the final answer to the task.<br/>`finish(result: str) -> str`

  - `result`: The result generated by the agent.
  - **Returns:** Returns the result of the agent's task.

______________________________________________________________________

#### `kiss.core.base` — *Base agent class with common functionality for all KISS agents.*

##### `class Base` — Base class for all KISS agents with common state management and persistence.

**Constructor:** `Base(name: str) -> None`

- `name`: The name identifier for the agent.

- **get_global_budget_used** — Return the global budget total under the shared class lock.<br/>`get_global_budget_used() -> float`

- **reset_global_budget** — Reset the shared process-wide budget counter to zero.<br/>`reset_global_budget() -> None`

- **set_printer** — Configure the output printer for this agent. If an explicit *printer* is provided, it is always used regardless of the verbose setting. Otherwise a `ConsolePrinter` is created when verbose output is enabled.<br/>`set_printer(printer: Printer | None = None, verbose: bool | None = None) -> None`

  - `printer`: An existing Printer instance to use directly. If provided, verbose is ignored.
  - `verbose`: Whether to print to the console. Defaults to True if None.

- **get_trajectory** — Return the trajectory as JSON for visualization.<br/>`get_trajectory() -> str`

  - **Returns:** str: A JSON-formatted string of all messages in the agent's history.

______________________________________________________________________

#### `kiss.core.config` — *Configuration Pydantic models for KISS agent settings with CLI support.*

**`set_artifact_base_dir`** — Set the base directory used to resolve `artifact_dir`.<br/>`def set_artifact_base_dir(base_dir: str | Path | None) -> str`

- `base_dir`: Directory whose `.kiss.artifacts` child should contain generated job artifacts. `None` resets to the project root.
- **Returns:** The resolved artifact job directory.

**`get_artifact_dir`** — Return the active artifact directory, creating it lazily if needed.<br/>`def get_artifact_dir() -> str`

______________________________________________________________________

#### `kiss.core.config_builder` — *Configuration builder for KISS agent settings with CLI support.*

**`add_config`** — Build the KISS config, optionally overriding with command-line arguments. This function accumulates configs - each call adds a new config field while preserving existing fields from previous calls.<br/>`def add_config(name: str, config_class: type[BaseModel]) -> None`

- `name`: Name of the config class.
- `config_class`: Class of the config.

______________________________________________________________________

#### `kiss.core.models` — *Model implementations for different LLM providers.*

```python
from kiss.core.models import Attachment, Model, AnthropicModel, ClaudeCodeModel, OpenAICompatibleModel, GeminiModel
```

##### `class Attachment` — A file attachment (image, document, audio, or video) to include in a prompt.

- **from_file** — Create an Attachment from a file path.<br/>`from_file(path: str) -> 'Attachment'`

  - `path`: Path to the file to attach.
  - **Returns:** An Attachment with the file's bytes and detected MIME type.

- **to_base64** — Return the file data as a base64-encoded string.<br/>`to_base64() -> str`

- **to_data_url** — Return a data: URL suitable for OpenAI image_url fields.<br/>`to_data_url() -> str`

##### `class Model(ABC)` — Abstract base class for LLM provider implementations.

**Constructor:** `Model(model_name: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)`

- `model_name`: The name/identifier of the model.

- `model_config`: Optional dictionary of model configuration parameters.

- `token_callback`: Optional callback invoked with each streamed text token.

- **reset_conversation** — Reset conversation state for reuse across sub-sessions. Clears the conversation history and usage info while keeping the HTTP client and model configuration intact.<br/>`reset_conversation() -> None`

- **initialize** — Initializes the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt to start the conversation.
  - `attachments`: Optional list of file attachments (images, PDFs, audio, video) to include. Provider support varies — unsupported types are skipped with a warning.

- **generate** — Generates content from prompt.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** tuple\[str, Any\]: A tuple of (generated_text, raw_response).

- **generate_and_process_with_tools** — Generates content with tools, processes the response, and adds it to conversation.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]], tools_schema: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - `tools_schema`: Optional pre-built tool schema list. When provided, skips schema rebuilding from function_map (performance optimization).
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

#### `kiss.core.models.model` — *Abstract base class for LLM provider model implementations.*

**`transcribe_audio`** — Transcribe audio bytes to text using OpenAI's Whisper API. This is used as a fallback for model providers that do not support audio attachments natively (e.g. Anthropic).<br/>`def transcribe_audio(data: bytes, mime_type: str, api_key: str | None = None) -> str`

- `data`: Raw audio file bytes.
- `mime_type`: MIME type of the audio (e.g. `"audio/mpeg"`).
- `api_key`: OpenAI API key. Falls back to the `OPENAI_API_KEY` environment variable when *None*.
- **Returns:** The transcribed text.

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

- `model_name`: The name of the model (with provider prefix if applicable). Accepts harbor-style `provider/model` names (e.g. `openai/gpt-5.4`, `anthropic/claude-opus-4-6`) — the redundant provider prefix is stripped automatically.
- `model_config`: Optional dictionary of model configuration parameters. If it contains "base_url", routing is bypassed and an OpenAICompatibleModel is built with that base_url and optional "api_key".
- `token_callback`: Optional callback invoked with each streamed text token.
- **Returns:** Model: An appropriate Model instance for the specified model.

**`get_available_models`** — Return model names for which an API key is configured and generation is supported.<br/>`def get_available_models() -> list[str]`

- **Returns:** list\[str\]: Sorted list of model name strings that have a configured API key and support text generation.

**`get_default_model`** — Return the best default model based on which API keys are configured. Priority order: Anthropic > OpenRouter > Gemini > OpenAI > Together AI. Falls back to `"claude-opus-4-6"` if no keys are set.<br/>`def get_default_model() -> str`

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

- `token_callback`: Optional callback invoked with each streamed text token.

- **initialize** — Initialize the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt to start the conversation.
  - `attachments`: Optional list of file attachments (images, PDFs) to include.

- **generate** — Generate content from prompt without tools.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** A tuple of (content, response) where content is the generated text and response is the raw API response object.

- **generate_and_process_with_tools** — Generate content with tools, process the response, and add it to conversation.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]], tools_schema: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - `tools_schema`: Optional pre-built tool schema list.
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

- `token_callback`: Optional callback invoked with each streamed text token.

- **initialize** — Initializes the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt to start the conversation.
  - `attachments`: Optional list of file attachments (images, PDFs, audio, video) to include. Audio attachments are automatically transcribed to text via OpenAI Whisper when an `OPENAI_API_KEY` is available; otherwise they are skipped with a warning. Video attachments are always skipped.

- **generate** — Generates content from the current conversation.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** tuple\[str, Any\]: A tuple of (generated_text, raw_response).

- **generate_and_process_with_tools** — Generates content with tools and processes the response.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]], tools_schema: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - `tools_schema`: Optional pre-built OpenAI-format tool schema list.
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

- `token_callback`: Optional callback invoked with each streamed text token.

- **reset_conversation** — Reset conversation state including thought signatures.<br/>`reset_conversation() -> None`

- **initialize** — Initializes the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt to start the conversation.
  - `attachments`: Optional list of file attachments (images, PDFs) to include.

- **generate** — Generates content from prompt without tools.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** tuple\[str, Any\]: A tuple of (generated_text, raw_response).

- **generate_and_process_with_tools** — Generates content with tools, processes the response, and adds it to conversation.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]], tools_schema: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - `tools_schema`: Optional pre-built OpenAI-format tool schema list.
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
  - `type`: Content type (e.g. "text", "prompt", "stream_event", "tool_call", "tool_result", "result", "message").
  - `**kwargs`: Additional type-specific options (e.g. tool_input, is_error).
  - **Returns:** str: Any extracted text (e.g. streamed text deltas), or empty string.

- **token_callback** — Handle a single streamed token from the LLM.<br/>`token_callback(token: str) -> None`

  - `token`: The text token to process.

- **reset** — Reset the printer's internal streaming state between messages.<br/>`reset() -> None`

##### `class MultiPrinter(Printer)`

**Constructor:** `MultiPrinter(printers: list[Printer]) -> None`

- **print** — Dispatch a print call to all child printers.<br/>`print(content: Any, type: str = 'text', **kwargs: Any) -> str`

  - `content`: The content to display.
  - `type`: Content type forwarded to each child printer.
  - `**kwargs`: Additional options forwarded to each child printer.
  - **Returns:** str: The first non-empty result from child printers.

- **token_callback** — Forward a streamed token to all child printers.<br/>`token_callback(token: str) -> None`

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
- **Returns:** dict\[str, str\]: Keys not in KNOWN_KEYS mapped to their string values.

______________________________________________________________________

#### `kiss.core.print_to_console` — *Console output formatting for KISS agents.*

##### `class ConsolePrinter(StreamEventParser, Printer)`

**Constructor:** `ConsolePrinter(file: Any = None) -> None`

- **reset** — Reset internal streaming and tool-parsing state for a new turn.<br/>`reset() -> None`

- **print** — Render content to the console using Rich formatting.<br/>`print(content: Any, type: str = 'text', **kwargs: Any) -> str`

  - `content`: The content to display.
  - `type`: Content type (e.g. "text", "prompt", "stream_event", "tool_call", "tool_result", "result", "message").
  - `**kwargs`: Additional options such as tool_input, is_error, cost, total_tokens.
  - **Returns:** str: Extracted text from stream events, or empty string.

- **token_callback** — Stream a single token to the console, styled by current block type.<br/>`token_callback(token: str) -> None`

  - `token`: The text token to display.

______________________________________________________________________

#### `kiss.agents.vscode.browser_ui` — *Shared browser UI components for KISS agent viewers.*

##### `class BaseBrowserPrinter(StreamEventParser, Printer)`

**Constructor:** `BaseBrowserPrinter() -> None`

- **tokens_offset** — Per-thread token offset for usage_info events.<br/>`tokens_offset() -> int` *(property)*

- **tokens_offset**<br/>`tokens_offset(value: int) -> None`

- **budget_offset** — Per-thread budget offset for usage_info events.<br/>`budget_offset() -> float` *(property)*

- **budget_offset**<br/>`budget_offset(value: float) -> None`

- **steps_offset** — Per-thread steps offset for usage_info events.<br/>`steps_offset() -> int` *(property)*

- **steps_offset**<br/>`steps_offset(value: int) -> None`

- **reset** — Reset internal streaming and tool-parsing state for a new turn.<br/>`reset() -> None`

- **start_recording** — Start recording broadcast events. Uses an explicit *recording_id* to avoid thread-ID reuse corruption. Falls back to thread ident when no ID is given (backward compat). When *tab_id* is provided, only events whose `tabId` matches are recorded. Events without a `tabId` are still recorded to all active recordings.<br/>`start_recording(recording_id: int | None = None, tab_id: str | None = None) -> None`

  - `recording_id`: Unique identifier for this recording session.
  - `tab_id`: Optional tab owner — restricts which events are recorded.

- **stop_recording** — Stop recording and return its display events.<br/>`stop_recording(recording_id: int | None = None) -> list[dict[str, Any]]`

  - `recording_id`: The recording ID passed to start_recording.
  - **Returns:** List of display-relevant events with consecutive deltas merged.

- **peek_recording** — Return a snapshot of the current recording without stopping it. Used for periodic crash-recovery flushes: the caller can persist a snapshot of events to the database while recording continues.<br/>`peek_recording(recording_id: int) -> list[dict[str, Any]]`

  - `recording_id`: The recording ID passed to start_recording.
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
  - `type`: Content type (e.g. "text", "prompt", "stream_event", "tool_call", "tool_result", "result", "message").
  - `**kwargs`: Additional options such as tool_input, is_error, cost, total_tokens.
  - **Returns:** str: Extracted text from stream events, or empty string.

- **token_callback** — Broadcast a streamed token as an SSE delta event to browser clients.<br/>`token_callback(token: str) -> None`

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

- **Bash** — Runs a bash command and returns its output.<br/>`Bash(command: str, description: str, timeout_seconds: float = 300, max_output_chars: int = 50000) -> str`

  - `command`: The bash command to run.
  - `description`: A brief description of the command.
  - `timeout_seconds`: Timeout in seconds for the command.
  - `max_output_chars`: Maximum characters in output before truncation.
  - **Returns:** The output of the command.

______________________________________________________________________

#### `kiss.agents.sorcar.web_use_tool` — *Browser automation tool for LLM agents using Playwright.*

##### `class WebUseTool` — Browser automation tool using non-headless Playwright Chromium.

**Constructor:** `WebUseTool(viewport: tuple[int, int] = (1280, 900), user_data_dir: str | None = _DEFAULT_USER_DATA_DIR, headless: bool = False, **_kwargs: Any) -> None`

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

- **get_tools** — Return callable web tools for registration with an agent.<br/>`get_tools() -> list[Callable[..., str]]`

  - **Returns:** List of callables: go_to_url, click, type_text, press_key, scroll, screenshot, get_page_content. Does not include close.

______________________________________________________________________

#### `kiss.core.utils` — *Utility functions for the KISS core module.*

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

**`finish`** — The agent must call this function with the final status, analysis, and result when it has solved the given task. Status **MUST** be 'success' or 'failure'.<br/>`def finish(status: str = 'success', analysis: str = '', result: str = '') -> str`

- `status`: The status of the agent's task ('success' or 'failure'). Defaults to 'success'.
- `analysis`: The analysis of the agent's trajectory.
- `result`: The result generated by the agent.
- **Returns:** A YAML string containing the status, analysis, and result of the agent's task.

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

______________________________________________________________________

#### `kiss.agents.kiss` — *Useful agents for the KISS Agent Framework.*

**`prompt_refiner_agent`** — Refines the prompt template based on the agent's trajectory summary.<br/>`def prompt_refiner_agent(original_prompt_template: str, previous_prompt_template: str, agent_trajectory_summary: str, model_name: str) -> str`

- `original_prompt_template`: The original prompt template.
- `previous_prompt_template`: The previous version of the prompt template that led to the given trajectory.
- `agent_trajectory_summary`: The agent's trajectory summary as a string.
- `model_name`: The name of the model to use for the agent.
- **Returns:** str: The refined prompt template.

**`run_bash_task_in_sandboxed_ubuntu_latest`** — Run a bash task in a sandboxed Ubuntu latest container.<br/>`def run_bash_task_in_sandboxed_ubuntu_latest(task: str, model_name: str) -> str`

- `task`: The task to run.
- `model_name`: The name of the model to use for the agent.
- **Returns:** str: The result of the task.

**`get_run_simple_coding_agent`** — Return a function that runs a simple coding agent with a test function.<br/>`def get_run_simple_coding_agent(test_fn: Callable[[str], bool]) -> Callable[..., str]`

- `test_fn`: The test function to use for the agent.
- **Returns:** Callable\[..., str\]: A function that runs a simple coding agent with a test function. Accepts keyword arguments: model_name (str), prompt_template (str), and arguments (dict[str, str]).

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
  - `verbose`: Whether to print output to console. Defaults to True.
  - `tools`: List of callable tools available to the agent during execution.
  - `attachments`: Optional file attachments (images, PDFs) for the initial prompt.
  - **Returns:** YAML string with 'success' and 'summary' keys.

**`finish`** — Finish execution with status and summary.<br/>`def finish(success: bool, is_continue: bool = False, summary: str = '') -> str`

- `success`: True if the agent has successfully completed the task, False otherwise
- `is_continue`: True if the task is incomplete and should continue, False otherwise
- `summary`: precise chronologically-ordered list of things the agent did with the reason for doing that along with relevant code snippets

______________________________________________________________________

#### `kiss.agents.sorcar.sorcar_agent` — *Sorcar agent with both coding tools and browser automation.*

##### `class SorcarAgent(RelentlessAgent)` — Agent with both coding tools and browser automation for web + code tasks.

**Constructor:** `SorcarAgent(name: str) -> None`

- **perform_task** — Execute the task, building docker-aware tools after docker_manager is set.<br/>`perform_task(tools: list, attachments: list | None = None) -> str`

  - `tools`: Extra tools passed by the caller (from run(tools=...)).
  - `attachments`: Optional file attachments for the initial prompt.
  - **Returns:** YAML string with 'success' and 'summary' keys.

- **run** — Run the assistant agent with coding tools and browser automation.<br/>`run(model_name: str | None = None, prompt_template: str = '', arguments: dict[str, str] | None = None, system_prompt: str | None = None, tools: list[Callable[..., Any]] | None = None, max_steps: int | None = None, max_budget: float | None = None, model_config: dict[str, Any] | None = None, work_dir: str | None = None, printer: Printer | None = None, max_sub_sessions: int | None = None, docker_image: str | None = None, web_tools: bool = True, is_parallel: bool = False, verbose: bool | None = None, current_editor_file: str | None = None, attachments: list[Attachment] | None = None, ask_user_question_callback: Callable[[str], str] | None = None) -> str`

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
  - `web_tools`: Whether to include browser/web tools. Defaults to True. Set to False for terminal-only environments.
  - `is_parallel`: Whether to include the run_parallel tool. Defaults to False. When True, the agent can spawn parallel sub-agents for independent tasks.
  - `verbose`: Whether to print output to console. Defaults to config verbose setting.
  - `current_editor_file`: Path to the currently active editor file, appended to prompt.
  - `attachments`: Optional file attachments (images, PDFs) for the initial prompt.
  - `ask_user_question_callback`: Optional callback used by the ask_user_question tool to collect a text response from the user.
  - **Returns:** YAML string with 'success' and 'summary' keys.

**`run_tasks_parallel`** — Execute multiple SorcarAgent tasks concurrently using threads. Each task gets its own `SorcarAgent` instance and runs in a separate thread via :class:`~concurrent.futures.ThreadPoolExecutor`. This is ideal for I/O-bound workloads (LLM API calls, network requests) where the GIL is released during I/O waits.<br/>`def run_tasks_parallel(tasks: list[str], max_workers: int | None = None, model: str | None = None, work_dir: str | None = None) -> list[str]`

- `tasks`: List of task description strings. Each string is passed as the `prompt_template` argument to :meth:`SorcarAgent.run`. Example:: [ "Summarize file A", "Summarize file B", ]
- `max_workers`: Maximum number of threads. `None` lets :class:`~concurrent.futures.ThreadPoolExecutor` pick a default (typically `min(32, cpu_count + 4)`).
- `model`: LLM model name for all parallel agents. `None` uses the default from persistence (same as :meth:`SorcarAgent.run`).
- `work_dir`: Working directory for all parallel agents. `None` uses the default (`artifact_dir/kiss_workdir`).
- **Returns:** List of YAML result strings in the **same order** as *tasks*. Each string contains `success` and `summary` keys. If a task raises an unhandled exception the corresponding entry is a YAML string with `success: false` and the traceback in `summary`.

**`cli_ask_user_question`** — CLI callback for agent questions (prints and reads from stdin).<br/>`def cli_ask_user_question(question: str) -> str`

- `question`: The question to display to the user.
- **Returns:** The user's typed response text.

______________________________________________________________________

#### `kiss.agents.gepa` — *GEPA (Genetic-Pareto) prompt optimization package.*

```python
from kiss.agents.gepa import GEPA, GEPAPhase, GEPAProgress, PromptCandidate, create_progress_callback
```

##### `class GEPA` — GEPA (Genetic-Pareto) prompt optimizer.

**Constructor:** `GEPA(agent_wrapper: Callable[[str, dict[str, str]], tuple[str, list[Any]]], initial_prompt_template: str, evaluation_fn: Callable[[str], dict[str, float]] | None = None, max_generations: int = 10, population_size: int = 8, pareto_size: int = 4, mutation_rate: float = 0.5, reflection_model: str = 'gemini-3-flash-preview', dev_val_split: float | None = None, perfect_score: float = 1.0, use_merge: bool = True, max_merge_invocations: int = 5, merge_val_overlap_floor: int = 2, progress_callback: Callable[[GEPAProgress], None] | None = None, batched_agent_wrapper: Callable[[str, list[dict[str, str]]], list[tuple[str, list[Any]]]] | None = None)`

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

#### `kiss.agents.kiss_evolve` — *KISSEvolve: Evolutionary Algorithm Discovery using LLMs.*

```python
from kiss.agents.kiss_evolve import CodeVariant, KISSEvolve, SimpleRAG
```

##### `class CodeVariant` — Represents a code variant in the evolutionary population.

##### `class KISSEvolve` — KISSEvolve: Evolutionary algorithm discovery using LLMs.

**Constructor:** `KISSEvolve(code_agent_wrapper: Callable[..., str], initial_code: str, evaluation_fn: Callable[[str], dict[str, Any]], model_names: list[tuple[str, float]], extra_coding_instructions: str = '', population_size: int = 8, max_generations: int = 10, mutation_rate: float = 0.7, elite_size: int = 2, num_islands: int = 2, migration_frequency: int = 5, migration_size: int = 1, migration_topology: str = 'ring', enable_novelty_rejection: bool = False, novelty_threshold: float = 0.95, max_rejection_attempts: int = 5, novelty_rag_model: Model | None = None, parent_sampling_method: str = 'power_law', power_law_alpha: float = 1.0, performance_novelty_lambda: float = 1.0)`

- `code_agent_wrapper`: The code generation agent wrapper. Should accept keyword arguments: model_name (str), prompt_template (str), and arguments (dict[str, str]).

- `initial_code`: The initial code to evolve.

- `evaluation_fn`: Function that takes code string and returns dict with: - 'fitness': float (higher is better) - 'metrics': dict[str, float] (optional additional metrics) - 'artifacts': dict[str, Any] (optional execution artifacts) - 'error': str (optional error message if evaluation failed)

- `model_names`: List of tuples containing (model_name, probability). Probabilities will be normalized to sum to 1.0.

- `extra_coding_instructions`: Extra instructions to add to the code generation prompt.

- `population_size`: Number of variants to maintain in population.

- `max_generations`: Maximum number of evolutionary generations.

- `mutation_rate`: Probability of mutating a variant.

- `elite_size`: Number of best variants to preserve each generation.

- `num_islands`: Number of islands for island-based evolution.

- `migration_frequency`: Number of generations between migrations.

- `migration_size`: Number of individuals to migrate between islands.

- `migration_topology`: Migration topology ('ring', 'fully_connected', 'random').

- `enable_novelty_rejection`: Enable code novelty rejection sampling.

- `novelty_threshold`: Cosine similarity threshold for rejecting code (0.0-1.0, higher = more strict).

- `max_rejection_attempts`: Maximum number of rejection attempts before accepting a variant anyway.

- `novelty_rag_model`: Model to use for generating code embeddings. If None and novelty rejection is enabled, uses the first model from models list.

- `parent_sampling_method`: Parent sampling method ('tournament', 'power_law', or 'performance_novelty').

- `power_law_alpha`: Power-law sampling parameter (α) for rank-based sampling. Lower = more exploration, higher = more exploitation.

- `performance_novelty_lambda`: Performance-novelty sampling parameter (λ) controlling selection pressure.

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

- **get_collection_stats** — Get statistics about the collection.<br/>`get_collection_stats() -> dict[str, Any]`

  - **Returns:** Dictionary containing collection statistics.

______________________________________________________________________

### `kiss.docker` — *Docker wrapper module for the KISS agent framework.*

```python
from kiss.docker import DockerManager, DockerTools
```

______________________________________________________________________

#### `kiss.docker.docker_manager` — *Docker library for managing Docker containers and executing commands.*

##### `class DockerManager` — Manages Docker container lifecycle and command execution.

**Constructor:** `DockerManager(image_name: str, tag: str = 'latest', workdir: str = '/', mount_shared_volume: bool = True, ports: dict[int, int] | None = None) -> None`

- `image_name`: The name of the Docker image (e.g., 'ubuntu', 'python')

- `tag`: The tag/version of the image (default: 'latest')

- `workdir`: The working directory inside the container

- `mount_shared_volume`: Whether to mount a shared volume. Set to False for images that already have content in the workdir (e.g., SWE-bench).

- `ports`: Port mapping from container port to host port. Example: {8080: 8080} maps container port 8080 to host port 8080. Example: {80: 8000, 443: 8443} maps multiple ports.

- **open** — Pull and load a Docker image, then create and start a container.<br/>`open() -> None`

- **Bash** — Execute a bash command in the running Docker container.<br/>`Bash(command: str, description: str, timeout_seconds: int = 30) -> str`

  - `command`: The bash command to execute
  - `description`: A short description of the command in natural language
  - `timeout_seconds`: Maximum time to wait before treating the command as hung.
  - **Returns:** The output of the command, including stdout, stderr, and exit code

- **get_host_port** — Get the host port mapped to a container port.<br/>`get_host_port(container_port: int) -> int | None`

  - `container_port`: The container port to look up.
  - **Returns:** The host port mapped to the container port, or None if not mapped.

- **close** — Stop and remove the Docker container. Handles cleanup of both the container and any temporary directories created for shared volumes.<br/>`close() -> None`

______________________________________________________________________

#### `kiss.agents.sorcar.git_worktree` — *Git worktree operations and state.*

##### `class GitWorktree` — Immutable snapshot of a pending worktree task.

##### `class MergeResult(enum.Enum)` — Outcome of a merge operation.

##### `class ManualMergeResult` — Outcome of a manual (--no-commit) merge operation.

##### `class GitWorktreeOps` — Stateless helper class with all git worktree operations.

- **discover_repo** — Find the git repo root containing *path*.<br/>`discover_repo(path: Path) -> Path | None`

  - `path`: Directory to start searching from.
  - **Returns:** The repo root path, or `None` if *path* is not in a repo.

- **current_branch** — Return the current branch name, or `None` for detached HEAD.<br/>`current_branch(repo: Path) -> str | None`

  - `repo`: Git repo root path.
  - **Returns:** Branch name string, or `None` if HEAD is detached or empty.

- **create** — Create a new worktree with a new branch.<br/>`create(repo: Path, branch: str, wt_dir: Path) -> bool`

  - `repo`: Git repo root path.
  - `branch`: New branch name to create.
  - `wt_dir`: Directory for the new worktree.
  - **Returns:** True if worktree was created successfully, False otherwise.

- **remove** — Remove a worktree directory (best-effort, force).<br/>`remove(repo: Path, wt_dir: Path) -> None`

  - `repo`: Git repo root path.
  - `wt_dir`: Worktree directory to remove.

- **prune** — Prune stale worktree bookkeeping entries.<br/>`prune(repo: Path) -> None`

  - `repo`: Git repo root path.

- **stage_all** — Stage all changes in the worktree (`git add -A`).<br/>`stage_all(wt_dir: Path) -> None`

  - `wt_dir`: Worktree directory.

- **commit_all** — Stage all changes and commit in the worktree.<br/>`commit_all(wt_dir: Path, message: str) -> bool`

  - `wt_dir`: Worktree directory.
  - `message`: Commit message.
  - **Returns:** True if a commit was created, False if nothing to commit.

- **staged_diff** — Return the staged diff text for the worktree.<br/>`staged_diff(wt_dir: Path) -> str`

  - `wt_dir`: Worktree directory (must have staged changes).
  - **Returns:** The diff text, or empty string if no staged changes.

- **checkout** — Checkout a branch in the main worktree.<br/>`checkout(repo: Path, branch: str) -> bool`

  - `repo`: Git repo root path.
  - `branch`: Branch name to checkout.
  - **Returns:** True if checkout succeeded, False otherwise.

- **checkout_error** — Return the stderr from a failed checkout attempt.<br/>`checkout_error(repo: Path, branch: str) -> str`

  - `repo`: Git repo root path.
  - `branch`: Branch name that failed to checkout.
  - **Returns:** The stderr text from the failed checkout.

- **merge_branch** — Merge a branch into the current HEAD with `--no-edit`. On conflict, the merge is aborted to leave a clean worktree.<br/>`merge_branch(repo: Path, branch: str) -> MergeResult`

  - `repo`: Git repo root path.
  - `branch`: Branch to merge.
  - **Returns:** :attr:`MergeResult.SUCCESS` or :attr:`MergeResult.CONFLICT`.

- **squash_merge_branch** — Squash-merge a branch and commit the result. Uses `git merge --squash` to apply all changes from *branch*, then commits them. The commit message is taken from git's auto-generated `SQUASH_MSG`. On conflict, resets to a clean state with `git reset --hard`.<br/>`squash_merge_branch(repo: Path, branch: str) -> MergeResult`

  - `repo`: Git repo root path.
  - `branch`: Branch to squash-merge.
  - **Returns:** :attr:`MergeResult.SUCCESS` or :attr:`MergeResult.CONFLICT`.

- **manual_merge_branch** — Merge with `--no-commit --no-ff` for interactive review. On success (no conflicts), unstages changes via `git reset HEAD` so the user can selectively stage hunks.<br/>`manual_merge_branch(repo: Path, branch: str) -> ManualMergeResult`

  - `repo`: Git repo root path.
  - `branch`: Branch to merge.
  - **Returns:** A :class:`ManualMergeResult` with status and conflict info.

- **delete_branch** — Delete a branch and its git config section (best-effort). Tries `-d` first (safe delete), falls back to `-D` (force). Also removes the `branch.<name>.*` config section.<br/>`delete_branch(repo: Path, branch: str) -> None`

  - `repo`: Git repo root path.
  - `branch`: Branch name to delete.

- **branch_exists** — Check if a branch exists.<br/>`branch_exists(repo: Path, branch: str) -> bool`

  - `repo`: Git repo root path.
  - `branch`: Branch name to check.
  - **Returns:** True if the branch exists.

- **ensure_excluded** — Add `.kiss-worktrees/` to local git exclude (not .gitignore). Uses `<git_common_dir>/info/exclude` so the agent never modifies any tracked file in the user's repo.<br/>`ensure_excluded(repo: Path) -> None`

  - `repo`: Git repo root path.

- **find_pending_branch** — Find the latest `kiss/wt-*` branch matching a prefix.<br/>`find_pending_branch(repo: Path, prefix: str) -> str | None`

  - `repo`: Git repo root path.
  - `prefix`: Branch name prefix (e.g. `kiss/wt-<chat_id[:12]>-`).
  - **Returns:** The lexicographically last matching branch, or `None`.

- **load_original_branch** — Load the original branch from git config.<br/>`load_original_branch(repo: Path, branch: str) -> str | None`

  - `repo`: Git repo root path.
  - `branch`: The worktree branch name.
  - **Returns:** The original branch name, or `None` if not stored.

- **save_original_branch** — Store the original branch in git config.<br/>`save_original_branch(repo: Path, branch: str, original: str) -> bool`

  - `repo`: Git repo root path.
  - `branch`: The worktree branch name.
  - `original`: The original branch to store.
  - **Returns:** True if config was saved successfully, False otherwise.

- **cleanup_partial** — Remove a partially-created worktree and branch (best-effort).<br/>`cleanup_partial(repo: Path, branch: str, wt_dir: Path) -> None`

  - `repo`: Git repo root path.
  - `branch`: The branch name to delete.
  - `wt_dir`: The worktree directory to remove.

- **cleanup_orphans** — Scan for orphaned `kiss/wt-*` branches and worktrees.<br/>`cleanup_orphans(repo: Path) -> str`

  - `repo`: Root of the git repository to scan.
  - **Returns:** Summary of findings and any cleanup actions taken.

______________________________________________________________________

#### `kiss.agents.sorcar.stateful_sorcar_agent` — *Stateful Sorcar agent with chat-session persistence.*

##### `class StatefulSorcarAgent(SorcarAgent)` — SorcarAgent with chat-session state management.

**Constructor:** `StatefulSorcarAgent(name: str) -> None`

- **chat_id** — Return the current chat session ID (0 means new session).<br/>`chat_id() -> int` *(property)*

- **new_chat** — Reset to a new chat session (equivalent to VS Code 'Clear').<br/>`new_chat() -> None`

- **resume_chat** — Resume a previous chat session by looking up the task's chat_id. If the task has an associated `chat_id` in history, subsequent `run()` calls will continue that session.<br/>`resume_chat(task: str) -> None`

  - `task`: The task description string to look up.

- **resume_chat_by_id** — Resume a chat session using a stable chat identifier.<br/>`resume_chat_by_id(chat_id: int) -> None`

  - `chat_id`: Integer chat session identifier to resume.

- **build_chat_prompt** — Load chat context and augment prompt with previous tasks/results.<br/>`build_chat_prompt(prompt: str) -> str`

  - `prompt`: The original task prompt.
  - **Returns:** The augmented prompt with chat history prepended, or the original prompt if no prior context exists.

- **run** — Run the agent with chat-session context management. Loads prior chat context, persists the new task, augments the prompt with previous tasks/results, runs the underlying agent, and saves the result back to history. Only the result summary is persisted here. Callers that record chat events (e.g. the VS Code server) should additionally call :func:`~kiss.agents.sorcar.persistence._set_latest_chat_events` to persist the full event stream.<br/>`run(prompt_template: str = '', **kwargs: Any) -> str`

  - `prompt_template`: The task prompt.
  - `**kwargs`: All other arguments forwarded to `SorcarAgent.run()`.
  - **Returns:** YAML string with 'success' and 'summary' keys.

______________________________________________________________________

#### `kiss.agents.sorcar.worktree_sorcar_agent` — *Worktree-based agent that runs each task on an isolated git branch.*

##### `class WorktreeSorcarAgent(StatefulSorcarAgent)` — SorcarAgent that isolates every task in a git worktree.

**Constructor:** `WorktreeSorcarAgent(name: str) -> None`

- **run** — Run a task on an isolated git worktree branch. Creates a new worktree and branch, redirects `work_dir` into the worktree, and delegates to `StatefulSorcarAgent.run()`. If a branch from this chat session is already pending, returns an error asking the user to merge or discard first. Falls back to direct execution (no worktree) when: - `work_dir` is not inside a git repo - The repo has no commits - HEAD is detached (no merge target) - Any git command fails during setup<br/>`run(prompt_template: str = '', **kwargs: Any) -> str`

  - `prompt_template`: The task prompt.
  - `**kwargs`: All other arguments forwarded to `StatefulSorcarAgent.run()`.
  - **Returns:** YAML string with 'success' and 'summary' keys.

- **merge** — Merge the task branch into the original branch. Every step is idempotent — safe to re-run after a crash. Auto-commits any uncommitted changes in the worktree before merging.<br/>`merge() -> str`

  - **Returns:** Success message, or error message if merge fails.

- **discard** — Throw away the task branch and worktree, checkout original. Every step is idempotent — safe to call multiple times.<br/>`discard() -> str`

  - **Returns:** Confirmation message.

- **do_nothing** — Leave the worktree branch as-is without any git operation. Clears the pending worktree state so the user regains control and can start new tasks. The branch and any committed work remain in git for the user to merge or discard manually later.<br/>`do_nothing() -> str`

  - **Returns:** Informational message with the branch name.

- **merge_instructions** — Return human-readable merge/discard/do-nothing instructions.<br/>`merge_instructions() -> str`

  - **Returns:** Multi-line string with merge, discard, and do-nothing instructions.

- **cleanup** — Scan for orphaned `kiss/wt-*` branches and worktrees.<br/>`cleanup(repo_root: Path | str) -> str`

  - `repo_root`: Root of the git repository to scan.
  - **Returns:** Summary of findings and any cleanup actions taken.

______________________________________________________________________

#### `kiss.agents.vscode` — *KISS Sorcar VS Code Extension backend.*

______________________________________________________________________

#### `kiss.agents.vscode.helpers` — *Helper utilities for Sorcar agent backends (autocomplete, model info, file ranking).*

**`clean_llm_output`** — Strip whitespace and surrounding quotes from LLM output.<br/>`def clean_llm_output(text: str) -> str`

**`clip_autocomplete_suggestion`** — Return the autocomplete continuation, stripped of the query prefix. Removes the query prefix if the LLM echoed it, strips surrounding whitespace, and stops at newlines.<br/>`def clip_autocomplete_suggestion(query: str, suggestion: str) -> str`

**`model_vendor`** — Return (vendor_display_name, sort_order) for a model name.<br/>`def model_vendor(name: str) -> tuple[str, int]`

- `name`: The model name string.
- **Returns:** Tuple of (display name, numeric sort order).

**`fast_model_for`** — Return a cheap/fast model based on which API keys are available. Priority: Anthropic/OpenRouter/Together → Gemini → OpenAI.<br/>`def fast_model_for() -> str`

- **Returns:** A fast model name for the first available provider.

**`generate_followup_text`** — Generate a follow-up task suggestion via LLM.<br/>`def generate_followup_text(task: str, result: str, model: str) -> str`

- `task`: The completed task description.
- `result`: The task result summary.
- `model`: The model to use for generation.
- **Returns:** Suggestion text, or empty string on failure.

**`rank_file_suggestions`** — Rank and filter file paths by query match, recency, and usage.<br/>`def rank_file_suggestions(file_cache: list[str], query: str, usage: dict[str, int], limit: int = 20) -> list[dict[str, str]]`

- `file_cache`: List of file paths to search.
- `query`: Case-sensitive substring to match against paths.
- `usage`: File usage counts keyed by path (insertion order encodes recency, last key = most recently used).
- `limit`: Maximum number of results to return.
- **Returns:** Sorted list of dicts with `type` (`"frequent"` or `"file"`) and `text` keys.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss` — *Top-level Kiss module for the project.*

```python
from kiss.agents.vscode.kiss_project.src.kiss import __version__
```

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents` — *KISS agents package with pre-built agent implementations.*

```python
from kiss.agents.vscode.kiss_project.src.kiss.agents import prompt_refiner_agent, get_run_simple_coding_agent, run_bash_task_in_sandboxed_ubuntu_latest
```

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.gepa` — *GEPA (Genetic-Pareto) prompt optimization package.*

```python
from kiss.agents.vscode.kiss_project.src.kiss.agents.gepa import GEPA, GEPAPhase, GEPAProgress, PromptCandidate, create_progress_callback
```

##### `class GEPA` — GEPA (Genetic-Pareto) prompt optimizer.

**Constructor:** `GEPA(agent_wrapper: Callable[[str, dict[str, str]], tuple[str, list[Any]]], initial_prompt_template: str, evaluation_fn: Callable[[str], dict[str, float]] | None = None, max_generations: int = 10, population_size: int = 8, pareto_size: int = 4, mutation_rate: float = 0.5, reflection_model: str = 'gemini-3-flash-preview', dev_val_split: float | None = None, perfect_score: float = 1.0, use_merge: bool = True, max_merge_invocations: int = 5, merge_val_overlap_floor: int = 2, progress_callback: Callable[[GEPAProgress], None] | None = None, batched_agent_wrapper: Callable[[str, list[dict[str, str]]], list[tuple[str, list[Any]]]] | None = None)`

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

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.gepa.gepa` — *GEPA (Genetic-Pareto): Reflective Prompt Evolution for Compound AI Systems.*

##### `class GEPAPhase(Enum)` — Enum representing the current phase of GEPA optimization.

##### `class GEPAProgress` — Progress information for GEPA optimization callbacks.

##### `class PromptCandidate` — Represents a prompt candidate with its performance metrics.

##### `class GEPA` — GEPA (Genetic-Pareto) prompt optimizer.

**Constructor:** `GEPA(agent_wrapper: Callable[[str, dict[str, str]], tuple[str, list[Any]]], initial_prompt_template: str, evaluation_fn: Callable[[str], dict[str, float]] | None = None, max_generations: int = 10, population_size: int = 8, pareto_size: int = 4, mutation_rate: float = 0.5, reflection_model: str = 'gemini-3-flash-preview', dev_val_split: float | None = None, perfect_score: float = 1.0, use_merge: bool = True, max_merge_invocations: int = 5, merge_val_overlap_floor: int = 2, progress_callback: Callable[[GEPAProgress], None] | None = None, batched_agent_wrapper: Callable[[str, list[dict[str, str]]], list[tuple[str, list[Any]]]] | None = None)`

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

**`create_progress_callback`** — Create a standard progress callback for GEPA optimization.<br/>`def create_progress_callback(verbose: bool = False) -> 'Callable[[GEPAProgress], None]'`

- `verbose`: If True, prints all phases. If False, only prints val evaluation completion messages (when a candidate has been fully evaluated).
- **Returns:** A callback function that prints progress updates during optimization.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.kiss` — *Useful agents for the KISS Agent Framework.*

**`prompt_refiner_agent`** — Refines the prompt template based on the agent's trajectory summary.<br/>`def prompt_refiner_agent(original_prompt_template: str, previous_prompt_template: str, agent_trajectory_summary: str, model_name: str) -> str`

- `original_prompt_template`: The original prompt template.
- `previous_prompt_template`: The previous version of the prompt template that led to the given trajectory.
- `agent_trajectory_summary`: The agent's trajectory summary as a string.
- `model_name`: The name of the model to use for the agent.
- **Returns:** str: The refined prompt template.

**`run_bash_task_in_sandboxed_ubuntu_latest`** — Run a bash task in a sandboxed Ubuntu latest container.<br/>`def run_bash_task_in_sandboxed_ubuntu_latest(task: str, model_name: str) -> str`

- `task`: The task to run.
- `model_name`: The name of the model to use for the agent.
- **Returns:** str: The result of the task.

**`get_run_simple_coding_agent`** — Return a function that runs a simple coding agent with a test function.<br/>`def get_run_simple_coding_agent(test_fn: Callable[[str], bool]) -> Callable[..., str]`

- `test_fn`: The test function to use for the agent.
- **Returns:** Callable\[..., str\]: A function that runs a simple coding agent with a test function. Accepts keyword arguments: model_name (str), prompt_template (str), and arguments (dict[str, str]).

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.kiss_evolve` — *KISSEvolve: Evolutionary Algorithm Discovery using LLMs.*

```python
from kiss.agents.vscode.kiss_project.src.kiss.agents.kiss_evolve import CodeVariant, KISSEvolve, SimpleRAG
```

##### `class CodeVariant` — Represents a code variant in the evolutionary population.

##### `class KISSEvolve` — KISSEvolve: Evolutionary algorithm discovery using LLMs.

**Constructor:** `KISSEvolve(code_agent_wrapper: Callable[..., str], initial_code: str, evaluation_fn: Callable[[str], dict[str, Any]], model_names: list[tuple[str, float]], extra_coding_instructions: str = '', population_size: int = 8, max_generations: int = 10, mutation_rate: float = 0.7, elite_size: int = 2, num_islands: int = 2, migration_frequency: int = 5, migration_size: int = 1, migration_topology: str = 'ring', enable_novelty_rejection: bool = False, novelty_threshold: float = 0.95, max_rejection_attempts: int = 5, novelty_rag_model: Model | None = None, parent_sampling_method: str = 'power_law', power_law_alpha: float = 1.0, performance_novelty_lambda: float = 1.0)`

- `code_agent_wrapper`: The code generation agent wrapper. Should accept keyword arguments: model_name (str), prompt_template (str), and arguments (dict[str, str]).

- `initial_code`: The initial code to evolve.

- `evaluation_fn`: Function that takes code string and returns dict with: - 'fitness': float (higher is better) - 'metrics': dict[str, float] (optional additional metrics) - 'artifacts': dict[str, Any] (optional execution artifacts) - 'error': str (optional error message if evaluation failed)

- `model_names`: List of tuples containing (model_name, probability). Probabilities will be normalized to sum to 1.0.

- `extra_coding_instructions`: Extra instructions to add to the code generation prompt.

- `population_size`: Number of variants to maintain in population.

- `max_generations`: Maximum number of evolutionary generations.

- `mutation_rate`: Probability of mutating a variant.

- `elite_size`: Number of best variants to preserve each generation.

- `num_islands`: Number of islands for island-based evolution.

- `migration_frequency`: Number of generations between migrations.

- `migration_size`: Number of individuals to migrate between islands.

- `migration_topology`: Migration topology ('ring', 'fully_connected', 'random').

- `enable_novelty_rejection`: Enable code novelty rejection sampling.

- `novelty_threshold`: Cosine similarity threshold for rejecting code (0.0-1.0, higher = more strict).

- `max_rejection_attempts`: Maximum number of rejection attempts before accepting a variant anyway.

- `novelty_rag_model`: Model to use for generating code embeddings. If None and novelty rejection is enabled, uses the first model from models list.

- `parent_sampling_method`: Parent sampling method ('tournament', 'power_law', or 'performance_novelty').

- `power_law_alpha`: Power-law sampling parameter (α) for rank-based sampling. Lower = more exploration, higher = more exploitation.

- `performance_novelty_lambda`: Performance-novelty sampling parameter (λ) controlling selection pressure.

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

- **get_collection_stats** — Get statistics about the collection.<br/>`get_collection_stats() -> dict[str, Any]`

  - **Returns:** Dictionary containing collection statistics.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.kiss_evolve.kiss_evolve` — *KISSEvolve: Evolutionary Algorithm Discovery using LLMs.*

##### `class CodeVariant` — Represents a code variant in the evolutionary population.

##### `class KISSEvolve` — KISSEvolve: Evolutionary algorithm discovery using LLMs.

**Constructor:** `KISSEvolve(code_agent_wrapper: Callable[..., str], initial_code: str, evaluation_fn: Callable[[str], dict[str, Any]], model_names: list[tuple[str, float]], extra_coding_instructions: str = '', population_size: int = 8, max_generations: int = 10, mutation_rate: float = 0.7, elite_size: int = 2, num_islands: int = 2, migration_frequency: int = 5, migration_size: int = 1, migration_topology: str = 'ring', enable_novelty_rejection: bool = False, novelty_threshold: float = 0.95, max_rejection_attempts: int = 5, novelty_rag_model: Model | None = None, parent_sampling_method: str = 'power_law', power_law_alpha: float = 1.0, performance_novelty_lambda: float = 1.0)`

- `code_agent_wrapper`: The code generation agent wrapper. Should accept keyword arguments: model_name (str), prompt_template (str), and arguments (dict[str, str]).

- `initial_code`: The initial code to evolve.

- `evaluation_fn`: Function that takes code string and returns dict with: - 'fitness': float (higher is better) - 'metrics': dict[str, float] (optional additional metrics) - 'artifacts': dict[str, Any] (optional execution artifacts) - 'error': str (optional error message if evaluation failed)

- `model_names`: List of tuples containing (model_name, probability). Probabilities will be normalized to sum to 1.0.

- `extra_coding_instructions`: Extra instructions to add to the code generation prompt.

- `population_size`: Number of variants to maintain in population.

- `max_generations`: Maximum number of evolutionary generations.

- `mutation_rate`: Probability of mutating a variant.

- `elite_size`: Number of best variants to preserve each generation.

- `num_islands`: Number of islands for island-based evolution.

- `migration_frequency`: Number of generations between migrations.

- `migration_size`: Number of individuals to migrate between islands.

- `migration_topology`: Migration topology ('ring', 'fully_connected', 'random').

- `enable_novelty_rejection`: Enable code novelty rejection sampling.

- `novelty_threshold`: Cosine similarity threshold for rejecting code (0.0-1.0, higher = more strict).

- `max_rejection_attempts`: Maximum number of rejection attempts before accepting a variant anyway.

- `novelty_rag_model`: Model to use for generating code embeddings. If None and novelty rejection is enabled, uses the first model from models list.

- `parent_sampling_method`: Parent sampling method ('tournament', 'power_law', or 'performance_novelty').

- `power_law_alpha`: Power-law sampling parameter (α) for rank-based sampling. Lower = more exploration, higher = more exploitation.

- `performance_novelty_lambda`: Performance-novelty sampling parameter (λ) controlling selection pressure.

- **evolve** — Run the evolutionary algorithm.<br/>`evolve() -> CodeVariant`

  - **Returns:** CodeVariant: The best code variant found during evolution.

- **get_best_variant** — Get the best variant from the current population or islands.<br/>`get_best_variant() -> CodeVariant`

  - **Returns:** The CodeVariant with the highest fitness from the current population or all islands. Returns a default variant with initial_code if no population exists.

- **get_population_stats** — Get statistics about the current population.<br/>`get_population_stats() -> dict[str, Any]`

  - **Returns:** Dictionary containing: - size: Total population size - avg_fitness: Average fitness across all variants - best_fitness: Maximum fitness value - worst_fitness: Minimum fitness value

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.kiss_evolve.simple_rag` — *Simple and elegant RAG system for document storage and retrieval using in-memory vector store.*

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

- **get_collection_stats** — Get statistics about the collection.<br/>`get_collection_stats() -> dict[str, Any]`

  - **Returns:** Dictionary containing collection statistics.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar` — *Sorcar agent with coding tools and browser automation.*

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar.git_worktree` — *Git worktree operations and state.*

##### `class GitWorktree` — Immutable snapshot of a pending worktree task.

##### `class MergeResult(enum.Enum)` — Outcome of a merge operation.

##### `class ManualMergeResult` — Outcome of a manual (--no-commit) merge operation.

##### `class GitWorktreeOps` — Stateless helper class with all git worktree operations.

- **discover_repo** — Find the git repo root containing *path*.<br/>`discover_repo(path: Path) -> Path | None`

  - `path`: Directory to start searching from.
  - **Returns:** The repo root path, or `None` if *path* is not in a repo.

- **current_branch** — Return the current branch name, or `None` for detached HEAD.<br/>`current_branch(repo: Path) -> str | None`

  - `repo`: Git repo root path.
  - **Returns:** Branch name string, or `None` if HEAD is detached or empty.

- **create** — Create a new worktree with a new branch.<br/>`create(repo: Path, branch: str, wt_dir: Path) -> bool`

  - `repo`: Git repo root path.
  - `branch`: New branch name to create.
  - `wt_dir`: Directory for the new worktree.
  - **Returns:** True if worktree was created successfully, False otherwise.

- **remove** — Remove a worktree directory (best-effort, force).<br/>`remove(repo: Path, wt_dir: Path) -> None`

  - `repo`: Git repo root path.
  - `wt_dir`: Worktree directory to remove.

- **prune** — Prune stale worktree bookkeeping entries.<br/>`prune(repo: Path) -> None`

  - `repo`: Git repo root path.

- **stage_all** — Stage all changes in the worktree (`git add -A`).<br/>`stage_all(wt_dir: Path) -> None`

  - `wt_dir`: Worktree directory.

- **commit_all** — Stage all changes and commit in the worktree.<br/>`commit_all(wt_dir: Path, message: str) -> bool`

  - `wt_dir`: Worktree directory.
  - `message`: Commit message.
  - **Returns:** True if a commit was created, False if nothing to commit.

- **staged_diff** — Return the staged diff text for the worktree.<br/>`staged_diff(wt_dir: Path) -> str`

  - `wt_dir`: Worktree directory (must have staged changes).
  - **Returns:** The diff text, or empty string if no staged changes.

- **checkout** — Checkout a branch in the main worktree.<br/>`checkout(repo: Path, branch: str) -> bool`

  - `repo`: Git repo root path.
  - `branch`: Branch name to checkout.
  - **Returns:** True if checkout succeeded, False otherwise.

- **checkout_error** — Return the stderr from a failed checkout attempt.<br/>`checkout_error(repo: Path, branch: str) -> str`

  - `repo`: Git repo root path.
  - `branch`: Branch name that failed to checkout.
  - **Returns:** The stderr text from the failed checkout.

- **merge_branch** — Merge a branch into the current HEAD with `--no-edit`. On conflict, the merge is aborted to leave a clean worktree.<br/>`merge_branch(repo: Path, branch: str) -> MergeResult`

  - `repo`: Git repo root path.
  - `branch`: Branch to merge.
  - **Returns:** :attr:`MergeResult.SUCCESS` or :attr:`MergeResult.CONFLICT`.

- **squash_merge_branch** — Squash-merge a branch and commit the result. Uses `git merge --squash` to apply all changes from *branch*, then commits them. The commit message is taken from git's auto-generated `SQUASH_MSG`. On conflict, resets to a clean state with `git reset --hard`.<br/>`squash_merge_branch(repo: Path, branch: str) -> MergeResult`

  - `repo`: Git repo root path.
  - `branch`: Branch to squash-merge.
  - **Returns:** :attr:`MergeResult.SUCCESS` or :attr:`MergeResult.CONFLICT`.

- **manual_merge_branch** — Merge with `--no-commit --no-ff` for interactive review. On success (no conflicts), unstages changes via `git reset HEAD` so the user can selectively stage hunks.<br/>`manual_merge_branch(repo: Path, branch: str) -> ManualMergeResult`

  - `repo`: Git repo root path.
  - `branch`: Branch to merge.
  - **Returns:** A :class:`ManualMergeResult` with status and conflict info.

- **delete_branch** — Delete a branch and its git config section (best-effort). Tries `-d` first (safe delete), falls back to `-D` (force). Also removes the `branch.<name>.*` config section.<br/>`delete_branch(repo: Path, branch: str) -> None`

  - `repo`: Git repo root path.
  - `branch`: Branch name to delete.

- **branch_exists** — Check if a branch exists.<br/>`branch_exists(repo: Path, branch: str) -> bool`

  - `repo`: Git repo root path.
  - `branch`: Branch name to check.
  - **Returns:** True if the branch exists.

- **ensure_excluded** — Add `.kiss-worktrees/` to local git exclude (not .gitignore). Uses `<git_common_dir>/info/exclude` so the agent never modifies any tracked file in the user's repo.<br/>`ensure_excluded(repo: Path) -> None`

  - `repo`: Git repo root path.

- **find_pending_branch** — Find the latest `kiss/wt-*` branch matching a prefix.<br/>`find_pending_branch(repo: Path, prefix: str) -> str | None`

  - `repo`: Git repo root path.
  - `prefix`: Branch name prefix (e.g. `kiss/wt-<chat_id[:12]>-`).
  - **Returns:** The lexicographically last matching branch, or `None`.

- **load_original_branch** — Load the original branch from git config.<br/>`load_original_branch(repo: Path, branch: str) -> str | None`

  - `repo`: Git repo root path.
  - `branch`: The worktree branch name.
  - **Returns:** The original branch name, or `None` if not stored.

- **save_original_branch** — Store the original branch in git config.<br/>`save_original_branch(repo: Path, branch: str, original: str) -> bool`

  - `repo`: Git repo root path.
  - `branch`: The worktree branch name.
  - `original`: The original branch to store.
  - **Returns:** True if config was saved successfully, False otherwise.

- **cleanup_partial** — Remove a partially-created worktree and branch (best-effort).<br/>`cleanup_partial(repo: Path, branch: str, wt_dir: Path) -> None`

  - `repo`: Git repo root path.
  - `branch`: The branch name to delete.
  - `wt_dir`: The worktree directory to remove.

- **cleanup_orphans** — Scan for orphaned `kiss/wt-*` branches and worktrees.<br/>`cleanup_orphans(repo: Path) -> str`

  - `repo`: Root of the git repository to scan.
  - **Returns:** Summary of findings and any cleanup actions taken.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar.sorcar_agent` — *Sorcar agent with both coding tools and browser automation.*

##### `class SorcarAgent(RelentlessAgent)` — Agent with both coding tools and browser automation for web + code tasks.

**Constructor:** `SorcarAgent(name: str) -> None`

- **perform_task** — Execute the task, building docker-aware tools after docker_manager is set.<br/>`perform_task(tools: list, attachments: list | None = None) -> str`

  - `tools`: Extra tools passed by the caller (from run(tools=...)).
  - `attachments`: Optional file attachments for the initial prompt.
  - **Returns:** YAML string with 'success' and 'summary' keys.

- **run** — Run the assistant agent with coding tools and browser automation.<br/>`run(model_name: str | None = None, prompt_template: str = '', arguments: dict[str, str] | None = None, system_prompt: str | None = None, tools: list[Callable[..., Any]] | None = None, max_steps: int | None = None, max_budget: float | None = None, model_config: dict[str, Any] | None = None, work_dir: str | None = None, printer: Printer | None = None, max_sub_sessions: int | None = None, docker_image: str | None = None, web_tools: bool = True, is_parallel: bool = False, verbose: bool | None = None, current_editor_file: str | None = None, attachments: list[Attachment] | None = None, ask_user_question_callback: Callable[[str], str] | None = None) -> str`

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
  - `web_tools`: Whether to include browser/web tools. Defaults to True. Set to False for terminal-only environments.
  - `is_parallel`: Whether to include the run_parallel tool. Defaults to False. When True, the agent can spawn parallel sub-agents for independent tasks.
  - `verbose`: Whether to print output to console. Defaults to config verbose setting.
  - `current_editor_file`: Path to the currently active editor file, appended to prompt.
  - `attachments`: Optional file attachments (images, PDFs) for the initial prompt.
  - `ask_user_question_callback`: Optional callback used by the ask_user_question tool to collect a text response from the user.
  - **Returns:** YAML string with 'success' and 'summary' keys.

**`run_tasks_parallel`** — Execute multiple SorcarAgent tasks concurrently using threads. Each task gets its own `SorcarAgent` instance and runs in a separate thread via :class:`~concurrent.futures.ThreadPoolExecutor`. This is ideal for I/O-bound workloads (LLM API calls, network requests) where the GIL is released during I/O waits.<br/>`def run_tasks_parallel(tasks: list[str], max_workers: int | None = None, model: str | None = None, work_dir: str | None = None) -> list[str]`

- `tasks`: List of task description strings. Each string is passed as the `prompt_template` argument to :meth:`SorcarAgent.run`. Example:: [ "Summarize file A", "Summarize file B", ]
- `max_workers`: Maximum number of threads. `None` lets :class:`~concurrent.futures.ThreadPoolExecutor` pick a default (typically `min(32, cpu_count + 4)`).
- `model`: LLM model name for all parallel agents. `None` uses the default from persistence (same as :meth:`SorcarAgent.run`).
- `work_dir`: Working directory for all parallel agents. `None` uses the default (`artifact_dir/kiss_workdir`).
- **Returns:** List of YAML result strings in the **same order** as *tasks*. Each string contains `success` and `summary` keys. If a task raises an unhandled exception the corresponding entry is a YAML string with `success: false` and the traceback in `summary`.

**`cli_ask_user_question`** — CLI callback for agent questions (prints and reads from stdin).<br/>`def cli_ask_user_question(question: str) -> str`

- `question`: The question to display to the user.
- **Returns:** The user's typed response text.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar.stateful_sorcar_agent` — *Stateful Sorcar agent with chat-session persistence.*

##### `class StatefulSorcarAgent(SorcarAgent)` — SorcarAgent with chat-session state management.

**Constructor:** `StatefulSorcarAgent(name: str) -> None`

- **chat_id** — Return the current chat session ID.<br/>`chat_id() -> str` *(property)*

- **new_chat** — Reset to a new chat session (equivalent to VS Code 'Clear').<br/>`new_chat() -> None`

- **resume_chat** — Resume a previous chat session by looking up the task's chat_id. If the task has an associated `chat_id` in history, subsequent `run()` calls will continue that session.<br/>`resume_chat(task: str) -> None`

  - `task`: The task description string to look up.

- **resume_chat_by_id** — Resume a chat session using a stable chat identifier.<br/>`resume_chat_by_id(chat_id: str) -> None`

  - `chat_id`: Persisted chat session identifier to resume.

- **build_chat_prompt** — Load chat context and augment prompt with previous tasks/results.<br/>`build_chat_prompt(prompt: str) -> str`

  - `prompt`: The original task prompt.
  - **Returns:** The augmented prompt with chat history prepended, or the original prompt if no prior context exists.

- **run** — Run the agent with chat-session context management. Loads prior chat context, persists the new task, augments the prompt with previous tasks/results, runs the underlying agent, and saves the result back to history. Only the result summary is persisted here. Callers that record chat events (e.g. the VS Code server) should additionally call :func:`~kiss.agents.sorcar.persistence._set_latest_chat_events` to persist the full event stream.<br/>`run(prompt_template: str = '', **kwargs: Any) -> str`

  - `prompt_template`: The task prompt.
  - `**kwargs`: All other arguments forwarded to `SorcarAgent.run()`.
  - **Returns:** YAML string with 'success' and 'summary' keys.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar.useful_tools` — *Useful tools for agents: file editing and bash execution.*

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

- **Bash** — Runs a bash command and returns its output.<br/>`Bash(command: str, description: str, timeout_seconds: float = 300, max_output_chars: int = 50000) -> str`

  - `command`: The bash command to run.
  - `description`: A brief description of the command.
  - `timeout_seconds`: Timeout in seconds for the command.
  - `max_output_chars`: Maximum characters in output before truncation.
  - **Returns:** The output of the command.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar.web_use_tool` — *Browser automation tool for LLM agents using Playwright.*

##### `class WebUseTool` — Browser automation tool using non-headless Playwright Chromium.

**Constructor:** `WebUseTool(viewport: tuple[int, int] = (1280, 900), user_data_dir: str | None = _DEFAULT_USER_DATA_DIR, headless: bool = False, **_kwargs: Any) -> None`

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

- **get_tools** — Return callable web tools for registration with an agent.<br/>`get_tools() -> list[Callable[..., str]]`

  - **Returns:** List of callables: go_to_url, click, type_text, press_key, scroll, screenshot, get_page_content. Does not include close.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.sorcar.worktree_sorcar_agent` — *Worktree-based agent that runs each task on an isolated git branch.*

##### `class WorktreeSorcarAgent(StatefulSorcarAgent)` — SorcarAgent that isolates every task in a git worktree.

**Constructor:** `WorktreeSorcarAgent(name: str) -> None`

- **run** — Run a task on an isolated git worktree branch. Creates a new worktree and branch, redirects `work_dir` into the worktree, and delegates to `StatefulSorcarAgent.run()`. If a branch from this chat session is already pending, returns an error asking the user to merge or discard first. Falls back to direct execution (no worktree) when: - `work_dir` is not inside a git repo - The repo has no commits - HEAD is detached (no merge target) - Any git command fails during setup<br/>`run(prompt_template: str = '', **kwargs: Any) -> str`

  - `prompt_template`: The task prompt.
  - `**kwargs`: All other arguments forwarded to `StatefulSorcarAgent.run()`.
  - **Returns:** YAML string with 'success' and 'summary' keys.

- **merge** — Merge the task branch into the original branch. Every step is idempotent — safe to re-run after a crash. Auto-commits any uncommitted changes in the worktree before merging.<br/>`merge() -> str`

  - **Returns:** Success message, or error message if merge fails.

- **discard** — Throw away the task branch and worktree, checkout original. Every step is idempotent — safe to call multiple times.<br/>`discard() -> str`

  - **Returns:** Confirmation message.

- **do_nothing** — Leave the worktree branch as-is without any git operation. Clears the pending worktree state so the user regains control and can start new tasks. The branch and any committed work remain in git for the user to merge or discard manually later.<br/>`do_nothing() -> str`

  - **Returns:** Informational message with the branch name.

- **merge_instructions** — Return human-readable merge/discard/do-nothing instructions.<br/>`merge_instructions() -> str`

  - **Returns:** Multi-line string with merge, discard, and do-nothing instructions.

- **cleanup** — Scan for orphaned `kiss/wt-*` branches and worktrees.<br/>`cleanup(repo_root: Path | str) -> str`

  - `repo_root`: Root of the git repository to scan.
  - **Returns:** Summary of findings and any cleanup actions taken.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.vscode` — *KISS Sorcar VS Code Extension backend.*

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.vscode.browser_ui` — *Shared browser UI components for KISS agent viewers.*

##### `class BaseBrowserPrinter(StreamEventParser, Printer)`

**Constructor:** `BaseBrowserPrinter() -> None`

- **tokens_offset** — Per-thread token offset for usage_info events.<br/>`tokens_offset() -> int` *(property)*

- **tokens_offset**<br/>`tokens_offset(value: int) -> None`

- **budget_offset** — Per-thread budget offset for usage_info events.<br/>`budget_offset() -> float` *(property)*

- **budget_offset**<br/>`budget_offset(value: float) -> None`

- **steps_offset** — Per-thread steps offset for usage_info events.<br/>`steps_offset() -> int` *(property)*

- **steps_offset**<br/>`steps_offset(value: int) -> None`

- **reset** — Reset internal streaming and tool-parsing state for a new turn.<br/>`reset() -> None`

- **start_recording** — Start recording broadcast events. Uses an explicit *recording_id* to avoid thread-ID reuse corruption. Falls back to thread ident when no ID is given (backward compat). When *tab_id* is provided, only events whose `tabId` matches are recorded. Events without a `tabId` are still recorded to all active recordings.<br/>`start_recording(recording_id: int | None = None, tab_id: str | None = None) -> None`

  - `recording_id`: Unique identifier for this recording session.
  - `tab_id`: Optional tab owner — restricts which events are recorded.

- **stop_recording** — Stop recording and return its display events.<br/>`stop_recording(recording_id: int | None = None) -> list[dict[str, Any]]`

  - `recording_id`: The recording ID passed to start_recording.
  - **Returns:** List of display-relevant events with consecutive deltas merged.

- **peek_recording** — Return a snapshot of the current recording without stopping it. Used for periodic crash-recovery flushes: the caller can persist a snapshot of events to the database while recording continues.<br/>`peek_recording(recording_id: int) -> list[dict[str, Any]]`

  - `recording_id`: The recording ID passed to start_recording.
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
  - `type`: Content type (e.g. "text", "prompt", "stream_event", "tool_call", "tool_result", "result", "message").
  - `**kwargs`: Additional options such as tool_input, is_error, cost, total_tokens.
  - **Returns:** str: Extracted text from stream events, or empty string.

- **token_callback** — Broadcast a streamed token as an SSE delta event to browser clients.<br/>`token_callback(token: str) -> None`

  - `token`: The text token to broadcast.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.vscode.helpers` — *Helper utilities for Sorcar agent backends (autocomplete, model info, file ranking).*

**`clean_llm_output`** — Strip whitespace and surrounding quotes from LLM output.<br/>`def clean_llm_output(text: str) -> str`

**`clip_autocomplete_suggestion`** — Return the autocomplete continuation, stripped of the query prefix. Removes the query prefix if the LLM echoed it, strips surrounding whitespace, and stops at newlines.<br/>`def clip_autocomplete_suggestion(query: str, suggestion: str) -> str`

**`model_vendor`** — Return (vendor_display_name, sort_order) for a model name.<br/>`def model_vendor(name: str) -> tuple[str, int]`

- `name`: The model name string.
- **Returns:** Tuple of (display name, numeric sort order).

**`fast_model_for`** — Return a cheap/fast model based on which API keys are available. Priority: Anthropic/OpenRouter/Together → Gemini → OpenAI.<br/>`def fast_model_for() -> str`

- **Returns:** A fast model name for the first available provider.

**`generate_followup_text`** — Generate a follow-up task suggestion via LLM.<br/>`def generate_followup_text(task: str, result: str, model: str) -> str`

- `task`: The completed task description.
- `result`: The task result summary.
- `model`: The model to use for generation.
- **Returns:** Suggestion text, or empty string on failure.

**`rank_file_suggestions`** — Rank and filter file paths by query match, recency, and usage.<br/>`def rank_file_suggestions(file_cache: list[str], query: str, usage: dict[str, int], limit: int = 20) -> list[dict[str, str]]`

- `file_cache`: List of file paths to search.
- `query`: Case-sensitive substring to match against paths.
- `usage`: File usage counts keyed by path (insertion order encodes recency, last key = most recently used).
- `limit`: Maximum number of results to return.
- **Returns:** Sorted list of dicts with `type` (`"frequent"` or `"file"`) and `text` keys.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.agents.vscode.server` — *VS Code extension backend server for Sorcar agent.*

##### `class VSCodePrinter(BaseBrowserPrinter)` — Printer that outputs JSON events to stdout for VS Code extension.

**Constructor:** `VSCodePrinter() -> None`

- **broadcast** — Write event as a JSON line to stdout and record it. Injects `tabId` from thread-local storage when available so the frontend can route events to the correct chat tab.<br/>`broadcast(event: dict[str, Any]) -> None`
  - `event`: The event dictionary to emit.

##### `class VSCodeServer` — Backend server for VS Code extension.

**Constructor:** `VSCodeServer() -> None`

- **run** — Main loop: read commands from stdin, execute them.<br/>`run() -> None`
  **`parse_task_tags`** — Parse `<task>...</task>` tags from *text* and return individual tasks. When the input contains one or more `<task>` blocks with non-empty content, each block's content is returned as a separate list element. If no valid `<task>` blocks are found (or all are empty/whitespace), the original *text* is returned as a single-element list so that callers can always iterate without special-casing.<br/>`def parse_task_tags(text: str) -> list[str]`

- `text`: Input text potentially containing `<task>...</task>` tags.

- **Returns:** List of task strings. Always contains at least one element.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.benchmarks` — *KISS Sorcar benchmark harnesses for SWE-bench Pro, Terminal-Bench 2.0, and WebArena.*

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.benchmarks.generate_dashboard` — *Generate a benchmark results dashboard as a self-contained HTML page.*

**`generate_dashboard`** — Read benchmark results and produce an HTML dashboard.<br/>`def generate_dashboard(swebench_results_path: str | None = None, tbench_results_path: str | None = None, output_path: str = 'results/dashboard.html') -> None`

- `swebench_results_path`: Path to SWE-bench Pro eval results JSON.
- `tbench_results_path`: Path to Terminal-Bench results JSONL.
- `output_path`: Path for the output HTML file.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.benchmarks.swebench_pro` — *SWE-bench Pro benchmark harness for KISS Sorcar.*

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.benchmarks.swebench_pro.adapter` — *Convert SWE-bench Pro instances into sorcar task prompts.*

**`make_sorcar_task`** — Build a sorcar task prompt from a SWE-bench Pro instance. Returns a task string that tells sorcar: - The repo is already cloned at /app - The issue to fix (problem_statement) - To produce a git diff as the solution<br/>`def make_sorcar_task(instance: dict) -> str`

- `instance`: A SWE-bench Pro dataset row with at least 'problem_statement' and 'repo' fields.
- **Returns:** A formatted task prompt string for sorcar.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.benchmarks.swebench_pro.eval` — *Thin wrapper around the official SWE-bench Pro evaluation script.*

**`run_eval`** — Run the official SWE-bench Pro evaluation script.<br/>`def run_eval(patch_path: str, num_workers: int = 8, use_local_docker: bool = True, block_network: bool = False, docker_platform: str | None = None, redo: bool = False) -> None`

- `patch_path`: Path to the patches JSON file.
- `num_workers`: Number of parallel Docker workers.
- `use_local_docker`: Use local Docker instead of Modal.
- `block_network`: Block network access inside eval containers.
- `docker_platform`: Docker platform (e.g. "linux/amd64" for Apple Silicon).
- `redo`: Re-evaluate even if output exists.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.benchmarks.swebench_pro.run` — *Run sorcar on SWE-bench Pro instances and collect patches.*

**`run_instance`** — Run sorcar on a single SWE-bench Pro instance inside its Docker container. Steps: 1. docker run the instance's image (jefzda/sweap-images:\<dockerhub_tag>) 2. Inside the container, invoke sorcar with the task prompt 3. Capture the generated patch (git diff from /app) 4. Return {"instance_id": ..., "model_patch": ..., "prefix": model}<br/>`def run_instance(instance: dict, model: str, budget: float) -> dict`

- `instance`: A SWE-bench Pro dataset row.
- `model`: LLM model name (e.g. "claude-opus-4-6").
- `budget`: Max USD budget per instance.
- **Returns:** A dict with instance_id, model_patch, and prefix fields.

**`run_all`** — Iterate over all SWE-bench Pro public instances, generate patches.<br/>`def run_all(model: str, budget: float, max_instances: int | None = None, workers: int = 1) -> None`

- `model`: LLM model name (e.g. "claude-opus-4-6").
- `budget`: Max USD budget per instance.
- `max_instances`: Cap for quick testing (None = all 731).
- `workers`: Number of parallel workers (currently sequential).

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.benchmarks.terminal_bench` — *Terminal-Bench 2.0 benchmark harness for KISS Sorcar.*

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.benchmarks.terminal_bench.agent` — *Harbor agent adapter that delegates to KISS Sorcar.*

##### `class SorcarHarborAgent(BaseAgent)` — Harbor-compatible agent that uses KISS Sorcar as the backend.

- **name** — Return the agent's name.<br/>`name() -> str`

- **version** — Return the agent version string.<br/>`version() -> str | None`

- **setup** — Install sorcar inside the harbor container. Installs uv, then kiss-agent-framework as a uv tool (which manages its own Python), and writes a tbench-specific SYSTEM.md that replaces the generic IDE system prompt with terminal-bench instructions. Each step is run separately so failures are logged clearly and do not silently abort the chain.<br/>`async setup(environment: BaseEnvironment) -> None`

  - `environment`: The harbor execution environment.

- **run** — Run sorcar with the task instruction inside the container. After the first sorcar run, automatically runs the task's test.sh and retries once with failure output if tests don't pass.<br/>`async run(instruction: str, environment: BaseEnvironment, context: AgentContext) -> None`

  - `instruction`: Natural language task description from harbor.
  - `environment`: The harbor execution environment.
  - `context`: Agent context for storing token/cost metadata.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.benchmarks.terminal_bench.run` — *Run Terminal-Bench 2.0 with the sorcar harbor agent.*

**`is_docker_hub_authenticated`** — Check whether Docker Hub credentials are configured. Reads ~/.docker/config.json to find the credential store, then queries it via `docker-credential-<store> list`. Returns True if any credential is stored for `https://index.docker.io/`. Falls back to checking the `auths` dict when no credential store is configured.<br/>`def is_docker_hub_authenticated() -> bool`

**`pre_pull_images`** — Pre-pull all Docker images needed by a harbor dataset. Resolves the dataset's task definitions, extracts unique Docker image names, and pulls each one sequentially. Because Docker caches pulled images locally, subsequent `docker compose up` calls by harbor will not trigger additional pulls, avoiding Docker Hub rate limits.<br/>`def pre_pull_images(dataset: str) -> None`

- `dataset`: Harbor dataset specifier (e.g. "terminal-bench@2.0").

**`run_terminal_bench`** — Run Terminal-Bench 2.0 using the harbor CLI with the sorcar agent. Before invoking harbor, checks that Docker Hub credentials are configured (to avoid unauthenticated pull rate limits) and pre-pulls all task Docker images so each unique image is fetched exactly once.<br/>`def run_terminal_bench(model: str = 'anthropic/claude-opus-4-6', dataset: str = 'terminal-bench@2.0', n_concurrent: int = 8, trials: int = 1, skip_pre_pull: bool = False) -> None`

- `model`: Model name in harbor format (provider/model).
- `dataset`: Harbor dataset specifier (e.g. "terminal-bench@2.0").
- `n_concurrent`: Number of concurrent task containers.
- `trials`: Number of attempts per task (-k flag). Use 5 for leaderboard.
- `skip_pre_pull`: If True, skip the image pre-pull step.

**`score_results`** — Print a graded summary table from a harbor results JSON file. Reads harbor's output JSON (list of task result dicts) and prints binary score, partial score (fraction of tests passed), and a summary line. Tasks with no partial score data (skipped or missing metadata) show "-" in the partial column.<br/>`def score_results(results_path: Path) -> None`

- `results_path`: Path to harbor results JSON file.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.benchmarks.terminal_bench.test_agent` — *Tests for the terminal bench harbor agent.*

##### `class FakeExecResult`

##### `class FakeEnvironment` — Minimal stand-in for BaseEnvironment.

- **exec**<br/>`async exec(command: str, **kwargs: object) -> FakeExecResult`
- **upload_file**<br/>`async upload_file(source_path: object, target_path: str) -> None`

##### `class FakeContext` — Minimal stand-in for AgentContext.

- **is_empty**<br/>`is_empty() -> bool`

##### `class TestSkipPhrases` — Verify \_SKIP_PHRASES is a non-empty tuple of strings.

- **test_skip_phrases_non_empty**<br/>`test_skip_phrases_non_empty() -> None`
- **test_skip_phrases_are_strings**<br/>`test_skip_phrases_are_strings() -> None`

##### `class TestAgentIdentity` — Agent name and version.

- **test_name**<br/>`test_name() -> None`
- **test_version_matches_package**<br/>`test_version_matches_package() -> None`

##### `class TestRunSkipsImpossibleTasks` — Verify that run() returns immediately for impossible tasks.

- **test_skip_compcert**<br/>`test_skip_compcert() -> None`
- **test_skip_windows_311**<br/>`test_skip_windows_311() -> None`
- **test_skip_ocaml_gc**<br/>`test_skip_ocaml_gc() -> None`
- **test_non_skip_task_runs_normally** — A normal task runs which-check, sorcar, then verifies.<br/>`test_non_skip_task_runs_normally() -> None`

##### `class TestSetup` — Verify setup runs the expected installation steps.

- **test_setup_three_steps**<br/>`test_setup_three_steps() -> None`
- **test_setup_aborts_on_uv_failure**<br/>`test_setup_aborts_on_uv_failure() -> None`
- **test_setup_aborts_on_pip_failure**<br/>`test_setup_aborts_on_pip_failure() -> None`

##### `class TestRunSorcarNotFound` — When sorcar is not installed, run returns early with an error.

- **test_sorcar_missing**<br/>`test_sorcar_missing() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.benchmarks.webarena` — *KISS Sorcar benchmark harness for WebArena.*

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.benchmarks.webarena.agent` — *WebArena agent adapter that delegates to KISS Sorcar.*

##### `class SorcarWebArenaAgent` — Agent that runs KISS Sorcar on WebArena tasks.

**Constructor:** `SorcarWebArenaAgent(model: str | None = None, timeout: int = 600) -> None`

- `model`: LLM model name (e.g. "claude-opus-4-6").

- `timeout`: Max seconds per task before killing sorcar.

- **run_task** — Run sorcar on a single WebArena task. Writes a SYSTEM.md with WebArena-specific instructions before invoking sorcar, then scores the result against reference answers.<br/>`run_task(config_file: Path) -> dict`

  - `config_file`: Path to the WebArena task JSON config file.
  - **Returns:** Dict with task_id, answer, score, stdout, stderr, return_code.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.benchmarks.webarena.run` — *Run WebArena with the sorcar agent.*

**`run_webarena`** — Run sorcar on WebArena task configs and save results.<br/>`def run_webarena(config_dir: Path, model: str = 'claude-opus-4-6', max_tasks: int | None = None, timeout: int = 600) -> None`

- `config_dir`: Directory containing WebArena JSON task configs.
- `model`: LLM model name (e.g. "claude-opus-4-6").
- `max_tasks`: Cap for quick testing (None = all configs).
- `timeout`: Max seconds per task.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels` — *Channel integrations for KISS agents.*

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels._backend_utils` — *Shared helpers for channel backend polling and lifecycle management.*

##### `class ThreadedHTTPServer(ThreadingMixIn, HTTPServer)` — HTTP server with per-request threads and address reuse enabled.

**`wait_for_matching_message`** — Wait for a message matching a predicate with timeout.<br/>`def wait_for_matching_message(*, poll: Callable[[], list[dict[str, Any]]], matches: Callable[[dict[str, Any]], bool], extract_text: Callable[[dict[str, Any]], str], timeout_seconds: float, poll_interval: float) -> str | None`

- `poll`: Callable returning newly observed messages.
- `matches`: Predicate selecting the desired message.
- `extract_text`: Callable extracting the reply text from a matching message.
- `timeout_seconds`: Maximum time to wait.
- `poll_interval`: Delay between polls.
- **Returns:** Extracted reply text, or `None` on timeout.

**`drain_queue_messages`** — Drain up to `limit` messages from a queue, optionally filtering.<br/>`def drain_queue_messages(message_queue: queue.Queue[dict[str, Any]], *, limit: int, keep: Callable[[dict[str, Any]], bool] | None = None) -> list[dict[str, Any]]`

- `message_queue`: Queue containing message dicts.
- `limit`: Maximum number of kept messages to return.
- `keep`: Optional predicate deciding whether a drained message should be kept.
- **Returns:** The kept messages in dequeue order.

**`stop_http_server`** — Shut down an embedded HTTP server and join its thread.<br/>`def stop_http_server(server: HTTPServer | None, server_thread: threading.Thread | None) -> tuple[None, None]`

- `server`: HTTP server instance to stop.
- `server_thread`: Background thread running `serve_forever()`.
- **Returns:** `(None, None)` so callers can reset both attributes succinctly.

**`is_headless_environment`** — Return True when running in a headless/Docker/Linux environment. Checks in order: 1. KISS_HEADLESS env var (explicit override, "1"/"true"/"yes" → headless) 2. Presence of /.dockerenv (running inside Docker) 3. Linux with no $DISPLAY and no $WAYLAND_DISPLAY set<br/>`def is_headless_environment() -> bool`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels._channel_agent_utils` — *Shared helpers for channel agent backends and local config persistence.*

##### `class ToolMethodBackend` — Mixin that exposes public backend methods as agent tools.

- **connection_info** — Human-readable connection status string.<br/>`connection_info() -> str` *(property)*

- **find_channel** — Return *name* as the channel ID. Override for platforms that resolve names via an API call.<br/>`find_channel(name: str) -> str | None`

  - `name`: Channel name or identifier.
  - **Returns:** The channel identifier, or `None` if *name* is empty.

- **find_user** — Return *username* as the user ID. Override for platforms that resolve usernames via an API call.<br/>`find_user(username: str) -> str | None`

  - `username`: Username or identifier.
  - **Returns:** The user identifier, or `None` if *username* is empty.

- **join_channel** — No-op. Override for platforms that require joining a channel.<br/>`join_channel(channel_id: str) -> None`

  - `channel_id`: Channel identifier.

- **disconnect** — No-op. Override for platforms that need connection cleanup.<br/>`disconnect() -> None`

- **is_from_bot** — Return `False`. Override for platforms that can identify bot messages.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

  - `msg`: Message dict from :meth:`poll_messages`.
  - **Returns:** Whether the message was sent by the bot itself.

- **strip_bot_mention** — Return *text* unchanged. Override for platforms with bot @-mentions.<br/>`strip_bot_mention(text: str) -> str`

  - `text`: Raw message text.
  - **Returns:** Text with bot mentions removed.

- **get_tool_methods** — Return the backend's public tool methods.<br/>`get_tool_methods() -> list`

  - **Returns:** List of bound callable methods intended for LLM tool use.

##### `class ChannelConfig` — Encapsulates the 4-function config persistence pattern used by channel agents.

**Constructor:** `ChannelConfig(channel_dir: Path, required_keys: tuple[str, ...]) -> None`

- **load** — Load the config, returning `None` if missing or invalid.<br/>`load() -> dict[str, str] | None`

  - **Returns:** Loaded string dictionary, or `None`.

- **save** — Save *data* to the config file with restricted permissions.<br/>`save(data: dict[str, str]) -> None`

  - `data`: String dictionary to persist.

- **clear** — Delete the config file if it exists.<br/>`clear() -> None`

##### `class BaseChannelAgent` — Mixin for channel agent classes that provides a standard `_get_tools()`

##### `class ChannelRunner` — One-shot channel message runner.

**Constructor:** `ChannelRunner(backend: Any, channel_name: str, agent_name: str, extra_tools: list | None = None, model_name: str = '', max_budget: float = 5.0, work_dir: str = '', allow_users: list[str] | None = None) -> None`

- **run_once** — Check for pending messages, process them, and exit. Connects to the backend, joins the configured channel, retrieves recent messages, filters to allowed users, skips messages the bot has already replied to, and runs a StatefulSorcarAgent for each pending message. Each message is processed synchronously.<br/>`run_once() -> int`
  - **Returns:** Number of messages processed.

**`load_json_config`** — Load a JSON config file containing string values.<br/>`def load_json_config(path: Path, required_keys: tuple[str, ...]) -> dict[str, str] | None`

- `path`: Config file path.
- `required_keys`: Keys that must be present and non-empty.
- **Returns:** Loaded string dictionary, or `None` if the file is missing, malformed, not a dict, or lacks a required key.

**`save_json_config`** — Save a JSON config file with restricted permissions.<br/>`def save_json_config(path: Path, data: dict[str, str]) -> None`

- `path`: Config file path.
- `data`: String dictionary to persist.

**`clear_json_config`** — Delete a JSON config file if it exists.<br/>`def clear_json_config(path: Path) -> None`

- `path`: Config file path.

**`channel_main`** — Standard CLI entry point shared by all channel agents. Handles argument parsing and either one-shot poll mode (when `--channel` is given) or interactive mode (when `-t` is given). Each channel agent's `main()` delegates to this function.<br/>`def channel_main(agent_cls: type, cli_name: str, *, channel_name: str = '', make_backend: Callable[..., Any] | None = None, extra_usage: str = '') -> None`

- `agent_cls`: The channel Agent class to instantiate (e.g. `SlackAgent`).
- `cli_name`: CLI command name for the usage message (e.g. `"kiss-slack"`).
- `channel_name`: Human-readable channel name (e.g. `"Slack"`). Used in status messages and agent naming.
- `make_backend`: Factory that creates and configures a backend for poll mode. May accept a `workspace` keyword argument; if so, the `--workspace` CLI value is forwarded. Should call `sys.exit(1)` if required config is missing. Pass `None` to disable poll mode.
- `extra_usage`: Additional usage flags to append to the usage line (e.g. `"[--list-workspaces]"`).

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.bluebubbles_agent` — *BlueBubbles Agent — StatefulSorcarAgent extension with BlueBubbles REST API tools.*

##### `class BlueBubblesChannelBackend(ToolMethodBackend)` — Channel backend for BlueBubbles REST API.

**Constructor:** `BlueBubblesChannelBackend() -> None`

- **connect** — Connect to BlueBubbles server.<br/>`connect() -> bool`

- **poll_messages** — Poll BlueBubbles for new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a BlueBubbles message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **list_chats** — List recent iMessage conversations.<br/>`list_chats(limit: int = 25, offset: int = 0) -> str`

  - `limit`: Maximum chats to return. Default: 25.
  - `offset`: Pagination offset. Default: 0.
  - **Returns:** JSON string with chat list.

- **get_chat** — Get a specific iMessage conversation.<br/>`get_chat(chat_guid: str) -> str`

  - `chat_guid`: Chat GUID (from list_chats).
  - **Returns:** JSON string with chat details.

- **get_chat_messages** — Get messages from a specific conversation.<br/>`get_chat_messages(chat_guid: str, limit: int = 25, before: str = '', after: str = '') -> str`

  - `chat_guid`: Chat GUID.
  - `limit`: Maximum messages to return. Default: 25.
  - `before`: Return messages before this timestamp (ms).
  - `after`: Return messages after this timestamp (ms).
  - **Returns:** JSON string with message list.

- **post_message** — Send a message to an iMessage conversation.<br/>`post_message(chat_guid: str, text: str) -> str`

  - `chat_guid`: Chat GUID to send to.
  - `text`: Message text.
  - **Returns:** JSON string with ok status.

- **get_server_info** — Get BlueBubbles server information.<br/>`get_server_info() -> str`

  - **Returns:** JSON string with server info.

- **mark_chat_read** — Mark a chat as read.<br/>`mark_chat_read(chat_guid: str) -> str`

  - `chat_guid`: Chat GUID to mark as read.
  - **Returns:** JSON string with ok status.

##### `class BlueBubblesAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with BlueBubbles REST API tools (macOS only).

**Constructor:** `BlueBubblesAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.discord_agent` — *Discord Agent — StatefulSorcarAgent extension with Discord REST API tools.*

##### `class DiscordChannelBackend(ToolMethodBackend)` — Channel backend for Discord REST API v10.

**Constructor:** `DiscordChannelBackend() -> None`

- **connect** — Authenticate with Discord using the stored bot token.<br/>`connect() -> bool`

- **find_channel** — Find a channel by name or numeric ID. If *name* is already a numeric snowflake ID, returns it as-is. Otherwise queries all guilds for a channel matching the name.<br/>`find_channel(name: str) -> str | None`

  - `name`: Channel name or numeric ID.
  - **Returns:** The channel snowflake ID string, or None if not found.

- **poll_messages** — Poll for new Discord messages using REST API.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Discord message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **disconnect** — Release Discord backend state before stop or reconnect.<br/>`disconnect() -> None`

- **list_guilds** — List guilds (servers) the bot is a member of.<br/>`list_guilds(limit: int = 100) -> str`

  - `limit`: Maximum guilds to return (1-200). Default: 100.
  - **Returns:** JSON string with guild list (id, name, icon).

- **list_channels** — List channels in a guild.<br/>`list_channels(guild_id: str, channel_type: str = '') -> str`

  - `guild_id`: Guild (server) ID.
  - `channel_type`: Optional filter by type (0=text, 2=voice, 4=category).
  - **Returns:** JSON string with channel list (id, name, type, topic).

- **get_channel** — Get information about a channel.<br/>`get_channel(channel_id: str) -> str`

  - `channel_id`: Channel ID.
  - **Returns:** JSON string with channel details.

- **get_channel_messages** — Get messages from a channel.<br/>`get_channel_messages(channel_id: str, limit: int = 50, before: str = '', after: str = '') -> str`

  - `channel_id`: Channel ID.
  - `limit`: Number of messages (1-100). Default: 50.
  - `before`: Get messages before this message ID.
  - `after`: Get messages after this message ID.
  - **Returns:** JSON string with message list.

- **post_message** — Send a message to a Discord channel.<br/>`post_message(channel_id: str, content: str, tts: bool = False, reply_to: str = '') -> str`

  - `channel_id`: Channel ID.
  - `content`: Message text (up to 2000 chars).
  - `tts`: Text-to-speech flag. Default: False.
  - `reply_to`: Optional message ID to reply to.
  - **Returns:** JSON string with ok status and message id.

- **edit_message** — Edit an existing Discord message.<br/>`edit_message(channel_id: str, message_id: str, content: str) -> str`

  - `channel_id`: Channel ID.
  - `message_id`: Message ID.
  - `content`: New content.
  - **Returns:** JSON string with ok status.

- **delete_message** — Delete a Discord message.<br/>`delete_message(channel_id: str, message_id: str) -> str`

  - `channel_id`: Channel ID.
  - `message_id`: Message ID to delete.
  - **Returns:** JSON string with ok status.

- **add_reaction** — Add a reaction to a message.<br/>`add_reaction(channel_id: str, message_id: str, emoji: str) -> str`

  - `channel_id`: Channel ID.
  - `message_id`: Message ID.
  - `emoji`: Emoji (e.g. "👍" or "name:id" for custom emojis).
  - **Returns:** JSON string with ok status.

- **create_thread** — Create a thread from a message.<br/>`create_thread(channel_id: str, message_id: str, name: str, auto_archive_duration: int = 1440) -> str`

  - `channel_id`: Channel ID.
  - `message_id`: Message ID to create thread from.
  - `name`: Thread name.
  - `auto_archive_duration`: Minutes before auto-archive (60/1440/4320/10080).
  - **Returns:** JSON string with thread id and name.

- **list_guild_members** — List members of a guild.<br/>`list_guild_members(guild_id: str, limit: int = 100, after: str = '') -> str`

  - `guild_id`: Guild ID.
  - `limit`: Max members to return (1-1000). Default: 100.
  - `after`: User ID to start after (for pagination).
  - **Returns:** JSON string with member list.

- **create_invite** — Create an invite link for a channel.<br/>`create_invite(channel_id: str, max_age: int = 86400, max_uses: int = 0) -> str`

  - `channel_id`: Channel ID.
  - `max_age`: Invite expiry in seconds (0 = never). Default: 86400 (1 day).
  - `max_uses`: Maximum uses (0 = unlimited). Default: 0.
  - **Returns:** JSON string with invite code and URL.

##### `class DiscordAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Discord REST API tools.

**Constructor:** `DiscordAgent() -> None`

- **run** — Run with Discord-specific system prompt encouraging browser-based auth.<br/>`run(**kwargs: Any) -> str`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.feishu_agent` — *Feishu/Lark Agent — StatefulSorcarAgent extension with Feishu Open Platform tools.*

##### `class FeishuChannelBackend(ToolMethodBackend)` — Channel backend for Feishu/Lark Open Platform.

**Constructor:** `FeishuChannelBackend() -> None`

- **connect** — Authenticate with Feishu using stored app credentials.<br/>`connect() -> bool`

- **poll_messages** — Poll Feishu chat for new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Feishu message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **send_text_message** — Send a text message to a Feishu chat or user.<br/>`send_text_message(receive_id: str, text: str, receive_id_type: str = 'chat_id') -> str`

  - `receive_id`: Chat ID, user ID, or open ID depending on receive_id_type.
  - `text`: Message text.
  - `receive_id_type`: "chat_id", "user_id", "open_id", or "email". Default: "chat_id".
  - **Returns:** JSON string with ok status and message id.

- **reply_message** — Reply to an existing Feishu message.<br/>`reply_message(message_id: str, text: str) -> str`

  - `message_id`: ID of the message to reply to.
  - `text`: Reply text.
  - **Returns:** JSON string with ok status and reply message id.

- **delete_message** — Delete a Feishu message.<br/>`delete_message(message_id: str) -> str`

  - `message_id`: Message ID to delete.
  - **Returns:** JSON string with ok status.

- **list_messages** — List messages in a Feishu chat.<br/>`list_messages(container_id: str, start_time: str = '', end_time: str = '', page_size: int = 20) -> str`

  - `container_id`: Chat ID.
  - `start_time`: Start Unix timestamp (seconds). Optional.
  - `end_time`: End Unix timestamp (seconds). Optional.
  - `page_size`: Maximum messages to return. Default: 20.
  - **Returns:** JSON string with message list.

- **list_chats** — List Feishu chats the bot is a member of.<br/>`list_chats(page_size: int = 20, page_token: str = '') -> str`

  - `page_size`: Maximum chats to return. Default: 20.
  - `page_token`: Pagination token.
  - **Returns:** JSON string with chat list (chat_id, name, description).

- **get_chat** — Get information about a Feishu chat.<br/>`get_chat(chat_id: str) -> str`

  - `chat_id`: Chat ID.
  - **Returns:** JSON string with chat details.

- **get_user_info** — Get Feishu user information.<br/>`get_user_info(user_id: str, user_id_type: str = 'open_id') -> str`

  - `user_id`: User ID.
  - `user_id_type`: ID type ("open_id", "user_id", "union_id"). Default: "open_id".
  - **Returns:** JSON string with user info.

##### `class FeishuAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Feishu/Lark Open Platform tools.

**Constructor:** `FeishuAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.gmail_agent` — *Gmail Agent — StatefulSorcarAgent extension with Gmail API tools.*

##### `class GmailChannelBackend(ToolMethodBackend)` — Channel backend for Gmail.

**Constructor:** `GmailChannelBackend() -> None`

- **connect** — Authenticate with Gmail using stored OAuth2 credentials.<br/>`connect() -> bool`

  - **Returns:** True on success, False on failure.

- **find_channel** — Find a Gmail label by name (used as channel ID).<br/>`find_channel(name: str) -> str | None`

  - `name`: Label name to search for.
  - **Returns:** Label ID string, or None if not found.

- **poll_messages** — Poll Gmail inbox for new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

  - `channel_id`: Label ID to poll (use "INBOX" for inbox).
  - `oldest`: History ID or timestamp string for incremental polling.
  - `limit`: Maximum messages to return.
  - **Returns:** Tuple of (messages, updated_oldest). Each message dict has: ts (date), user (from address), text (body).

- **send_message** — Send an email (reply to a thread if thread_ts provided).<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

  - `channel_id`: Recipient email address.
  - `text`: Email body text.
  - `thread_ts`: Thread ID to reply to (optional).

- **wait_for_reply** — Poll a Gmail thread for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

  - `channel_id`: Label ID (unused for Gmail).
  - `thread_ts`: Thread ID to poll.
  - `user_id`: Email address of expected sender.
  - **Returns:** The text of the user's reply.

- **get_profile** — Get the current user's Gmail profile.<br/>`get_profile() -> str`

  - **Returns:** JSON string with email address, messages total, threads total, and history ID.

- **list_messages** — List messages in the user's mailbox.<br/>`list_messages(query: str = '', max_results: int = 20, page_token: str = '', label_ids: str = '') -> str`

  - `query`: Gmail search query (same syntax as Gmail search box). Examples: "is:unread", "from:alice@example.com", "subject:meeting", "newer_than:1d", "has:attachment".
  - `max_results`: Maximum number of messages to return (1-500). Default: 20.
  - `page_token`: Page token for pagination from a previous response.
  - `label_ids`: Comma-separated label IDs to filter by (e.g. "INBOX", "UNREAD", "STARRED").
  - **Returns:** JSON string with message IDs, snippet, and pagination token. Use get_message() with the ID to read full content.

- **get_message** — Get a specific message by ID.<br/>`get_message(message_id: str, format: str = 'full') -> str`

  - `message_id`: The message ID (from list_messages).
  - `format`: Response format. Options: "full" — full message with parsed payload (default). "metadata" — headers only (faster). "raw" — raw RFC 2822 message. "minimal" — just IDs, labels, snippet.
  - **Returns:** JSON string with message headers, body text, labels, and attachment info.

- **send_email** — Send an email message.<br/>`send_email(to: str, subject: str, body: str, cc: str = '', bcc: str = '', html: bool = False) -> str`

  - `to`: Recipient email address(es), comma-separated.
  - `subject`: Email subject line.
  - `body`: Email body text (plain text or HTML).
  - `cc`: CC recipients, comma-separated. Optional.
  - `bcc`: BCC recipients, comma-separated. Optional.
  - `html`: If True, body is treated as HTML. Default: False.
  - **Returns:** JSON string with ok status and the sent message ID.

- **reply_to_message** — Reply to an existing email message.<br/>`reply_to_message(message_id: str, body: str, reply_all: bool = False, html: bool = False) -> str`

  - `message_id`: ID of the message to reply to.
  - `body`: Reply body text (plain text or HTML).
  - `reply_all`: If True, reply to all recipients. Default: False.
  - `html`: If True, body is treated as HTML. Default: False.
  - **Returns:** JSON string with ok status and the reply message ID.

- **create_draft** — Create a draft email.<br/>`create_draft(to: str, subject: str, body: str, cc: str = '', bcc: str = '', html: bool = False) -> str`

  - `to`: Recipient email address(es), comma-separated.
  - `subject`: Email subject line.
  - `body`: Email body text (plain text or HTML).
  - `cc`: CC recipients, comma-separated. Optional.
  - `bcc`: BCC recipients, comma-separated. Optional.
  - `html`: If True, body is treated as HTML. Default: False.
  - **Returns:** JSON string with ok status and draft ID.

- **trash_message** — Move a message to the trash.<br/>`trash_message(message_id: str) -> str`

  - `message_id`: ID of the message to trash.
  - **Returns:** JSON string with ok status.

- **untrash_message** — Remove a message from the trash.<br/>`untrash_message(message_id: str) -> str`

  - `message_id`: ID of the message to untrash.
  - **Returns:** JSON string with ok status.

- **delete_message** — Permanently delete a message (cannot be undone).<br/>`delete_message(message_id: str) -> str`

  - `message_id`: ID of the message to permanently delete.
  - **Returns:** JSON string with ok status.

- **modify_labels** — Modify labels on a message (star, archive, mark read/unread, etc.). Common label IDs: INBOX, UNREAD, STARRED, IMPORTANT, SPAM, TRASH, CATEGORY_PERSONAL, CATEGORY_SOCIAL, CATEGORY_PROMOTIONS. To archive: remove "INBOX". To mark as read: remove "UNREAD". To star: add "STARRED".<br/>`modify_labels(message_id: str, add_label_ids: str = '', remove_label_ids: str = '') -> str`

  - `message_id`: ID of the message to modify.
  - `add_label_ids`: Comma-separated label IDs to add.
  - `remove_label_ids`: Comma-separated label IDs to remove.
  - **Returns:** JSON string with ok status and updated label list.

- **list_labels** — List all labels in the user's mailbox.<br/>`list_labels() -> str`

  - **Returns:** JSON string with label list (id, name, type).

- **create_label** — Create a new label.<br/>`create_label(name: str, text_color: str = '', background_color: str = '') -> str`

  - `name`: Label name (e.g. "Projects/Important"). Use "/" for nested labels.
  - `text_color`: Optional hex text color (e.g. "#000000").
  - `background_color`: Optional hex background color (e.g. "#16a765").
  - **Returns:** JSON string with the new label's id and name.

- **get_attachment** — Download a message attachment.<br/>`get_attachment(message_id: str, attachment_id: str) -> str`

  - `message_id`: ID of the message containing the attachment.
  - `attachment_id`: Attachment ID (from get_message response).
  - **Returns:** JSON string with base64-encoded attachment data and size.

- **get_thread** — Get all messages in an email thread/conversation.<br/>`get_thread(thread_id: str) -> str`

  - `thread_id`: Thread ID (from list_messages or get_message).
  - **Returns:** JSON string with all messages in the thread.

##### `class GmailAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Gmail API tools.

**Constructor:** `GmailAgent() -> None`

- **run** — Run with Gmail-specific system prompt encouraging browser-based auth.<br/>`run(**kwargs: Any) -> str`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.googlechat_agent` — *Google Chat Agent — StatefulSorcarAgent extension with Google Chat API tools.*

##### `class GoogleChatChannelBackend(ToolMethodBackend)` — Channel backend for Google Chat API.

**Constructor:** `GoogleChatChannelBackend() -> None`

- **connect** — Authenticate with Google Chat.<br/>`connect() -> bool`

- **find_channel** — Find a Google Chat space by display name.<br/>`find_channel(name: str) -> str | None`

- **poll_messages** — Poll a Google Chat space for new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Google Chat message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **list_spaces** — List Google Chat spaces (rooms and DMs).<br/>`list_spaces(page_size: int = 20, page_token: str = '') -> str`

  - `page_size`: Maximum spaces to return. Default: 20.
  - `page_token`: Pagination token from a previous response.
  - **Returns:** JSON string with space list (name, displayName, type).

- **get_space** — Get information about a Google Chat space.<br/>`get_space(space_name: str) -> str`

  - `space_name`: Space resource name (e.g. "spaces/ABCDEF").
  - **Returns:** JSON string with space details.

- **list_members** — List members of a Google Chat space.<br/>`list_members(space_name: str, page_size: int = 20, page_token: str = '') -> str`

  - `space_name`: Space resource name.
  - `page_size`: Maximum members to return. Default: 20.
  - `page_token`: Pagination token.
  - **Returns:** JSON string with member list.

- **list_messages** — List messages in a Google Chat space.<br/>`list_messages(space_name: str, page_size: int = 20, page_token: str = '', filter: str = '') -> str`

  - `space_name`: Space resource name (e.g. "spaces/ABCDEF").
  - `page_size`: Maximum messages to return. Default: 20.
  - `page_token`: Pagination token.
  - `filter`: Optional filter (e.g. 'createTime > "2024-01-01T00:00:00Z"').
  - **Returns:** JSON string with message list.

- **get_message** — Get a specific Google Chat message.<br/>`get_message(message_name: str) -> str`

  - `message_name`: Message resource name (e.g. "spaces/X/messages/Y").
  - **Returns:** JSON string with message details.

- **post_message** — Send a message to a Google Chat space.<br/>`post_message(space_name: str, text: str, thread_key: str = '') -> str`

  - `space_name`: Space resource name (e.g. "spaces/ABCDEF").
  - `text`: Message text.
  - `thread_key`: Optional thread key to reply in an existing thread.
  - **Returns:** JSON string with ok status and message name.

- **update_message** — Update an existing Google Chat message.<br/>`update_message(message_name: str, text: str) -> str`

  - `message_name`: Message resource name.
  - `text`: New message text.
  - **Returns:** JSON string with ok status.

- **delete_message** — Delete a Google Chat message.<br/>`delete_message(message_name: str) -> str`

  - `message_name`: Message resource name.
  - **Returns:** JSON string with ok status.

- **create_space** — Create a new Google Chat space.<br/>`create_space(display_name: str, space_type: str = 'SPACE') -> str`

  - `display_name`: Space display name.
  - `space_type`: Space type ("SPACE" or "GROUP_CHAT"). Default: "SPACE".
  - **Returns:** JSON string with space name and display name.

##### `class GoogleChatAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Google Chat API tools.

**Constructor:** `GoogleChatAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.imessage_agent` — *iMessage Agent — StatefulSorcarAgent extension with iMessage tools via AppleScript.*

##### `class IMessageChannelBackend(ToolMethodBackend)` — Channel backend for iMessage via AppleScript.

**Constructor:** `IMessageChannelBackend() -> None`

- **connect** — Check macOS and Messages.app availability.<br/>`connect() -> bool`

- **poll_messages** — Poll iMessage via AppleScript (basic implementation).<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send an iMessage.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Reply waiting is not supported for AppleScript-based iMessage.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **send_imessage** — Send an iMessage or SMS to a recipient.<br/>`send_imessage(recipient: str, text: str, service: str = 'iMessage') -> str`

  - `recipient`: Phone number or Apple ID email to send to.
  - `text`: Message text.
  - `service`: "iMessage" or "SMS". Default: "iMessage".
  - **Returns:** JSON string with ok status.

- **send_attachment** — Send a file attachment via iMessage.<br/>`send_attachment(recipient: str, file_path: str, service: str = 'iMessage') -> str`

  - `recipient`: Phone number or Apple ID email.
  - `file_path`: Absolute path to the file to send.
  - `service`: "iMessage" or "SMS". Default: "iMessage".
  - **Returns:** JSON string with ok status.

- **list_conversations** — List recent iMessage conversations.<br/>`list_conversations() -> str`

  - **Returns:** JSON string with conversation list.

- **get_messages** — Get recent messages with a recipient (basic implementation).<br/>`get_messages(recipient: str, limit: int = 20) -> str`

  - `recipient`: Phone number or email to get messages for.
  - `limit`: Maximum messages to return. Default: 20.
  - **Returns:** JSON string with message list (basic).

##### `class IMessageAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with iMessage tools (macOS only).

**Constructor:** `IMessageAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.irc_agent` — *IRC Agent — StatefulSorcarAgent extension with IRC tools.*

##### `class IRCChannelBackend(ToolMethodBackend)` — Channel backend for IRC via raw socket.

**Constructor:** `IRCChannelBackend() -> None`

- **connect** — Connect to IRC server.<br/>`connect() -> bool`

- **join_channel** — Join an IRC channel.<br/>`join_channel(channel_id: str) -> None`

- **poll_messages** — Return buffered IRC messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send an IRC PRIVMSG.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **disconnect** — Close the IRC socket and join the reader thread.<br/>`disconnect() -> None`

- **is_from_bot** — Check if message is from the bot.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **strip_bot_mention** — Remove bot mention from text.<br/>`strip_bot_mention(text: str) -> str`

- **connect_irc** — Connect to an IRC server.<br/>`connect_irc(server: str, port: int = 6667, nick: str = 'KISSBot', realname: str = 'KISS Agent', password: str = '', use_tls: bool = False) -> str`

  - `server`: IRC server hostname or IP.
  - `port`: Server port. Default: 6667.
  - `nick`: Nickname to use. Default: "KISSBot".
  - `realname`: Real name. Default: "KISS Agent".
  - `password`: Server password. Optional.
  - `use_tls`: Use TLS encryption. Default: False.
  - **Returns:** JSON string with ok status.

- **join_irc_channel** — Join an IRC channel.<br/>`join_irc_channel(channel: str) -> str`

  - `channel`: Channel name (e.g. "#general").
  - **Returns:** JSON string with ok status.

- **leave_channel** — Leave an IRC channel.<br/>`leave_channel(channel: str, reason: str = '') -> str`

  - `channel`: Channel name.
  - `reason`: Optional leave reason.
  - **Returns:** JSON string with ok status.

- **post_message** — Send a message to an IRC channel or user.<br/>`post_message(channel_or_nick: str, text: str) -> str`

  - `channel_or_nick`: Target channel (e.g. "#general") or nick.
  - `text`: Message text.
  - **Returns:** JSON string with ok status.

- **send_notice** — Send a NOTICE to an IRC channel or user.<br/>`send_notice(channel_or_nick: str, text: str) -> str`

  - `channel_or_nick`: Target channel or nick.
  - `text`: Notice text.
  - **Returns:** JSON string with ok status.

- **get_topic** — Get the topic of an IRC channel.<br/>`get_topic(channel: str) -> str`

  - `channel`: Channel name.
  - **Returns:** JSON string with ok status (topic comes via server response).

- **set_topic** — Set the topic of an IRC channel.<br/>`set_topic(channel: str, topic: str) -> str`

  - `channel`: Channel name.
  - `topic`: New topic text.
  - **Returns:** JSON string with ok status.

- **kick_user** — Kick a user from an IRC channel.<br/>`kick_user(channel: str, nick: str, reason: str = '') -> str`

  - `channel`: Channel name.
  - `nick`: Nickname to kick.
  - `reason`: Optional kick reason.
  - **Returns:** JSON string with ok status.

- **whois** — Get WHOIS information about a user.<br/>`whois(nick: str) -> str`

  - `nick`: Nickname to look up.
  - **Returns:** JSON string with ok status (data comes via server response).

- **identify_nickserv** — Identify to NickServ.<br/>`identify_nickserv(password: str) -> str`

  - `password`: NickServ password.
  - **Returns:** JSON string with ok status.

##### `class IRCAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with IRC tools.

**Constructor:** `IRCAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.line_agent` — *LINE Agent — StatefulSorcarAgent extension with LINE Messaging API tools.*

##### `class LineChannelBackend(ToolMethodBackend)` — Channel backend for LINE Messaging API.

**Constructor:** `LineChannelBackend() -> None`

- **connect** — Authenticate with LINE and start webhook server.<br/>`connect() -> bool`

- **poll_messages** — Drain the webhook message queue.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a LINE push message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **disconnect** — Stop the embedded webhook server and release backend resources.<br/>`disconnect() -> None`

- **push_text_message** — Send a push text message to a LINE user or group.<br/>`push_text_message(to: str, text: str) -> str`

  - `to`: Target user ID, group ID, or room ID.
  - `text`: Message text (up to 5000 characters).
  - **Returns:** JSON string with ok status.

- **reply_message** — Reply to a message using the reply token.<br/>`reply_message(reply_token: str, messages_json: str) -> str`

  - `reply_token`: Reply token from an inbound message event.
  - `messages_json`: JSON array of message objects. Example: '[{"type":"text","text":"Hello!"}]'
  - **Returns:** JSON string with ok status.

- **get_profile** — Get a LINE user's profile.<br/>`get_profile(user_id: str) -> str`

  - `user_id`: LINE user ID.
  - **Returns:** JSON string with user profile (displayName, pictureUrl, statusMessage).

- **get_quota** — Get the LINE messaging quota for the current month.<br/>`get_quota() -> str`

  - **Returns:** JSON string with quota information.

- **leave_group** — Leave a LINE group.<br/>`leave_group(group_id: str) -> str`

  - `group_id`: Group ID to leave.
  - **Returns:** JSON string with ok status.

- **push_image_message** — Send a push image message.<br/>`push_image_message(to: str, image_url: str, preview_url: str) -> str`

  - `to`: Target user ID, group ID, or room ID.
  - `image_url`: URL of the full-size image.
  - `preview_url`: URL of the preview image.
  - **Returns:** JSON string with ok status.

##### `class LineAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with LINE Messaging API tools.

**Constructor:** `LineAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.matrix_agent` — *Matrix Agent — StatefulSorcarAgent extension with Matrix protocol tools.*

##### `class MatrixChannelBackend(ToolMethodBackend)` — Channel backend for Matrix via matrix-nio.

**Constructor:** `MatrixChannelBackend() -> None`

- **connect** — Authenticate with Matrix using stored config.<br/>`connect() -> bool`

- **join_channel** — Join a Matrix room.<br/>`join_channel(channel_id: str) -> None`

- **poll_messages** — Poll for new Matrix messages via sync.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Matrix text message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **is_from_bot** — Check if message is from the bot.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **list_rooms** — List joined Matrix rooms.<br/>`list_rooms() -> str`

  - **Returns:** JSON string with room list (id, name, topic).

- **join_room** — Join a Matrix room.<br/>`join_room(room_id_or_alias: str) -> str`

  - `room_id_or_alias`: Room ID (!room:server.org) or alias (#room:server.org).
  - **Returns:** JSON string with ok status and room id.

- **leave_room** — Leave a Matrix room.<br/>`leave_room(room_id: str) -> str`

  - `room_id`: Room ID to leave.
  - **Returns:** JSON string with ok status.

- **send_text_message** — Send a text message to a Matrix room.<br/>`send_text_message(room_id: str, text: str) -> str`

  - `room_id`: Room ID.
  - `text`: Message text.
  - **Returns:** JSON string with ok status and event id.

- **send_notice** — Send a notice (bot message) to a Matrix room.<br/>`send_notice(room_id: str, text: str) -> str`

  - `room_id`: Room ID.
  - `text`: Notice text.
  - **Returns:** JSON string with ok status and event id.

- **get_room_members** — Get members of a Matrix room.<br/>`get_room_members(room_id: str) -> str`

  - `room_id`: Room ID.
  - **Returns:** JSON string with member list.

- **invite_user** — Invite a user to a Matrix room.<br/>`invite_user(room_id: str, user_id: str) -> str`

  - `room_id`: Room ID.
  - `user_id`: User ID to invite (@user:server.org).
  - **Returns:** JSON string with ok status.

- **kick_user** — Kick a user from a Matrix room.<br/>`kick_user(room_id: str, user_id: str, reason: str = '') -> str`

  - `room_id`: Room ID.
  - `user_id`: User ID to kick.
  - `reason`: Optional reason for kick.
  - **Returns:** JSON string with ok status.

- **create_room** — Create a new Matrix room.<br/>`create_room(name: str = '', topic: str = '', is_public: bool = False, alias: str = '') -> str`

  - `name`: Room display name.
  - `topic`: Room topic.
  - `is_public`: Whether the room is publicly joinable. Default: False.
  - `alias`: Optional local alias (without server part).
  - **Returns:** JSON string with room id.

- **get_profile** — Get a Matrix user's profile.<br/>`get_profile(user_id: str) -> str`

  - `user_id`: User ID (@user:server.org).
  - **Returns:** JSON string with display name and avatar.

##### `class MatrixAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Matrix protocol tools.

**Constructor:** `MatrixAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.mattermost_agent` — *Mattermost Agent — StatefulSorcarAgent extension with Mattermost REST API tools.*

##### `class MattermostChannelBackend(ToolMethodBackend)` — Channel backend for Mattermost REST API.

**Constructor:** `MattermostChannelBackend() -> None`

- **connect** — Authenticate with Mattermost using stored config.<br/>`connect() -> bool`

- **poll_messages** — Poll Mattermost channel for new posts.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Mattermost post.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **list_teams** — List Mattermost teams.<br/>`list_teams() -> str`

  - **Returns:** JSON string with team list (id, name, display_name).

- **list_channels** — List channels in a Mattermost team.<br/>`list_channels(team_id: str, page: int = 0, per_page: int = 60) -> str`

  - `team_id`: Team ID.
  - `page`: Page number for pagination. Default: 0.
  - `per_page`: Channels per page. Default: 60.
  - **Returns:** JSON string with channel list.

- **get_channel** — Get information about a Mattermost channel.<br/>`get_channel(channel_id: str) -> str`

  - `channel_id`: Channel ID.
  - **Returns:** JSON string with channel details.

- **list_channel_posts** — List posts in a Mattermost channel.<br/>`list_channel_posts(channel_id: str, page: int = 0, per_page: int = 30) -> str`

  - `channel_id`: Channel ID.
  - `page`: Page number. Default: 0.
  - `per_page`: Posts per page. Default: 30.
  - **Returns:** JSON string with post list.

- **create_post** — Create a post in a Mattermost channel.<br/>`create_post(channel_id: str, message: str, root_id: str = '', file_ids: str = '') -> str`

  - `channel_id`: Channel ID.
  - `message`: Post message text.
  - `root_id`: Root post ID if this is a reply.
  - `file_ids`: Comma-separated file IDs to attach.
  - **Returns:** JSON string with ok status and post id.

- **delete_post** — Delete a Mattermost post.<br/>`delete_post(post_id: str) -> str`

  - `post_id`: Post ID to delete.
  - **Returns:** JSON string with ok status.

- **get_user** — Get a Mattermost user's information.<br/>`get_user(user_id_or_username: str) -> str`

  - `user_id_or_username`: User ID or username. Use "me" for current user.
  - **Returns:** JSON string with user details.

- **list_users** — List Mattermost users.<br/>`list_users(page: int = 0, per_page: int = 60, in_team: str = '', in_channel: str = '') -> str`

  - `page`: Page number. Default: 0.
  - `per_page`: Users per page. Default: 60.
  - `in_team`: Optional team ID to filter by.
  - `in_channel`: Optional channel ID to filter by.
  - **Returns:** JSON string with user list.

- **create_direct_message_channel** — Create a direct message channel between two users.<br/>`create_direct_message_channel(user1_id: str, user2_id: str) -> str`

  - `user1_id`: First user ID.
  - `user2_id`: Second user ID.
  - **Returns:** JSON string with channel id.

- **add_reaction** — Add a reaction to a post.<br/>`add_reaction(user_id: str, post_id: str, emoji_name: str) -> str`

  - `user_id`: User ID adding the reaction.
  - `post_id`: Post ID.
  - `emoji_name`: Emoji name (without colons, e.g. "thumbsup").
  - **Returns:** JSON string with ok status.

##### `class MattermostAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Mattermost REST API tools.

**Constructor:** `MattermostAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.msteams_agent` — *Microsoft Teams Agent — StatefulSorcarAgent extension with MS Teams Graph API tools.*

##### `class MSTeamsChannelBackend(ToolMethodBackend)` — Channel backend for Microsoft Teams via Graph API.

**Constructor:** `MSTeamsChannelBackend() -> None`

- **connect** — Authenticate with Microsoft Graph API.<br/>`connect() -> bool`

- **poll_messages** — Poll MS Teams channel for new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Teams channel message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **is_from_bot** — Check if a message is from the bot.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **list_teams** — List Microsoft Teams the bot/user is a member of.<br/>`list_teams(limit: int = 20) -> str`

  - `limit`: Maximum teams to return. Default: 20.
  - **Returns:** JSON string with team list (id, displayName, description).

- **get_team** — Get details about a Microsoft Team.<br/>`get_team(team_id: str) -> str`

  - `team_id`: Team ID.
  - **Returns:** JSON string with team details.

- **list_channels** — List channels in a Microsoft Team.<br/>`list_channels(team_id: str) -> str`

  - `team_id`: Team ID.
  - **Returns:** JSON string with channel list (id, displayName, membershipType).

- **list_channel_messages** — List messages in a Teams channel.<br/>`list_channel_messages(team_id: str, channel_id: str, top: int = 20) -> str`

  - `team_id`: Team ID.
  - `channel_id`: Channel ID.
  - `top`: Maximum messages to return. Default: 20.
  - **Returns:** JSON string with message list.

- **post_channel_message** — Post a message to a Teams channel.<br/>`post_channel_message(team_id: str, channel_id: str, content: str, content_type: str = 'html') -> str`

  - `team_id`: Team ID.
  - `channel_id`: Channel ID.
  - `content`: Message content.
  - `content_type`: "html" or "text". Default: "html".
  - **Returns:** JSON string with ok status and message id.

- **reply_to_message** — Reply to a Teams channel message.<br/>`reply_to_message(team_id: str, channel_id: str, message_id: str, content: str) -> str`

  - `team_id`: Team ID.
  - `channel_id`: Channel ID.
  - `message_id`: Parent message ID.
  - `content`: Reply content.
  - **Returns:** JSON string with ok status and reply id.

- **list_chats** — List chats for the authenticated user.<br/>`list_chats(top: int = 20) -> str`

  - `top`: Maximum chats to return. Default: 20.
  - **Returns:** JSON string with chat list.

- **post_chat_message** — Post a message to a Teams chat.<br/>`post_chat_message(chat_id: str, content: str, content_type: str = 'text') -> str`

  - `chat_id`: Chat ID.
  - `content`: Message content.
  - `content_type`: "text" or "html". Default: "text".
  - **Returns:** JSON string with ok status and message id.

- **list_team_members** — List members of a Microsoft Team.<br/>`list_team_members(team_id: str, top: int = 50) -> str`

  - `team_id`: Team ID.
  - `top`: Maximum members to return. Default: 50.
  - **Returns:** JSON string with member list.

##### `class MSTeamsAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Microsoft Teams Graph API tools.

**Constructor:** `MSTeamsAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.nextcloud_talk_agent` — *Nextcloud Talk Agent — StatefulSorcarAgent extension with Nextcloud Talk API tools.*

##### `class NextcloudTalkChannelBackend(ToolMethodBackend)` — Channel backend for Nextcloud Talk REST API.

**Constructor:** `NextcloudTalkChannelBackend() -> None`

- **connect** — Authenticate with Nextcloud Talk.<br/>`connect() -> bool`

- **join_channel** — Join a Nextcloud Talk room.<br/>`join_channel(channel_id: str) -> None`

- **poll_messages** — Poll a Nextcloud Talk room for new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Nextcloud Talk message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **is_from_bot** — Check if message is from the bot.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **list_rooms** — List Nextcloud Talk rooms.<br/>`list_rooms() -> str`

  - **Returns:** JSON string with room list (token, displayName, type).

- **get_room** — Get information about a Nextcloud Talk room.<br/>`get_room(token: str) -> str`

  - `token`: Room token.
  - **Returns:** JSON string with room details.

- **create_room** — Create a Nextcloud Talk room.<br/>`create_room(room_type: int = 3, invite: str = '', room_name: str = '') -> str`

  - `room_type`: 1=one-to-one, 2=group, 3=public. Default: 3.
  - `invite`: User ID, group ID, or circle ID to invite.
  - `room_name`: Room display name.
  - **Returns:** JSON string with room token.

- **list_participants** — List participants in a room.<br/>`list_participants(token: str) -> str`

  - `token`: Room token.
  - **Returns:** JSON string with participant list.

- **list_messages** — List messages in a Nextcloud Talk room.<br/>`list_messages(token: str, look_into_future: int = 0, limit: int = 100, last_known_message_id: int = 0) -> str`

  - `token`: Room token.
  - `look_into_future`: 0 for history, 1 for new messages. Default: 0.
  - `limit`: Maximum messages. Default: 100.
  - `last_known_message_id`: Last message ID seen (for pagination).
  - **Returns:** JSON string with message list.

- **post_message** — Post a message to a Nextcloud Talk room.<br/>`post_message(token: str, message: str, reply_to: int = 0) -> str`

  - `token`: Room token.
  - `message`: Message text.
  - `reply_to`: Message ID to reply to. Default: 0 (no reply).
  - **Returns:** JSON string with ok status and message id.

- **set_room_name** — Set the name of a Nextcloud Talk room.<br/>`set_room_name(token: str, name: str) -> str`

  - `token`: Room token.
  - `name`: New room name.
  - **Returns:** JSON string with ok status.

- **delete_message** — Delete a message from a room.<br/>`delete_message(token: str, message_id: int) -> str`

  - `token`: Room token.
  - `message_id`: Message ID to delete.
  - **Returns:** JSON string with ok status.

##### `class NextcloudTalkAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Nextcloud Talk API tools.

**Constructor:** `NextcloudTalkAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.nostr_agent` — *Nostr Agent — StatefulSorcarAgent extension with Nostr protocol tools.*

##### `class NostrChannelBackend(ToolMethodBackend)` — Channel backend for Nostr protocol via pynostr.

**Constructor:** `NostrChannelBackend() -> None`

- **connect** — Load Nostr keys from stored config.<br/>`connect() -> bool`

- **poll_messages** — Poll Nostr relays for new events (basic implementation).<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Publish a Nostr note.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Reply waiting is not currently supported for Nostr.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **is_from_bot** — Check if event is from this key.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **publish_note** — Publish a text note (kind 1) to Nostr.<br/>`publish_note(content: str) -> str`

  - `content`: Note content text.
  - **Returns:** JSON string with ok status and event id.

- **publish_reply** — Publish a reply to an existing Nostr event.<br/>`publish_reply(content: str, reply_to_event_id: str) -> str`

  - `content`: Reply content.
  - `reply_to_event_id`: Event ID to reply to.
  - **Returns:** JSON string with ok status and event id.

- **send_dm** — Send an encrypted direct message (NIP-04).<br/>`send_dm(recipient_pubkey: str, content: str) -> str`

  - `recipient_pubkey`: Recipient's public key (hex).
  - `content`: Message content (will be encrypted).
  - **Returns:** JSON string with ok status and event id.

- **get_profile** — Get the current user's Nostr profile.<br/>`get_profile() -> str`

  - **Returns:** JSON string with public key info.

- **set_profile** — Set the Nostr user profile (kind 0).<br/>`set_profile(name: str = '', about: str = '', picture: str = '', nip05: str = '') -> str`

  - `name`: Display name.
  - `about`: Bio/about text.
  - `picture`: Profile picture URL.
  - `nip05`: NIP-05 identifier (user@domain.com).
  - **Returns:** JSON string with ok status and event id.

- **list_relays** — List configured Nostr relays.<br/>`list_relays() -> str`

  - **Returns:** JSON string with relay list.

- **add_relay** — Add a Nostr relay to the configuration.<br/>`add_relay(relay_url: str) -> str`

  - `relay_url`: WebSocket URL of the relay (wss://...).
  - **Returns:** JSON string with ok status.

- **remove_relay** — Remove a Nostr relay from the configuration.<br/>`remove_relay(relay_url: str) -> str`

  - `relay_url`: Relay URL to remove.
  - **Returns:** JSON string with ok status.

##### `class NostrAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Nostr protocol tools.

**Constructor:** `NostrAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.phone_control_agent` — *Phone Control Agent — StatefulSorcarAgent extension with Android phone control tools.*

##### `class PhoneControlChannelBackend(ToolMethodBackend)` — Channel backend for Android phone control via REST API.

**Constructor:** `PhoneControlChannelBackend() -> None`

- **connect** — Connect to phone companion app.<br/>`connect() -> bool`

- **poll_messages** — Poll for new SMS messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send an SMS.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply SMS from a specific number.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **send_sms** — Send an SMS message.<br/>`send_sms(to: str, text: str) -> str`

  - `to`: Recipient phone number.
  - `text`: Message text.
  - **Returns:** JSON string with ok status.

- **make_call** — Make a phone call.<br/>`make_call(to: str) -> str`

  - `to`: Phone number to call.
  - **Returns:** JSON string with ok status.

- **end_call** — End the current active call.<br/>`end_call() -> str`

  - **Returns:** JSON string with ok status.

- **list_sms_conversations** — List recent SMS conversations.<br/>`list_sms_conversations(limit: int = 20) -> str`

  - `limit`: Maximum conversations to return. Default: 20.
  - **Returns:** JSON string with conversation list.

- **get_sms_messages** — Get messages in an SMS thread.<br/>`get_sms_messages(thread_id: str, limit: int = 50) -> str`

  - `thread_id`: Thread ID from list_sms_conversations.
  - `limit`: Maximum messages to return. Default: 50.
  - **Returns:** JSON string with message list.

- **get_call_log** — Get recent call log.<br/>`get_call_log(limit: int = 20) -> str`

  - `limit`: Maximum calls to return. Default: 20.
  - **Returns:** JSON string with call list.

- **get_device_info** — Get phone device information.<br/>`get_device_info() -> str`

  - **Returns:** JSON string with device info (model, battery, etc).

- **list_notifications** — List current phone notifications.<br/>`list_notifications() -> str`

  - **Returns:** JSON string with notification list.

- **dismiss_notification** — Dismiss a phone notification.<br/>`dismiss_notification(notification_id: str) -> str`

  - `notification_id`: Notification ID to dismiss.
  - **Returns:** JSON string with ok status.

- **send_notification_reply** — Reply to a phone notification (e.g. WhatsApp, Signal).<br/>`send_notification_reply(notification_id: str, text: str) -> str`

  - `notification_id`: Notification ID to reply to.
  - `text`: Reply text.
  - **Returns:** JSON string with ok status.

##### `class PhoneControlAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Android phone control tools.

**Constructor:** `PhoneControlAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.signal_agent` — *Signal Agent — StatefulSorcarAgent extension with Signal CLI tools.*

##### `class SignalChannelBackend(ToolMethodBackend)` — Channel backend for Signal via signal-cli.

**Constructor:** `SignalChannelBackend() -> None`

- **connect** — Load Signal config.<br/>`connect() -> bool`

- **poll_messages** — Receive pending Signal messages via signal-cli.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Signal message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **is_from_bot** — Check if a message is from the bot.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **send_signal_message** — Send a Signal text message.<br/>`send_signal_message(recipient: str, message: str) -> str`

  - `recipient`: Recipient phone number in E.164 format.
  - `message`: Message text to send.
  - **Returns:** JSON string with ok status.

- **receive_messages** — Receive pending Signal messages.<br/>`receive_messages(timeout: int = 5) -> str`

  - `timeout`: Seconds to wait for messages. Default: 5.
  - **Returns:** JSON string with list of received messages.

- **send_attachment** — Send a Signal message with an attachment.<br/>`send_attachment(recipient: str, message: str, file_path: str) -> str`

  - `recipient`: Recipient phone number.
  - `message`: Message text.
  - `file_path`: Local path to the file to attach.
  - **Returns:** JSON string with ok status.

- **list_contacts** — List Signal contacts.<br/>`list_contacts() -> str`

  - **Returns:** JSON string with contact list.

- **list_groups** — List Signal groups.<br/>`list_groups() -> str`

  - **Returns:** JSON string with group list.

##### `class SignalAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Signal CLI tools.

**Constructor:** `SignalAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.slack_agent` — *Slack Agent — StatefulSorcarAgent extension with Slack API tools.*

##### `class SlackChannelBackend(ToolMethodBackend)` — Slack channel backend.

**Constructor:** `SlackChannelBackend(workspace: str = 'default') -> None`

- **connect** — Authenticate with Slack using the stored bot token. Uses the workspace set at construction time to load the appropriate token.<br/>`connect() -> bool`

  - **Returns:** True on success, False on failure.

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

- **poll_thread_messages** — Poll a Slack thread for new replies since *oldest*. Used by the poller to detect user replies within active threads. The parent message itself is excluded from the results. Retries up to 3 times on transient network errors with exponential backoff (same strategy as `poll_messages`).<br/>`poll_thread_messages(channel_id: str, thread_ts: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

  - `channel_id`: Channel ID containing the thread.
  - `thread_ts`: Timestamp of the parent message (thread root).
  - `oldest`: Only return messages newer than this timestamp.
  - `limit`: Maximum number of messages to return.
  - **Returns:** Tuple of (reply messages sorted oldest-first, updated oldest timestamp).

- **send_message** — Send a message to a Slack channel, optionally in a thread.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

  - `channel_id`: Channel ID to post to.
  - `text`: Message text (supports Slack mrkdwn formatting).
  - `thread_ts`: If non-empty, reply in this thread.

- **wait_for_reply** — Poll a Slack thread for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

  - `channel_id`: Channel ID containing the thread.
  - `thread_ts`: Timestamp of the parent message (thread root).
  - `user_id`: User ID to wait for a reply from.
  - **Returns:** The text of the user's reply message, or `None` on timeout.

- **disconnect** — Release Slack backend state before stop or reconnect.<br/>`disconnect() -> None`

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

##### `class SlackAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Slack workspace tools.

**Constructor:** `SlackAgent(workspace: str = 'default') -> None`

- **run** — Run with Slack-specific system prompt encouraging browser-based auth.<br/>`run(**kwargs: Any) -> str`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.sms_agent` — *SMS Agent — StatefulSorcarAgent extension with Twilio SMS tools.*

##### `class SMSChannelBackend(ToolMethodBackend)` — Channel backend for Twilio SMS.

**Constructor:** `SMSChannelBackend() -> None`

- **connect** — Authenticate with Twilio using stored config.<br/>`connect() -> bool`

- **poll_messages** — Poll Twilio for recent inbound messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send an SMS.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific number.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **is_from_bot** — Check if message is from the bot's number.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **send_sms** — Send an SMS message via Twilio.<br/>`send_sms(to: str, body: str) -> str`

  - `to`: Recipient phone number in E.164 format.
  - `body`: Message text (up to 1600 characters).
  - **Returns:** JSON string with ok status and message SID.

- **send_mms** — Send an MMS message with media via Twilio.<br/>`send_mms(to: str, body: str, media_url: str) -> str`

  - `to`: Recipient phone number in E.164 format.
  - `body`: Message text.
  - `media_url`: Publicly accessible URL of the media file.
  - **Returns:** JSON string with ok status and message SID.

- **list_messages** — List Twilio messages.<br/>`list_messages(to: str = '', from_: str = '', limit: int = 20, page_token: str = '') -> str`

  - `to`: Filter by recipient phone number. Optional.
  - `from_`: Filter by sender phone number. Optional.
  - `limit`: Maximum messages to return. Default: 20.
  - `page_token`: Pagination token. Optional.
  - **Returns:** JSON string with message list.

- **get_message** — Get details about a specific Twilio message.<br/>`get_message(message_sid: str) -> str`

  - `message_sid`: Message SID (e.g. "SM...").
  - **Returns:** JSON string with message details.

- **list_phone_numbers** — List Twilio phone numbers on the account.<br/>`list_phone_numbers(limit: int = 20) -> str`

  - `limit`: Maximum numbers to return. Default: 20.
  - **Returns:** JSON string with phone number list.

- **get_account_info** — Get Twilio account information.<br/>`get_account_info() -> str`

  - **Returns:** JSON string with account details.

- **send_whatsapp_message** — Send a WhatsApp message via Twilio.<br/>`send_whatsapp_message(to: str, body: str) -> str`

  - `to`: Recipient WhatsApp number in format "whatsapp:+14155238886".
  - `body`: Message text.
  - **Returns:** JSON string with ok status and message SID.

- **create_call** — Create a Twilio voice call.<br/>`create_call(to: str, url: str, method: str = 'GET') -> str`

  - `to`: Phone number to call.
  - `url`: TwiML URL for the call instructions.
  - `method`: HTTP method for the URL. Default: "GET".
  - **Returns:** JSON string with ok status and call SID.

- **list_calls** — List recent Twilio calls.<br/>`list_calls(to: str = '', from_: str = '', limit: int = 20) -> str`

  - `to`: Filter by recipient phone number. Optional.
  - `from_`: Filter by caller phone number. Optional.
  - `limit`: Maximum calls to return. Default: 20.
  - **Returns:** JSON string with call list.

- **get_call** — Get details about a specific Twilio call.<br/>`get_call(call_sid: str) -> str`

  - `call_sid`: Call SID (e.g. "CA...").
  - **Returns:** JSON string with call details.

- **cancel_message** — Cancel a queued or scheduled Twilio message.<br/>`cancel_message(message_sid: str) -> str`

  - `message_sid`: Message SID to cancel.
  - **Returns:** JSON string with ok status.

##### `class SMSAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Twilio SMS tools.

**Constructor:** `SMSAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.synology_chat_agent` — *Synology Chat Agent — StatefulSorcarAgent extension with Synology Chat webhook API.*

##### `class SynologyChatChannelBackend(ToolMethodBackend)` — Channel backend for Synology Chat webhooks.

**Constructor:** `SynologyChatChannelBackend() -> None`

- **connect** — Load Synology config and start webhook server.<br/>`connect() -> bool`

- **poll_messages** — Drain the webhook message queue.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Synology Chat message via incoming webhook.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **disconnect** — Stop the embedded webhook server and release backend resources.<br/>`disconnect() -> None`

- **post_message** — Send a message to Synology Chat via incoming webhook.<br/>`post_message(text: str, user_ids: str = '') -> str`

  - `text`: Message text.
  - `user_ids`: Comma-separated user IDs to send to (optional). If empty, sends to the default channel.
  - **Returns:** JSON string with ok status.

- **send_file_message** — Send a message with a file attachment.<br/>`send_file_message(text: str, file_url: str) -> str`

  - `text`: Message text.
  - `file_url`: URL of the file to attach.
  - **Returns:** JSON string with ok status.

##### `class SynologyChatAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Synology Chat webhook tools.

**Constructor:** `SynologyChatAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.telegram_agent` — *Telegram Agent — StatefulSorcarAgent extension with Telegram Bot API tools.*

##### `class TelegramChannelBackend(ToolMethodBackend)` — Channel backend for Telegram Bot API.

**Constructor:** `TelegramChannelBackend() -> None`

- **connect** — Authenticate with Telegram using the stored bot token.<br/>`connect() -> bool`

- **poll_messages** — Poll for new Telegram updates via getUpdates.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Telegram message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **send_text** — Send a text message to a Telegram chat.<br/>`send_text(chat_id: str, text: str, reply_to_message_id: str = '') -> str`

  - `chat_id`: Chat ID (integer as string) or @username.
  - `text`: Message text (supports Markdown).
  - `reply_to_message_id`: Optional message ID to reply to.
  - **Returns:** JSON string with ok status and message_id.

- **send_photo** — Send a photo to a Telegram chat.<br/>`send_photo(chat_id: str, photo_url_or_path: str, caption: str = '') -> str`

  - `chat_id`: Chat ID or @username.
  - `photo_url_or_path`: URL or local file path of the photo.
  - `caption`: Optional caption text.
  - **Returns:** JSON string with ok status and message_id.

- **send_document** — Send a document/file to a Telegram chat.<br/>`send_document(chat_id: str, document_path: str, caption: str = '') -> str`

  - `chat_id`: Chat ID or @username.
  - `document_path`: Local file path to send.
  - `caption`: Optional caption text.
  - **Returns:** JSON string with ok status and message_id.

- **edit_message_text** — Edit an existing message text.<br/>`edit_message_text(chat_id: str, message_id: str, text: str) -> str`

  - `chat_id`: Chat ID where the message is.
  - `message_id`: ID of the message to edit.
  - `text`: New message text.
  - **Returns:** JSON string with ok status.

- **delete_message** — Delete a message.<br/>`delete_message(chat_id: str, message_id: str) -> str`

  - `chat_id`: Chat ID where the message is.
  - `message_id`: ID of the message to delete.
  - **Returns:** JSON string with ok status.

- **pin_message** — Pin a message in a chat.<br/>`pin_message(chat_id: str, message_id: str) -> str`

  - `chat_id`: Chat ID.
  - `message_id`: ID of the message to pin.
  - **Returns:** JSON string with ok status.

- **unpin_message** — Unpin a message (or all messages) in a chat.<br/>`unpin_message(chat_id: str, message_id: str = '') -> str`

  - `chat_id`: Chat ID.
  - `message_id`: ID of specific message to unpin. If empty, unpins all.
  - **Returns:** JSON string with ok status.

- **get_chat** — Get information about a chat.<br/>`get_chat(chat_id: str) -> str`

  - `chat_id`: Chat ID or @username.
  - **Returns:** JSON string with chat info (id, title, type, members_count).

- **get_chat_members_count** — Get the number of members in a chat.<br/>`get_chat_members_count(chat_id: str) -> str`

  - `chat_id`: Chat ID or @username.
  - **Returns:** JSON string with member count.

- **get_chat_member** — Get information about a chat member.<br/>`get_chat_member(chat_id: str, user_id: str) -> str`

  - `chat_id`: Chat ID.
  - `user_id`: User ID.
  - **Returns:** JSON string with member info (user, status).

- **ban_chat_member** — Ban a user from a chat.<br/>`ban_chat_member(chat_id: str, user_id: str) -> str`

  - `chat_id`: Chat ID.
  - `user_id`: User ID to ban.
  - **Returns:** JSON string with ok status.

- **unban_chat_member** — Unban a user from a chat.<br/>`unban_chat_member(chat_id: str, user_id: str) -> str`

  - `chat_id`: Chat ID.
  - `user_id`: User ID to unban.
  - **Returns:** JSON string with ok status.

- **get_updates** — Get recent updates (messages) from the bot.<br/>`get_updates(offset: str = '', limit: int = 10) -> str`

  - `offset`: Update ID offset for pagination.
  - `limit`: Maximum number of updates to return (1-100).
  - **Returns:** JSON string with list of update objects.

- **send_poll** — Send a poll to a chat.<br/>`send_poll(chat_id: str, question: str, options_json: str, is_anonymous: bool = True) -> str`

  - `chat_id`: Chat ID.
  - `question`: Poll question.
  - `options_json`: JSON array of option strings (2-10 options).
  - `is_anonymous`: Whether the poll is anonymous. Default: True.
  - **Returns:** JSON string with ok status and message_id.

- **forward_message** — Forward a message to another chat.<br/>`forward_message(chat_id: str, from_chat_id: str, message_id: str) -> str`

  - `chat_id`: Target chat ID.
  - `from_chat_id`: Source chat ID.
  - `message_id`: ID of the message to forward.
  - **Returns:** JSON string with ok status and message_id.

##### `class TelegramAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Telegram Bot API tools.

**Constructor:** `TelegramAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.tlon_agent` — *Tlon/Urbit Agent — StatefulSorcarAgent extension with Tlon/Urbit Eyre HTTP tools.*

##### `class TlonChannelBackend(ToolMethodBackend)` — Channel backend for Tlon/Urbit Eyre HTTP.

**Constructor:** `TlonChannelBackend() -> None`

- **connect** — Authenticate with Urbit ship.<br/>`connect() -> bool`

- **poll_messages** — Poll event queue for messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Tlon/Urbit poke.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **list_groups** — List Urbit groups.<br/>`list_groups() -> str`

  - **Returns:** JSON string with group list.

- **list_channels** — List channels in an Urbit group.<br/>`list_channels(group_path: str) -> str`

  - `group_path`: Group path (e.g. "~sampel/my-group").
  - **Returns:** JSON string with channel list.

- **get_messages** — Get recent messages from a Tlon channel.<br/>`get_messages(group_path: str, channel_name: str, count: int = 20) -> str`

  - `group_path`: Group path.
  - `channel_name`: Channel name within the group.
  - `count`: Number of messages to retrieve. Default: 20.
  - **Returns:** JSON string with messages.

- **post_message** — Post a message to a Tlon channel.<br/>`post_message(group_path: str, channel_name: str, content: str) -> str`

  - `group_path`: Group path (e.g. "~sampel/my-group").
  - `channel_name`: Channel name within the group.
  - `content`: Message content text.
  - **Returns:** JSON string with ok status.

- **get_profile** — Get the current ship's profile.<br/>`get_profile() -> str`

  - **Returns:** JSON string with profile info.

- **poke** — Send a poke to an Urbit app.<br/>`poke(app: str, mark: str, json_body: str) -> str`

  - `app`: Gall agent name (e.g. "groups").
  - `mark`: Mark name (e.g. "groups-action").
  - `json_body`: JSON string of the poke body.
  - **Returns:** JSON string with ok status.

- **scry** — Perform a scry request on an Urbit app.<br/>`scry(app: str, path: str) -> str`

  - `app`: Gall agent name.
  - `path`: Scry path (starting with /).
  - **Returns:** JSON string with scry result.

##### `class TlonAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Tlon/Urbit Eyre HTTP tools.

**Constructor:** `TlonAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.twitch_agent` — *Twitch Agent — StatefulSorcarAgent extension with Twitch Helix API + Chat tools.*

##### `class TwitchChannelBackend(ToolMethodBackend)` — Channel backend for Twitch Helix API.

**Constructor:** `TwitchChannelBackend() -> None`

- **connect** — Authenticate with Twitch using stored config.<br/>`connect() -> bool`

- **poll_messages** — Poll for Twitch events (basic REST polling).<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Twitch chat message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Reply waiting is not currently supported for Twitch.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **get_stream_info** — Get live stream information for a Twitch channel.<br/>`get_stream_info(broadcaster_login: str) -> str`

  - `broadcaster_login`: Twitch channel username.
  - **Returns:** JSON string with stream info (game, title, viewer count, etc).

- **get_channel_info** — Get channel information for a Twitch broadcaster.<br/>`get_channel_info(broadcaster_id: str) -> str`

  - `broadcaster_id`: Twitch broadcaster ID.
  - **Returns:** JSON string with channel info.

- **get_user_info** — Get Twitch user information.<br/>`get_user_info(login_or_id: str) -> str`

  - `login_or_id`: Twitch username (login) or user ID.
  - **Returns:** JSON string with user info.

- **get_chatters** — Get current chatters in a Twitch channel.<br/>`get_chatters(broadcaster_id: str, moderator_id: str = '') -> str`

  - `broadcaster_id`: Broadcaster user ID.
  - `moderator_id`: Moderator user ID (optional, defaults to broadcaster).
  - **Returns:** JSON string with chatters list.

- **send_chat_message** — Send a message to a Twitch chat.<br/>`send_chat_message(broadcaster_id: str, sender_id: str, message: str) -> str`

  - `broadcaster_id`: Broadcaster channel ID.
  - `sender_id`: Sender user ID.
  - `message`: Message text.
  - **Returns:** JSON string with ok status.

- **ban_user** — Ban or timeout a Twitch user.<br/>`ban_user(broadcaster_id: str, moderator_id: str, user_id: str, duration: int = 0, reason: str = '') -> str`

  - `broadcaster_id`: Broadcaster channel ID.
  - `moderator_id`: Moderator user ID.
  - `user_id`: User ID to ban.
  - `duration`: Timeout duration in seconds (0 = permanent ban).
  - `reason`: Optional ban reason.
  - **Returns:** JSON string with ok status.

- **search_channels** — Search for Twitch channels by name.<br/>`search_channels(query: str, limit: int = 10) -> str`

  - `query`: Search query.
  - `limit`: Maximum channels to return. Default: 10.
  - **Returns:** JSON string with matching channels.

- **get_clips** — Get clips from a Twitch channel.<br/>`get_clips(broadcaster_id: str, limit: int = 20) -> str`

  - `broadcaster_id`: Broadcaster ID.
  - `limit`: Maximum clips to return. Default: 20.
  - **Returns:** JSON string with clip list.

- **create_clip** — Create a clip from a live stream.<br/>`create_clip(broadcaster_id: str, has_delay: bool = False) -> str`

  - `broadcaster_id`: Broadcaster ID.
  - `has_delay`: Whether to add a 5-second delay. Default: False.
  - **Returns:** JSON string with clip edit URL.

##### `class TwitchAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Twitch Helix API tools.

**Constructor:** `TwitchAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.whatsapp_agent` — *WhatsApp Agent — StatefulSorcarAgent extension with WhatsApp Business Cloud API tools.*

##### `class WhatsAppChannelBackend(ToolMethodBackend)` — Channel backend for WhatsApp Business Cloud API.

**Constructor:** `WhatsAppChannelBackend() -> None`

- **connect** — Authenticate with WhatsApp using stored config and start webhook server.<br/>`connect() -> bool`

  - **Returns:** True on success, False on failure.

- **poll_messages** — Drain the webhook message queue and return new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

  - `channel_id`: Recipient phone number (unused — all messages returned).
  - `oldest`: Unused for push-mode channels.
  - `limit`: Maximum messages to return.
  - **Returns:** Tuple of (messages, oldest). Each message dict has at minimum: ts, user (from), text.

- **send_message** — Send a text message to a WhatsApp number.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

  - `channel_id`: Recipient phone number in E.164 format.
  - `text`: Message text.
  - `thread_ts`: Unused for WhatsApp.

- **wait_for_reply** — Block until a message from a specific user is received.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

  - `channel_id`: Unused for WhatsApp.
  - `thread_ts`: Unused for WhatsApp.
  - `user_id`: Phone number to wait for.
  - **Returns:** The text of the user's reply.

- **disconnect** — Stop the embedded webhook server and release backend resources.<br/>`disconnect() -> None`

- **send_text_message** — Send a text message to a WhatsApp number.<br/>`send_text_message(to: str, body: str, preview_url: bool = False) -> str`

  - `to`: Recipient phone number in E.164 format (e.g. "+14155238886"). Include country code, no spaces or dashes.
  - `body`: Message text (up to 4096 characters).
  - `preview_url`: If True, URLs in the body will show a preview. Default: False.
  - **Returns:** JSON string with ok status and message_id.

- **send_template_message** — Send a pre-approved template message. Template messages are required to initiate conversations outside the 24-hour customer service window.<br/>`send_template_message(to: str, template_name: str, language_code: str = 'en_US', components: str = '') -> str`

  - `to`: Recipient phone number in E.164 format.
  - `template_name`: Name of the approved message template.
  - `language_code`: Template language code (e.g. "en_US"). Default: "en_US".
  - `components`: Optional JSON string of template components (header, body, button parameters).
  - **Returns:** JSON string with ok status and message_id.

- **send_media_message** — Send a media message (image, document, audio, video, sticker). Provide either media_id (from upload_media) or link (public URL).<br/>`send_media_message(to: str, media_type: str, media_id: str = '', link: str = '', caption: str = '', filename: str = '') -> str`

  - `to`: Recipient phone number in E.164 format.
  - `media_type`: Type of media. Options: "image", "document", "audio", "video", "sticker".
  - `media_id`: Media ID from a previous upload_media call.
  - `link`: Public URL of the media file. Used if media_id is empty.
  - `caption`: Optional caption (supported for image, video, document).
  - `filename`: Optional filename (for document type).
  - **Returns:** JSON string with ok status and message_id.

- **send_reaction** — React to a message with an emoji.<br/>`send_reaction(to: str, message_id: str, emoji: str) -> str`

  - `to`: Phone number of the message recipient.
  - `message_id`: ID of the message to react to.
  - `emoji`: Emoji character (e.g. "👍", "❤️", "😂").
  - **Returns:** JSON string with ok status and message_id.

- **send_location_message** — Send a location message.<br/>`send_location_message(to: str, latitude: str, longitude: str, name: str = '', address: str = '') -> str`

  - `to`: Recipient phone number in E.164 format.
  - `latitude`: Latitude of the location (e.g. "37.7749").
  - `longitude`: Longitude of the location (e.g. "-122.4194").
  - `name`: Optional name of the location.
  - `address`: Optional address of the location.
  - **Returns:** JSON string with ok status and message_id.

- **send_interactive_message** — Send an interactive message (buttons, lists, or product messages).<br/>`send_interactive_message(to: str, interactive_json: str) -> str`

  - `to`: Recipient phone number in E.164 format.
  - `interactive_json`: JSON string of the interactive object.
  - **Returns:** JSON string with ok status and message_id.

- **send_contact_message** — Send a contact card message.<br/>`send_contact_message(to: str, contacts_json: str) -> str`

  - `to`: Recipient phone number in E.164 format.
  - `contacts_json`: JSON string of contacts array.
  - **Returns:** JSON string with ok status and message_id.

- **mark_as_read** — Mark a received message as read.<br/>`mark_as_read(message_id: str) -> str`

  - `message_id`: ID of the message to mark as read.
  - **Returns:** JSON string with ok status.

- **get_business_profile** — Get the WhatsApp Business profile information.<br/>`get_business_profile() -> str`

  - **Returns:** JSON string with business profile data (about, address, description, email, websites, profile_picture_url).

- **update_business_profile** — Update the WhatsApp Business profile.<br/>`update_business_profile(about: str = '', address: str = '', description: str = '', email: str = '', websites: str = '', vertical: str = '') -> str`

  - `about`: Short description (max 139 characters).
  - `address`: Business address.
  - `description`: Full business description (max 512 characters).
  - `email`: Business email address.
  - `websites`: Comma-separated list of website URLs (max 2).
  - `vertical`: Business category (e.g. "RETAIL", "FOOD", "HEALTH").
  - **Returns:** JSON string with ok status.

- **upload_media** — Upload a media file for later sending.<br/>`upload_media(file_path: str, mime_type: str) -> str`

  - `file_path`: Local path to the file to upload.
  - `mime_type`: MIME type of the file (e.g. "image/jpeg", "application/pdf", "video/mp4", "audio/ogg").
  - **Returns:** JSON string with ok status and media_id (use in send_media_message).

- **get_media_url** — Get the download URL for an uploaded media file.<br/>`get_media_url(media_id: str) -> str`

  - `media_id`: Media ID from upload_media or a received message.
  - **Returns:** JSON string with ok status, url, mime_type, and file_size.

- **delete_media** — Delete an uploaded media file.<br/>`delete_media(media_id: str) -> str`

  - `media_id`: Media ID to delete.
  - **Returns:** JSON string with ok status.

- **list_message_templates** — List available message templates for the WhatsApp Business Account. Requires waba_id to be configured.<br/>`list_message_templates(limit: int = 20, status: str = '') -> str`

  - `limit`: Maximum number of templates to return. Default: 20.
  - `status`: Filter by status ("APPROVED", "PENDING", "REJECTED"). If empty, returns all statuses.
  - **Returns:** JSON string with template list (name, status, category, language).

##### `class WhatsAppAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with WhatsApp Business Cloud API tools.

**Constructor:** `WhatsAppAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.channels.zalo_agent` — *Zalo Agent — StatefulSorcarAgent extension with Zalo Official Account API tools.*

##### `class ZaloChannelBackend(ToolMethodBackend)` — Channel backend for Zalo OA API.

**Constructor:** `ZaloChannelBackend() -> None`

- **connect** — Load Zalo config and start webhook server.<br/>`connect() -> bool`

- **poll_messages** — Drain the webhook message queue.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Zalo text message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **disconnect** — Stop the embedded webhook server and release backend resources.<br/>`disconnect() -> None`

- **send_text_message** — Send a text message to a Zalo user.<br/>`send_text_message(to_user_id: str, text: str) -> str`

  - `to_user_id`: Zalo user ID.
  - `text`: Message text.
  - **Returns:** JSON string with ok status.

- **send_image_message** — Send an image message to a Zalo user.<br/>`send_image_message(to_user_id: str, image_url: str, caption: str = '') -> str`

  - `to_user_id`: Zalo user ID.
  - `image_url`: URL of the image to send.
  - `caption`: Optional image caption.
  - **Returns:** JSON string with ok status.

- **get_follower_profile** — Get a Zalo follower's profile.<br/>`get_follower_profile(user_id: str) -> str`

  - `user_id`: Zalo user ID.
  - **Returns:** JSON string with user profile.

- **get_followers** — Get followers of the Zalo OA.<br/>`get_followers(offset: int = 0, count: int = 50) -> str`

  - `offset`: Pagination offset. Default: 0.
  - `count`: Number of followers to return (max 50). Default: 50.
  - **Returns:** JSON string with follower list.

- **get_oa_info** — Get Zalo Official Account information.<br/>`get_oa_info() -> str`

  - **Returns:** JSON string with OA info (name, id, description, etc).

- **get_recent_messages** — Get recent messages from the OA.<br/>`get_recent_messages(offset: int = 0, count: int = 10) -> str`

  - `offset`: Pagination offset. Default: 0.
  - `count`: Number of messages. Default: 10.
  - **Returns:** JSON string with message list.

- **get_conversation** — Get conversation history with a specific user.<br/>`get_conversation(user_id: str, offset: int = 0, count: int = 20) -> str`

  - `user_id`: Zalo user ID.
  - `offset`: Pagination offset. Default: 0.
  - `count`: Number of messages. Default: 20.
  - **Returns:** JSON string with conversation messages.

- **upload_image** — Upload an image file to Zalo.<br/>`upload_image(file_path: str) -> str`

  - `file_path`: Local path to the image file.
  - **Returns:** JSON string with ok status and attachment_id.

##### `class ZaloAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Zalo OA API tools.

**Constructor:** `ZaloAgent() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core` — *Core module for the KISS agent framework.*

```python
from kiss.agents.vscode.kiss_project.src.kiss.core import Config, DEFAULT_CONFIG, KISSError
```

##### `class Config(BaseModel)`

##### `class KISSError(ValueError)` — Custom exception class for KISS framework errors.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.base` — *Base agent class with common functionality for all KISS agents.*

##### `class Base` — Base class for all KISS agents with common state management and persistence.

**Constructor:** `Base(name: str) -> None`

- `name`: The name identifier for the agent.

- **get_global_budget_used** — Return the global budget total under the shared class lock.<br/>`get_global_budget_used() -> float`

- **reset_global_budget** — Reset the shared process-wide budget counter to zero.<br/>`reset_global_budget() -> None`

- **set_printer** — Configure the output printer for this agent. If an explicit *printer* is provided, it is always used regardless of the verbose setting. Otherwise a `ConsolePrinter` is created when verbose output is enabled.<br/>`set_printer(printer: Printer | None = None, verbose: bool | None = None) -> None`

  - `printer`: An existing Printer instance to use directly. If provided, verbose is ignored.
  - `verbose`: Whether to print to the console. Defaults to True if None.

- **get_trajectory** — Return the trajectory as JSON for visualization.<br/>`get_trajectory() -> str`

  - **Returns:** str: A JSON-formatted string of all messages in the agent's history.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.config` — *Configuration Pydantic models for KISS agent settings with CLI support.*

##### `class Config(BaseModel)`

**`set_artifact_base_dir`** — Set the base directory used to resolve `artifact_dir`.<br/>`def set_artifact_base_dir(base_dir: str | Path | None) -> str`

- `base_dir`: Directory whose `.kiss.artifacts` child should contain generated job artifacts. `None` resets to the project root.
- **Returns:** The resolved artifact job directory.

**`get_artifact_dir`** — Return the active artifact directory, creating it lazily if needed.<br/>`def get_artifact_dir() -> str`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.config_builder` — *Configuration builder for KISS agent settings with CLI support.*

**`add_config`** — Build the KISS config, optionally overriding with command-line arguments. This function accumulates configs - each call adds a new config field while preserving existing fields from previous calls.<br/>`def add_config(name: str, config_class: type[BaseModel]) -> None`

- `name`: Name of the config class.
- `config_class`: Class of the config.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.kiss_agent` — *Core KISS agent implementation with native function calling support.*

##### `class KISSAgent(Base)` — A KISS agent using native function calling.

**Constructor:** `KISSAgent(name: str) -> None`

- **run** — Runs the agent's main ReAct loop to solve the task.<br/>`run(model_name: str, prompt_template: str, arguments: dict[str, str] | None = None, system_prompt: str = '', tools: list[Callable[..., Any]] | None = None, is_agentic: bool = True, max_steps: int | None = None, max_budget: float | None = None, model_config: dict[str, Any] | None = None, printer: Printer | None = None, verbose: bool | None = None, attachments: list[Attachment] | None = None) -> str`

  - `model_name`: The name of the model to use for the agent.
  - `prompt_template`: The prompt template for the agent.
  - `arguments`: The arguments to be substituted into the prompt template. Default is None.
  - `system_prompt`: Optional system prompt to provide to the model. Default is empty string (no system prompt).
  - `tools`: The tools to use for the agent. If None, no tools are provided (only the built-in finish tool is added).
  - `is_agentic`: Whether the agent is agentic. Default is True.
  - `max_steps`: The maximum number of steps to take. Default is 100.
  - `max_budget`: The maximum budget to spend. Default is 10.0.
  - `model_config`: The model configuration to use for the agent. Default is None.
  - `printer`: Optional printer for streaming output. Default is None.
  - `verbose`: Whether to print output to console. Default is None (verbose enabled).
  - `attachments`: Optional file attachments (images, PDFs) to include in the initial prompt. Default is None.
  - **Returns:** str: The result of the agent's task.

- **finish** — The agent must call this function with the final answer to the task.<br/>`finish(result: str) -> str`

  - `result`: The result generated by the agent.
  - **Returns:** Returns the result of the agent's task.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.kiss_error` — *Custom error class for KISS framework exceptions.*

##### `class KISSError(ValueError)` — Custom exception class for KISS framework errors.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.models` — *Model implementations for different LLM providers.*

```python
from kiss.agents.vscode.kiss_project.src.kiss.core.models import Attachment, Model, AnthropicModel, ClaudeCodeModel, OpenAICompatibleModel, GeminiModel
```

##### `class Attachment` — A file attachment (image, document, audio, or video) to include in a prompt.

- **from_file** — Create an Attachment from a file path.<br/>`from_file(path: str) -> 'Attachment'`

  - `path`: Path to the file to attach.
  - **Returns:** An Attachment with the file's bytes and detected MIME type.

- **to_base64** — Return the file data as a base64-encoded string.<br/>`to_base64() -> str`

- **to_data_url** — Return a data: URL suitable for OpenAI image_url fields.<br/>`to_data_url() -> str`

##### `class Model(ABC)` — Abstract base class for LLM provider implementations.

**Constructor:** `Model(model_name: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)`

- `model_name`: The name/identifier of the model.

- `model_config`: Optional dictionary of model configuration parameters.

- `token_callback`: Optional callback invoked with each streamed text token.

- **reset_conversation** — Reset conversation state for reuse across sub-sessions. Clears the conversation history and usage info while keeping the HTTP client and model configuration intact.<br/>`reset_conversation() -> None`

- **initialize** — Initializes the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt to start the conversation.
  - `attachments`: Optional list of file attachments (images, PDFs, audio, video) to include. Provider support varies — unsupported types are skipped with a warning.

- **generate** — Generates content from prompt.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** tuple\[str, Any\]: A tuple of (generated_text, raw_response).

- **generate_and_process_with_tools** — Generates content with tools, processes the response, and adds it to conversation.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]], tools_schema: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - `tools_schema`: Optional pre-built tool schema list. When provided, skips schema rebuilding from function_map (performance optimization).
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

#### `kiss.agents.vscode.kiss_project.src.kiss.core.models.anthropic_model` — *Anthropic model implementation for Claude models.*

##### `class AnthropicModel(Model)` — A model that uses Anthropic's Messages API (Claude).

**Constructor:** `AnthropicModel(model_name: str, api_key: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)`

- `model_name`: The name of the Claude model to use.

- `api_key`: The Anthropic API key for authentication.

- `model_config`: Optional dictionary of model configuration parameters.

- `token_callback`: Optional callback invoked with each streamed text token.

- **initialize** — Initializes the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt to start the conversation.
  - `attachments`: Optional list of file attachments (images, PDFs, audio, video) to include. Audio attachments are automatically transcribed to text via OpenAI Whisper when an `OPENAI_API_KEY` is available; otherwise they are skipped with a warning. Video attachments are always skipped.

- **generate** — Generates content from the current conversation.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** tuple\[str, Any\]: A tuple of (generated_text, raw_response).

- **generate_and_process_with_tools** — Generates content with tools and processes the response.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]], tools_schema: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - `tools_schema`: Optional pre-built OpenAI-format tool schema list.
  - **Returns:** tuple\[list\[dict[str, Any]\], str, Any\]: A tuple of (function_calls, response_text, raw_response).

- **add_function_results_to_conversation_and_return** — Add tool results to the conversation.<br/>`add_function_results_to_conversation_and_return(function_results: list[tuple[str, dict[str, Any]]]) -> None`

  - `function_results`: List of (func_name, result_dict) tuples. result_dict can contain: - "result": The result content string - "tool_use_id": Optional explicit tool_use_id to use

- **extract_input_output_token_counts_from_response** — Extracts token counts from an Anthropic API response.<br/>`extract_input_output_token_counts_from_response(response: Any) -> tuple[int, int, int, int]`

  - **Returns:** (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).

- **get_embedding** — Generates an embedding vector for the given text.<br/>`get_embedding(text: str, embedding_model: str | None = None) -> list[float]`

  - `text`: The text to generate an embedding for.
  - `embedding_model`: Optional model name (not used by Anthropic).

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.models.claude_code_model` — *Claude Code model implementation — uses the `claude` CLI as an LLM backend.*

##### `class ClaudeCodeModel(Model)` — A model that delegates to the Claude Code CLI for LLM completions.

**Constructor:** `ClaudeCodeModel(model_name: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)`

- `model_name`: Full model name including `cc/` prefix (e.g. `cc/opus`).

- `model_config`: Optional configuration. Recognised keys: - `system_instruction` (str): System prompt for the session. - `timeout` (int): Subprocess timeout in seconds (default 300).

- `token_callback`: Optional callback invoked with each streamed text token.

- **initialize** — Initialize the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt.
  - `attachments`: Not supported — ignored with a warning if provided.

- **generate** — Generate a response using the Claude Code CLI.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** tuple\[str, Any\]: (generated_text, parsed_json_response).

- **generate_and_process_with_tools** — Generate with text-based tool calling via the Claude Code CLI. Tool descriptions are injected into the system prompt. The model's text output is parsed for JSON `tool_calls` blocks, which are returned to the framework for execution — the CLI itself runs in pure LLM mode (`--tools ""`), **not** as an agent.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]], tools_schema: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - `tools_schema`: Ignored (text-based tool calling builds its own prompt).
  - **Returns:** Tuple of `(function_calls, content, response)`.

- **extract_input_output_token_counts_from_response** — Extract token counts from the Claude Code CLI JSON response.<br/>`extract_input_output_token_counts_from_response(response: Any) -> tuple[int, int, int, int]`

  - `response`: The parsed JSON response from the CLI.
  - **Returns:** (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).

- **get_embedding** — Not supported — Claude Code CLI does not provide embeddings.<br/>`get_embedding(text: str, embedding_model: str | None = None) -> list[float]`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.models.gemini_model` — *Gemini model implementation for Google's GenAI models.*

##### `class GeminiModel(Model)` — A model that uses Google's GenAI API (Gemini).

**Constructor:** `GeminiModel(model_name: str, api_key: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)`

- `model_name`: The name of the Gemini model to use.

- `api_key`: The Google API key for authentication.

- `model_config`: Optional dictionary of model configuration parameters.

- `token_callback`: Optional callback invoked with each streamed text token.

- **reset_conversation** — Reset conversation state including thought signatures.<br/>`reset_conversation() -> None`

- **initialize** — Initializes the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt to start the conversation.
  - `attachments`: Optional list of file attachments (images, PDFs) to include.

- **generate** — Generates content from prompt without tools.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** tuple\[str, Any\]: A tuple of (generated_text, raw_response).

- **generate_and_process_with_tools** — Generates content with tools, processes the response, and adds it to conversation.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]], tools_schema: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - `tools_schema`: Optional pre-built OpenAI-format tool schema list.
  - **Returns:** tuple\[list\[dict[str, Any]\], str, Any\]: A tuple of (function_calls, response_text, raw_response).

- **extract_input_output_token_counts_from_response** — Extracts token counts from a Gemini API response.<br/>`extract_input_output_token_counts_from_response(response: Any) -> tuple[int, int, int, int]`

  - **Returns:** (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).

- **get_embedding** — Generates an embedding vector for the given text.<br/>`get_embedding(text: str, embedding_model: str | None = None) -> list[float]`

  - `text`: The text to generate an embedding for.
  - `embedding_model`: Optional model name. Defaults to "text-embedding-004".
  - **Returns:** list\[float\]: The embedding vector as a list of floats.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.models.model` — *Abstract base class for LLM provider model implementations.*

##### `class Attachment` — A file attachment (image, document, audio, or video) to include in a prompt.

- **from_file** — Create an Attachment from a file path.<br/>`from_file(path: str) -> 'Attachment'`

  - `path`: Path to the file to attach.
  - **Returns:** An Attachment with the file's bytes and detected MIME type.

- **to_base64** — Return the file data as a base64-encoded string.<br/>`to_base64() -> str`

- **to_data_url** — Return a data: URL suitable for OpenAI image_url fields.<br/>`to_data_url() -> str`

##### `class Model(ABC)` — Abstract base class for LLM provider implementations.

**Constructor:** `Model(model_name: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)`

- `model_name`: The name/identifier of the model.

- `model_config`: Optional dictionary of model configuration parameters.

- `token_callback`: Optional callback invoked with each streamed text token.

- **reset_conversation** — Reset conversation state for reuse across sub-sessions. Clears the conversation history and usage info while keeping the HTTP client and model configuration intact.<br/>`reset_conversation() -> None`

- **initialize** — Initializes the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt to start the conversation.
  - `attachments`: Optional list of file attachments (images, PDFs, audio, video) to include. Provider support varies — unsupported types are skipped with a warning.

- **generate** — Generates content from prompt.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** tuple\[str, Any\]: A tuple of (generated_text, raw_response).

- **generate_and_process_with_tools** — Generates content with tools, processes the response, and adds it to conversation.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]], tools_schema: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - `tools_schema`: Optional pre-built tool schema list. When provided, skips schema rebuilding from function_map (performance optimization).
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

**`transcribe_audio`** — Transcribe audio bytes to text using OpenAI's Whisper API. This is used as a fallback for model providers that do not support audio attachments natively (e.g. Anthropic).<br/>`def transcribe_audio(data: bytes, mime_type: str, api_key: str | None = None) -> str`

- `data`: Raw audio file bytes.
- `mime_type`: MIME type of the audio (e.g. `"audio/mpeg"`).
- `api_key`: OpenAI API key. Falls back to the `OPENAI_API_KEY` environment variable when *None*.
- **Returns:** The transcribed text.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.models.model_info` — *Model information: pricing and context lengths for supported LLM providers.*

##### `class ModelInfo` — Container for model metadata including pricing and capabilities.

**Constructor:** `ModelInfo(context_length: int, input_price_per_million: float, output_price_per_million: float, is_function_calling_supported: bool, is_embedding_supported: bool, is_generation_supported: bool, cache_read_price_per_million: float | None = None, cache_write_price_per_million: float | None = None)`

**`is_model_flaky`** — Check if a model is known to be flaky.<br/>`def is_model_flaky(model_name: str) -> bool`

- `model_name`: The name of the model to check.
- **Returns:** bool: True if the model is known to have reliability issues.

**`get_flaky_reason`** — Get the reason why a model is flaky.<br/>`def get_flaky_reason(model_name: str) -> str`

- `model_name`: The name of the model to check.
- **Returns:** str: The reason for flakiness, or empty string if not flaky.

**`model`** — Get a model instance based on model name prefix.<br/>`def model(model_name: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None) -> Model`

- `model_name`: The name of the model (with provider prefix if applicable). Accepts harbor-style `provider/model` names (e.g. `openai/gpt-5.4`, `anthropic/claude-opus-4-6`) — the redundant provider prefix is stripped automatically.
- `model_config`: Optional dictionary of model configuration parameters. If it contains "base_url", routing is bypassed and an OpenAICompatibleModel is built with that base_url and optional "api_key".
- `token_callback`: Optional callback invoked with each streamed text token.
- **Returns:** Model: An appropriate Model instance for the specified model.

**`get_available_models`** — Return model names for which an API key is configured and generation is supported.<br/>`def get_available_models() -> list[str]`

- **Returns:** list\[str\]: Sorted list of model name strings that have a configured API key and support text generation.

**`get_default_model`** — Return the best default model based on which API keys are configured. Priority order: Anthropic > OpenRouter > Gemini > OpenAI > Together AI. Falls back to `"claude-opus-4-6"` if no keys are set.<br/>`def get_default_model() -> str`

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

#### `kiss.agents.vscode.kiss_project.src.kiss.core.models.openai_compatible_model` — *OpenAI-compatible model implementation for custom endpoints.*

##### `class OpenAICompatibleModel(Model)` — A model that uses an OpenAI-compatible API with a custom base URL.

**Constructor:** `OpenAICompatibleModel(model_name: str, base_url: str, api_key: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)`

- `model_name`: The name/identifier of the model to use.

- `base_url`: The base URL for the API endpoint (e.g., "http://localhost:11434/v1").

- `api_key`: API key for authentication.

- `model_config`: Optional dictionary of model configuration parameters.

- `token_callback`: Optional callback invoked with each streamed text token.

- **initialize** — Initialize the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt to start the conversation.
  - `attachments`: Optional list of file attachments (images, PDFs) to include.

- **generate** — Generate content from prompt without tools.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** A tuple of (content, response) where content is the generated text and response is the raw API response object.

- **generate_and_process_with_tools** — Generate content with tools, process the response, and add it to conversation.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]], tools_schema: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - `tools_schema`: Optional pre-built tool schema list.
  - **Returns:** A tuple of (function_calls, content, response) where function_calls is a list of dictionaries containing tool call information, content is the text response, and response is the raw API response object.

- **extract_input_output_token_counts_from_response** — Extract token counts from an API response.<br/>`extract_input_output_token_counts_from_response(response: Any) -> tuple[int, int, int, int]`

  - **Returns:** (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens). For OpenAI, cached_tokens is a subset of prompt_tokens; input_tokens is reported as (prompt_tokens - cached_tokens) so costs apply correctly. OpenRouter returns cache_write_tokens in prompt_tokens_details. OpenAI reasoning models may report reasoning tokens in completion_tokens_details.reasoning_tokens; those are counted as output tokens so Sorcar shows thinking-token usage.

- **get_embedding** — Generate an embedding vector for the given text.<br/>`get_embedding(text: str, embedding_model: str | None = None) -> list[float]`

  - `text`: The text to generate an embedding for.
  - `embedding_model`: Optional model name for embedding generation. Uses the model's name if not specified.
  - **Returns:** A list of floating point numbers representing the embedding vector.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.print_to_console` — *Console output formatting for KISS agents.*

##### `class ConsolePrinter(StreamEventParser, Printer)`

**Constructor:** `ConsolePrinter(file: Any = None) -> None`

- **reset** — Reset internal streaming and tool-parsing state for a new turn.<br/>`reset() -> None`

- **print** — Render content to the console using Rich formatting.<br/>`print(content: Any, type: str = 'text', **kwargs: Any) -> str`

  - `content`: The content to display.
  - `type`: Content type (e.g. "text", "prompt", "stream_event", "tool_call", "tool_result", "result", "message").
  - `**kwargs`: Additional options such as tool_input, is_error, cost, total_tokens.
  - **Returns:** str: Extracted text from stream events, or empty string.

- **token_callback** — Stream a single token to the console, styled by current block type.<br/>`token_callback(token: str) -> None`

  - `token`: The text token to display.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.printer` — *Abstract base class and shared utilities for KISS agent printers.*

##### `class StreamEventParser` — Shared parser for LLM stream events used by both console and browser printers.

**Constructor:** `StreamEventParser() -> None`

- **reset_stream_state** — Reset block type and tool buffer state.<br/>`reset_stream_state() -> None`
- **parse_stream_event** — Parse a stream event, dispatch to on\_\* callbacks, return extracted text.<br/>`parse_stream_event(event: Any) -> str`
  - `event`: An event object with an `event` dict attribute.
  - **Returns:** str: Any text content extracted from text or thinking deltas.

##### `class Printer(ABC)`

- **print** — Render content to the output destination.<br/>`print(content: Any, type: str = 'text', **kwargs: Any) -> str`

  - `content`: The content to display.
  - `type`: Content type (e.g. "text", "prompt", "stream_event", "tool_call", "tool_result", "result", "message").
  - `**kwargs`: Additional type-specific options (e.g. tool_input, is_error).
  - **Returns:** str: Any extracted text (e.g. streamed text deltas), or empty string.

- **token_callback** — Handle a single streamed token from the LLM.<br/>`token_callback(token: str) -> None`

  - `token`: The text token to process.

- **reset** — Reset the printer's internal streaming state between messages.<br/>`reset() -> None`

##### `class MultiPrinter(Printer)`

**Constructor:** `MultiPrinter(printers: list[Printer]) -> None`

- **print** — Dispatch a print call to all child printers.<br/>`print(content: Any, type: str = 'text', **kwargs: Any) -> str`

  - `content`: The content to display.
  - `type`: Content type forwarded to each child printer.
  - `**kwargs`: Additional options forwarded to each child printer.
  - **Returns:** str: The first non-empty result from child printers.

- **token_callback** — Forward a streamed token to all child printers.<br/>`token_callback(token: str) -> None`

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
- **Returns:** dict\[str, str\]: Keys not in KNOWN_KEYS mapped to their string values.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.relentless_agent` — *Base relentless agent with smart continuation for long tasks.*

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
  - `verbose`: Whether to print output to console. Defaults to True.
  - `tools`: List of callable tools available to the agent during execution.
  - `attachments`: Optional file attachments (images, PDFs) for the initial prompt.
  - **Returns:** YAML string with 'success' and 'summary' keys.

**`finish`** — Finish execution with status and summary.<br/>`def finish(success: bool, is_continue: bool = False, summary: str = '') -> str`

- `success`: True if the agent has successfully completed the task, False otherwise
- `is_continue`: True if the task is incomplete and should continue, False otherwise
- `summary`: precise chronologically-ordered list of things the agent did with the reason for doing that along with relevant code snippets

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.core.utils` — *Utility functions for the KISS core module.*

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

**`finish`** — The agent must call this function with the final status, analysis, and result when it has solved the given task. Status **MUST** be 'success' or 'failure'.<br/>`def finish(status: str = 'success', analysis: str = '', result: str = '') -> str`

- `status`: The status of the agent's task ('success' or 'failure'). Defaults to 'success'.
- `analysis`: The analysis of the agent's trajectory.
- `result`: The result generated by the agent.
- **Returns:** A YAML string containing the status, analysis, and result of the agent's task.

**`resolve_path`** — Resolve a path relative to base_dir if not absolute.<br/>`def resolve_path(p: str, base_dir: str) -> Path`

- `p`: The path string to resolve.
- `base_dir`: The base directory for relative path resolution.
- **Returns:** Path: The resolved absolute path.

**`is_subpath`** — Check if target has any prefix in whitelist.<br/>`def is_subpath(target: Path, whitelist: list[Path]) -> bool`

- `target`: The path to check.
- `whitelist`: List of allowed path prefixes.
- **Returns:** bool: True if target is under any path in whitelist, False otherwise.

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.docker` — *Docker wrapper module for the KISS agent framework.*

```python
from kiss.agents.vscode.kiss_project.src.kiss.docker import DockerManager, DockerTools
```

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.docker.docker_manager` — *Docker library for managing Docker containers and executing commands.*

##### `class DockerManager` — Manages Docker container lifecycle and command execution.

**Constructor:** `DockerManager(image_name: str, tag: str = 'latest', workdir: str = '/', mount_shared_volume: bool = True, ports: dict[int, int] | None = None) -> None`

- `image_name`: The name of the Docker image (e.g., 'ubuntu', 'python')

- `tag`: The tag/version of the image (default: 'latest')

- `workdir`: The working directory inside the container

- `mount_shared_volume`: Whether to mount a shared volume. Set to False for images that already have content in the workdir (e.g., SWE-bench).

- `ports`: Port mapping from container port to host port. Example: {8080: 8080} maps container port 8080 to host port 8080. Example: {80: 8000, 443: 8443} maps multiple ports.

- **open** — Pull and load a Docker image, then create and start a container.<br/>`open() -> None`

- **Bash** — Execute a bash command in the running Docker container.<br/>`Bash(command: str, description: str, timeout_seconds: int = 30) -> str`

  - `command`: The bash command to execute
  - `description`: A short description of the command in natural language
  - `timeout_seconds`: Maximum time to wait before treating the command as hung.
  - **Returns:** The output of the command, including stdout, stderr, and exit code

- **get_host_port** — Get the host port mapped to a container port.<br/>`get_host_port(container_port: int) -> int | None`

  - `container_port`: The container port to look up.
  - **Returns:** The host port mapped to the container port, or None if not mapped.

- **close** — Stop and remove the Docker container. Handles cleanup of both the container and any temporary directories created for shared volumes.<br/>`close() -> None`

______________________________________________________________________

#### `kiss.agents.vscode.kiss_project.src.kiss.docker.docker_tools` — *File tools (Read, Write, Edit) that execute inside a Docker container via bash.*

##### `class DockerTools` — File tools that execute inside a Docker container via bash.

**Constructor:** `DockerTools(bash_fn: Callable[[str, str], str]) -> None`

- `bash_fn`: Callable(command, description) -> output string. Executes a bash command inside the Docker container.

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

______________________________________________________________________

#### `kiss.agents.vscode.server` — *VS Code extension backend server for Sorcar agent.*

##### `class VSCodePrinter(BaseBrowserPrinter)` — Printer that outputs JSON events to stdout for VS Code extension.

**Constructor:** `VSCodePrinter() -> None`

- **broadcast** — Write event as a JSON line to stdout and record it. Injects `tabId` from thread-local storage when available so the frontend can route events to the correct chat tab.<br/>`broadcast(event: dict[str, Any]) -> None`
  - `event`: The event dictionary to emit.

##### `class VSCodeServer` — Backend server for VS Code extension.

**Constructor:** `VSCodeServer() -> None`

- **run** — Main loop: read commands from stdin, execute them.<br/>`run() -> None`
  **`parse_task_tags`** — Parse `<task>...</task>` tags from *text* and return individual tasks. When the input contains one or more `<task>` blocks with non-empty content, each block's content is returned as a separate list element. If no valid `<task>` blocks are found (or all are empty/whitespace), the original *text* is returned as a single-element list so that callers can always iterate without special-casing.<br/>`def parse_task_tags(text: str) -> list[str]`

- `text`: Input text potentially containing `<task>...</task>` tags.

- **Returns:** List of task strings. Always contains at least one element.

______________________________________________________________________

### `kiss.benchmarks` — *KISS Sorcar benchmark harnesses for SWE-bench Pro, Terminal-Bench 2.0, and WebArena.*

______________________________________________________________________

#### `kiss.benchmarks.generate_dashboard` — *Generate a benchmark results dashboard as a self-contained HTML page.*

**`generate_dashboard`** — Read benchmark results and produce an HTML dashboard.<br/>`def generate_dashboard(swebench_results_path: str | None = None, tbench_results_path: str | None = None, output_path: str = 'results/dashboard.html') -> None`

- `swebench_results_path`: Path to SWE-bench Pro eval results JSON.
- `tbench_results_path`: Path to Terminal-Bench results JSONL.
- `output_path`: Path for the output HTML file.

______________________________________________________________________

#### `kiss.benchmarks.swebench_pro` — *SWE-bench Pro benchmark harness for KISS Sorcar.*

______________________________________________________________________

#### `kiss.benchmarks.swebench_pro.adapter` — *Convert SWE-bench Pro instances into sorcar task prompts.*

**`make_sorcar_task`** — Build a sorcar task prompt from a SWE-bench Pro instance. Returns a task string that tells sorcar: - The repo is already cloned at /app - The issue to fix (problem_statement) - To produce a git diff as the solution<br/>`def make_sorcar_task(instance: dict) -> str`

- `instance`: A SWE-bench Pro dataset row with at least 'problem_statement' and 'repo' fields.
- **Returns:** A formatted task prompt string for sorcar.

______________________________________________________________________

#### `kiss.benchmarks.swebench_pro.eval` — *Thin wrapper around the official SWE-bench Pro evaluation script.*

**`run_eval`** — Run the official SWE-bench Pro evaluation script.<br/>`def run_eval(patch_path: str, num_workers: int = 8, use_local_docker: bool = True, block_network: bool = False, docker_platform: str | None = None, redo: bool = False) -> None`

- `patch_path`: Path to the patches JSON file.
- `num_workers`: Number of parallel Docker workers.
- `use_local_docker`: Use local Docker instead of Modal.
- `block_network`: Block network access inside eval containers.
- `docker_platform`: Docker platform (e.g. "linux/amd64" for Apple Silicon).
- `redo`: Re-evaluate even if output exists.

______________________________________________________________________

#### `kiss.benchmarks.swebench_pro.run` — *Run sorcar on SWE-bench Pro instances and collect patches.*

**`run_instance`** — Run sorcar on a single SWE-bench Pro instance inside its Docker container. Steps: 1. docker run the instance's image (jefzda/sweap-images:\<dockerhub_tag>) 2. Inside the container, invoke sorcar with the task prompt 3. Capture the generated patch (git diff from /app) 4. Return {"instance_id": ..., "model_patch": ..., "prefix": model}<br/>`def run_instance(instance: dict, model: str, budget: float) -> dict`

- `instance`: A SWE-bench Pro dataset row.
- `model`: LLM model name (e.g. "claude-opus-4-6").
- `budget`: Max USD budget per instance.
- **Returns:** A dict with instance_id, model_patch, and prefix fields.

**`run_all`** — Iterate over all SWE-bench Pro public instances, generate patches.<br/>`def run_all(model: str, budget: float, max_instances: int | None = None, workers: int = 1) -> None`

- `model`: LLM model name (e.g. "claude-opus-4-6").
- `budget`: Max USD budget per instance.
- `max_instances`: Cap for quick testing (None = all 731).
- `workers`: Number of parallel workers (currently sequential).

______________________________________________________________________

#### `kiss.benchmarks.terminal_bench` — *Terminal-Bench 2.0 benchmark harness for KISS Sorcar.*

______________________________________________________________________

#### `kiss.benchmarks.terminal_bench.agent` — *Harbor agent adapter that delegates to KISS Sorcar.*

##### `class SorcarHarborAgent(BaseAgent)` — Harbor-compatible agent that uses KISS Sorcar as the backend.

- **name** — Return the agent's name.<br/>`name() -> str`

- **version** — Return the agent version string.<br/>`version() -> str | None`

- **setup** — Install sorcar inside the harbor container. Installs uv, then kiss-agent-framework as a uv tool (which manages its own Python), and writes a tbench-specific SYSTEM.md that replaces the generic IDE system prompt with terminal-bench instructions. Each step is run separately so failures are logged clearly and do not silently abort the chain.<br/>`async setup(environment: BaseEnvironment) -> None`

  - `environment`: The harbor execution environment.

- **run** — Run sorcar with the task instruction inside the container. After the first sorcar run, automatically runs the task's test.sh and retries once with failure output if tests don't pass.<br/>`async run(instruction: str, environment: BaseEnvironment, context: AgentContext) -> None`

  - `instruction`: Natural language task description from harbor.
  - `environment`: The harbor execution environment.
  - `context`: Agent context for storing token/cost metadata.

______________________________________________________________________

#### `kiss.benchmarks.terminal_bench.run` — *Run Terminal-Bench 2.0 with the sorcar harbor agent.*

**`is_docker_hub_authenticated`** — Check whether Docker Hub credentials are configured. Reads ~/.docker/config.json to find the credential store, then queries it via `docker-credential-<store> list`. Returns True if any credential is stored for `https://index.docker.io/`. Falls back to checking the `auths` dict when no credential store is configured.<br/>`def is_docker_hub_authenticated() -> bool`

**`pre_pull_images`** — Pre-pull all Docker images needed by a harbor dataset. Resolves the dataset's task definitions, extracts unique Docker image names, and pulls each one sequentially. Because Docker caches pulled images locally, subsequent `docker compose up` calls by harbor will not trigger additional pulls, avoiding Docker Hub rate limits.<br/>`def pre_pull_images(dataset: str) -> None`

- `dataset`: Harbor dataset specifier (e.g. "terminal-bench@2.0").

**`run_terminal_bench`** — Run Terminal-Bench 2.0 using the harbor CLI with the sorcar agent. Before invoking harbor, checks that Docker Hub credentials are configured (to avoid unauthenticated pull rate limits) and pre-pulls all task Docker images so each unique image is fetched exactly once.<br/>`def run_terminal_bench(model: str = 'anthropic/claude-opus-4-6', dataset: str = 'terminal-bench@2.0', n_concurrent: int = 8, trials: int = 1, skip_pre_pull: bool = False) -> None`

- `model`: Model name in harbor format (provider/model).
- `dataset`: Harbor dataset specifier (e.g. "terminal-bench@2.0").
- `n_concurrent`: Number of concurrent task containers.
- `trials`: Number of attempts per task (-k flag). Use 5 for leaderboard.
- `skip_pre_pull`: If True, skip the image pre-pull step.

**`score_results`** — Print a graded summary table from a harbor results JSON file. Reads harbor's output JSON (list of task result dicts) and prints binary score, partial score (fraction of tests passed), and a summary line. Tasks with no partial score data (skipped or missing metadata) show "-" in the partial column.<br/>`def score_results(results_path: Path) -> None`

- `results_path`: Path to harbor results JSON file.

______________________________________________________________________

#### `kiss.benchmarks.terminal_bench.test_agent` — *Tests for the terminal bench harbor agent.*

##### `class FakeExecResult`

##### `class FakeEnvironment` — Minimal stand-in for BaseEnvironment.

- **exec**<br/>`async exec(command: str, **kwargs: object) -> FakeExecResult`
- **upload_file**<br/>`async upload_file(source_path: object, target_path: str) -> None`

##### `class FakeContext` — Minimal stand-in for AgentContext.

- **is_empty**<br/>`is_empty() -> bool`

##### `class TestSkipPhrases` — Verify \_SKIP_PHRASES is a non-empty tuple of strings.

- **test_skip_phrases_non_empty**<br/>`test_skip_phrases_non_empty() -> None`
- **test_skip_phrases_are_strings**<br/>`test_skip_phrases_are_strings() -> None`

##### `class TestAgentIdentity` — Agent name and version.

- **test_name**<br/>`test_name() -> None`
- **test_version_matches_package**<br/>`test_version_matches_package() -> None`

##### `class TestRunSkipsImpossibleTasks` — Verify that run() returns immediately for impossible tasks.

- **test_skip_compcert**<br/>`test_skip_compcert() -> None`
- **test_skip_windows_311**<br/>`test_skip_windows_311() -> None`
- **test_skip_ocaml_gc**<br/>`test_skip_ocaml_gc() -> None`
- **test_non_skip_task_runs_normally** — A normal task runs which-check, sorcar, then verifies.<br/>`test_non_skip_task_runs_normally() -> None`

##### `class TestSetup` — Verify setup runs the expected installation steps.

- **test_setup_three_steps**<br/>`test_setup_three_steps() -> None`
- **test_setup_aborts_on_uv_failure**<br/>`test_setup_aborts_on_uv_failure() -> None`
- **test_setup_aborts_on_pip_failure**<br/>`test_setup_aborts_on_pip_failure() -> None`

##### `class TestRunSorcarNotFound` — When sorcar is not installed, run returns early with an error.

- **test_sorcar_missing**<br/>`test_sorcar_missing() -> None`

______________________________________________________________________

#### `kiss.benchmarks.webarena` — *KISS Sorcar benchmark harness for WebArena.*

______________________________________________________________________

#### `kiss.benchmarks.webarena.agent` — *WebArena agent adapter that delegates to KISS Sorcar.*

##### `class SorcarWebArenaAgent` — Agent that runs KISS Sorcar on WebArena tasks.

**Constructor:** `SorcarWebArenaAgent(model: str | None = None, timeout: int = 600) -> None`

- `model`: LLM model name (e.g. "claude-opus-4-6").

- `timeout`: Max seconds per task before killing sorcar.

- **run_task** — Run sorcar on a single WebArena task. Writes a SYSTEM.md with WebArena-specific instructions before invoking sorcar, then scores the result against reference answers.<br/>`run_task(config_file: Path) -> dict`

  - `config_file`: Path to the WebArena task JSON config file.
  - **Returns:** Dict with task_id, answer, score, stdout, stderr, return_code.

______________________________________________________________________

#### `kiss.benchmarks.webarena.run` — *Run WebArena with the sorcar agent.*

**`run_webarena`** — Run sorcar on WebArena task configs and save results.<br/>`def run_webarena(config_dir: Path, model: str = 'claude-opus-4-6', max_tasks: int | None = None, timeout: int = 600) -> None`

- `config_dir`: Directory containing WebArena JSON task configs.
- `model`: LLM model name (e.g. "claude-opus-4-6").
- `max_tasks`: Cap for quick testing (None = all configs).
- `timeout`: Max seconds per task.

______________________________________________________________________

### `kiss.channels` — *Channel integrations for KISS agents.*

______________________________________________________________________

#### `kiss.channels._backend_utils` — *Shared helpers for channel backend polling and lifecycle management.*

##### `class ThreadedHTTPServer(ThreadingMixIn, HTTPServer)` — HTTP server with per-request threads and address reuse enabled.

**`wait_for_matching_message`** — Wait for a message matching a predicate with timeout.<br/>`def wait_for_matching_message(*, poll: Callable[[], list[dict[str, Any]]], matches: Callable[[dict[str, Any]], bool], extract_text: Callable[[dict[str, Any]], str], timeout_seconds: float, poll_interval: float) -> str | None`

- `poll`: Callable returning newly observed messages.
- `matches`: Predicate selecting the desired message.
- `extract_text`: Callable extracting the reply text from a matching message.
- `timeout_seconds`: Maximum time to wait.
- `poll_interval`: Delay between polls.
- **Returns:** Extracted reply text, or `None` on timeout.

**`drain_queue_messages`** — Drain up to `limit` messages from a queue, optionally filtering.<br/>`def drain_queue_messages(message_queue: queue.Queue[dict[str, Any]], *, limit: int, keep: Callable[[dict[str, Any]], bool] | None = None) -> list[dict[str, Any]]`

- `message_queue`: Queue containing message dicts.
- `limit`: Maximum number of kept messages to return.
- `keep`: Optional predicate deciding whether a drained message should be kept.
- **Returns:** The kept messages in dequeue order.

**`stop_http_server`** — Shut down an embedded HTTP server and join its thread.<br/>`def stop_http_server(server: HTTPServer | None, server_thread: threading.Thread | None) -> tuple[None, None]`

- `server`: HTTP server instance to stop.
- `server_thread`: Background thread running `serve_forever()`.
- **Returns:** `(None, None)` so callers can reset both attributes succinctly.

**`is_headless_environment`** — Return True when running in a headless/Docker/Linux environment. Checks in order: 1. KISS_HEADLESS env var (explicit override, "1"/"true"/"yes" → headless) 2. Presence of /.dockerenv (running inside Docker) 3. Linux with no $DISPLAY and no $WAYLAND_DISPLAY set<br/>`def is_headless_environment() -> bool`

______________________________________________________________________

#### `kiss.channels._channel_agent_utils` — *Shared helpers for channel agent backends and local config persistence.*

##### `class ToolMethodBackend` — Mixin that exposes public backend methods as agent tools.

- **connection_info** — Human-readable connection status string.<br/>`connection_info() -> str` *(property)*

- **find_channel** — Return *name* as the channel ID. Override for platforms that resolve names via an API call.<br/>`find_channel(name: str) -> str | None`

  - `name`: Channel name or identifier.
  - **Returns:** The channel identifier, or `None` if *name* is empty.

- **find_user** — Return *username* as the user ID. Override for platforms that resolve usernames via an API call.<br/>`find_user(username: str) -> str | None`

  - `username`: Username or identifier.
  - **Returns:** The user identifier, or `None` if *username* is empty.

- **join_channel** — No-op. Override for platforms that require joining a channel.<br/>`join_channel(channel_id: str) -> None`

  - `channel_id`: Channel identifier.

- **disconnect** — No-op. Override for platforms that need connection cleanup.<br/>`disconnect() -> None`

- **is_from_bot** — Return `False`. Override for platforms that can identify bot messages.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

  - `msg`: Message dict from :meth:`poll_messages`.
  - **Returns:** Whether the message was sent by the bot itself.

- **strip_bot_mention** — Return *text* unchanged. Override for platforms with bot @-mentions.<br/>`strip_bot_mention(text: str) -> str`

  - `text`: Raw message text.
  - **Returns:** Text with bot mentions removed.

- **get_tool_methods** — Return the backend's public tool methods.<br/>`get_tool_methods() -> list`

  - **Returns:** List of bound callable methods intended for LLM tool use.

##### `class ChannelConfig` — Encapsulates the 4-function config persistence pattern used by channel agents.

**Constructor:** `ChannelConfig(channel_dir: Path, required_keys: tuple[str, ...]) -> None`

- **load** — Load the config, returning `None` if missing or invalid.<br/>`load() -> dict[str, str] | None`

  - **Returns:** Loaded string dictionary, or `None`.

- **save** — Save *data* to the config file with restricted permissions.<br/>`save(data: dict[str, str]) -> None`

  - `data`: String dictionary to persist.

- **clear** — Delete the config file if it exists.<br/>`clear() -> None`

##### `class BaseChannelAgent` — Mixin for channel agent classes that provides a standard `_get_tools()`

##### `class ChannelRunner` — One-shot channel message runner.

**Constructor:** `ChannelRunner(backend: Any, channel_name: str, agent_name: str, extra_tools: list | None = None, model_name: str = '', max_budget: float = 5.0, work_dir: str = '', allow_users: list[str] | None = None) -> None`

- **run_once** — Check for pending messages, process them, and exit. Connects to the backend, joins the configured channel, retrieves recent messages, filters to allowed users, skips messages the bot has already replied to, and runs a StatefulSorcarAgent for each pending message. Each message is processed synchronously.<br/>`run_once() -> int`
  - **Returns:** Number of messages processed.

**`load_json_config`** — Load a JSON config file containing string values.<br/>`def load_json_config(path: Path, required_keys: tuple[str, ...]) -> dict[str, str] | None`

- `path`: Config file path.
- `required_keys`: Keys that must be present and non-empty.
- **Returns:** Loaded string dictionary, or `None` if the file is missing, malformed, not a dict, or lacks a required key.

**`save_json_config`** — Save a JSON config file with restricted permissions.<br/>`def save_json_config(path: Path, data: dict[str, str]) -> None`

- `path`: Config file path.
- `data`: String dictionary to persist.

**`clear_json_config`** — Delete a JSON config file if it exists.<br/>`def clear_json_config(path: Path) -> None`

- `path`: Config file path.

**`channel_main`** — Standard CLI entry point shared by all channel agents. Handles argument parsing and either one-shot poll mode (when `--channel` is given) or interactive mode (when `-t` is given). Each channel agent's `main()` delegates to this function.<br/>`def channel_main(agent_cls: type, cli_name: str, *, channel_name: str = '', make_backend: Callable[..., Any] | None = None, extra_usage: str = '') -> None`

- `agent_cls`: The channel Agent class to instantiate (e.g. `SlackAgent`).
- `cli_name`: CLI command name for the usage message (e.g. `"kiss-slack"`).
- `channel_name`: Human-readable channel name (e.g. `"Slack"`). Used in status messages and agent naming.
- `make_backend`: Factory that creates and configures a backend for poll mode. May accept a `workspace` keyword argument; if so, the `--workspace` CLI value is forwarded. Should call `sys.exit(1)` if required config is missing. Pass `None` to disable poll mode.
- `extra_usage`: Additional usage flags to append to the usage line (e.g. `"[--list-workspaces]"`).

______________________________________________________________________

#### `kiss.channels.bluebubbles_agent` — *BlueBubbles Agent — StatefulSorcarAgent extension with BlueBubbles REST API tools.*

##### `class BlueBubblesChannelBackend(ToolMethodBackend)` — Channel backend for BlueBubbles REST API.

**Constructor:** `BlueBubblesChannelBackend() -> None`

- **connect** — Connect to BlueBubbles server.<br/>`connect() -> bool`

- **poll_messages** — Poll BlueBubbles for new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a BlueBubbles message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **list_chats** — List recent iMessage conversations.<br/>`list_chats(limit: int = 25, offset: int = 0) -> str`

  - `limit`: Maximum chats to return. Default: 25.
  - `offset`: Pagination offset. Default: 0.
  - **Returns:** JSON string with chat list.

- **get_chat** — Get a specific iMessage conversation.<br/>`get_chat(chat_guid: str) -> str`

  - `chat_guid`: Chat GUID (from list_chats).
  - **Returns:** JSON string with chat details.

- **get_chat_messages** — Get messages from a specific conversation.<br/>`get_chat_messages(chat_guid: str, limit: int = 25, before: str = '', after: str = '') -> str`

  - `chat_guid`: Chat GUID.
  - `limit`: Maximum messages to return. Default: 25.
  - `before`: Return messages before this timestamp (ms).
  - `after`: Return messages after this timestamp (ms).
  - **Returns:** JSON string with message list.

- **post_message** — Send a message to an iMessage conversation.<br/>`post_message(chat_guid: str, text: str) -> str`

  - `chat_guid`: Chat GUID to send to.
  - `text`: Message text.
  - **Returns:** JSON string with ok status.

- **get_server_info** — Get BlueBubbles server information.<br/>`get_server_info() -> str`

  - **Returns:** JSON string with server info.

- **mark_chat_read** — Mark a chat as read.<br/>`mark_chat_read(chat_guid: str) -> str`

  - `chat_guid`: Chat GUID to mark as read.
  - **Returns:** JSON string with ok status.

##### `class BlueBubblesAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with BlueBubbles REST API tools (macOS only).

**Constructor:** `BlueBubblesAgent() -> None`

______________________________________________________________________

#### `kiss.channels.discord_agent` — *Discord Agent — StatefulSorcarAgent extension with Discord REST API tools.*

##### `class DiscordChannelBackend(ToolMethodBackend)` — Channel backend for Discord REST API v10.

**Constructor:** `DiscordChannelBackend() -> None`

- **connect** — Authenticate with Discord using the stored bot token.<br/>`connect() -> bool`

- **find_channel** — Find a channel by name or numeric ID. If *name* is already a numeric snowflake ID, returns it as-is. Otherwise queries all guilds for a channel matching the name.<br/>`find_channel(name: str) -> str | None`

  - `name`: Channel name or numeric ID.
  - **Returns:** The channel snowflake ID string, or None if not found.

- **poll_messages** — Poll for new Discord messages using REST API.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Discord message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **disconnect** — Release Discord backend state before stop or reconnect.<br/>`disconnect() -> None`

- **list_guilds** — List guilds (servers) the bot is a member of.<br/>`list_guilds(limit: int = 100) -> str`

  - `limit`: Maximum guilds to return (1-200). Default: 100.
  - **Returns:** JSON string with guild list (id, name, icon).

- **list_channels** — List channels in a guild.<br/>`list_channels(guild_id: str, channel_type: str = '') -> str`

  - `guild_id`: Guild (server) ID.
  - `channel_type`: Optional filter by type (0=text, 2=voice, 4=category).
  - **Returns:** JSON string with channel list (id, name, type, topic).

- **get_channel** — Get information about a channel.<br/>`get_channel(channel_id: str) -> str`

  - `channel_id`: Channel ID.
  - **Returns:** JSON string with channel details.

- **get_channel_messages** — Get messages from a channel.<br/>`get_channel_messages(channel_id: str, limit: int = 50, before: str = '', after: str = '') -> str`

  - `channel_id`: Channel ID.
  - `limit`: Number of messages (1-100). Default: 50.
  - `before`: Get messages before this message ID.
  - `after`: Get messages after this message ID.
  - **Returns:** JSON string with message list.

- **post_message** — Send a message to a Discord channel.<br/>`post_message(channel_id: str, content: str, tts: bool = False, reply_to: str = '') -> str`

  - `channel_id`: Channel ID.
  - `content`: Message text (up to 2000 chars).
  - `tts`: Text-to-speech flag. Default: False.
  - `reply_to`: Optional message ID to reply to.
  - **Returns:** JSON string with ok status and message id.

- **edit_message** — Edit an existing Discord message.<br/>`edit_message(channel_id: str, message_id: str, content: str) -> str`

  - `channel_id`: Channel ID.
  - `message_id`: Message ID.
  - `content`: New content.
  - **Returns:** JSON string with ok status.

- **delete_message** — Delete a Discord message.<br/>`delete_message(channel_id: str, message_id: str) -> str`

  - `channel_id`: Channel ID.
  - `message_id`: Message ID to delete.
  - **Returns:** JSON string with ok status.

- **add_reaction** — Add a reaction to a message.<br/>`add_reaction(channel_id: str, message_id: str, emoji: str) -> str`

  - `channel_id`: Channel ID.
  - `message_id`: Message ID.
  - `emoji`: Emoji (e.g. "👍" or "name:id" for custom emojis).
  - **Returns:** JSON string with ok status.

- **create_thread** — Create a thread from a message.<br/>`create_thread(channel_id: str, message_id: str, name: str, auto_archive_duration: int = 1440) -> str`

  - `channel_id`: Channel ID.
  - `message_id`: Message ID to create thread from.
  - `name`: Thread name.
  - `auto_archive_duration`: Minutes before auto-archive (60/1440/4320/10080).
  - **Returns:** JSON string with thread id and name.

- **list_guild_members** — List members of a guild.<br/>`list_guild_members(guild_id: str, limit: int = 100, after: str = '') -> str`

  - `guild_id`: Guild ID.
  - `limit`: Max members to return (1-1000). Default: 100.
  - `after`: User ID to start after (for pagination).
  - **Returns:** JSON string with member list.

- **create_invite** — Create an invite link for a channel.<br/>`create_invite(channel_id: str, max_age: int = 86400, max_uses: int = 0) -> str`

  - `channel_id`: Channel ID.
  - `max_age`: Invite expiry in seconds (0 = never). Default: 86400 (1 day).
  - `max_uses`: Maximum uses (0 = unlimited). Default: 0.
  - **Returns:** JSON string with invite code and URL.

##### `class DiscordAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Discord REST API tools.

**Constructor:** `DiscordAgent() -> None`

- **run** — Run with Discord-specific system prompt encouraging browser-based auth.<br/>`run(**kwargs: Any) -> str`

______________________________________________________________________

#### `kiss.channels.feishu_agent` — *Feishu/Lark Agent — StatefulSorcarAgent extension with Feishu Open Platform tools.*

##### `class FeishuChannelBackend(ToolMethodBackend)` — Channel backend for Feishu/Lark Open Platform.

**Constructor:** `FeishuChannelBackend() -> None`

- **connect** — Authenticate with Feishu using stored app credentials.<br/>`connect() -> bool`

- **poll_messages** — Poll Feishu chat for new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Feishu message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **send_text_message** — Send a text message to a Feishu chat or user.<br/>`send_text_message(receive_id: str, text: str, receive_id_type: str = 'chat_id') -> str`

  - `receive_id`: Chat ID, user ID, or open ID depending on receive_id_type.
  - `text`: Message text.
  - `receive_id_type`: "chat_id", "user_id", "open_id", or "email". Default: "chat_id".
  - **Returns:** JSON string with ok status and message id.

- **reply_message** — Reply to an existing Feishu message.<br/>`reply_message(message_id: str, text: str) -> str`

  - `message_id`: ID of the message to reply to.
  - `text`: Reply text.
  - **Returns:** JSON string with ok status and reply message id.

- **delete_message** — Delete a Feishu message.<br/>`delete_message(message_id: str) -> str`

  - `message_id`: Message ID to delete.
  - **Returns:** JSON string with ok status.

- **list_messages** — List messages in a Feishu chat.<br/>`list_messages(container_id: str, start_time: str = '', end_time: str = '', page_size: int = 20) -> str`

  - `container_id`: Chat ID.
  - `start_time`: Start Unix timestamp (seconds). Optional.
  - `end_time`: End Unix timestamp (seconds). Optional.
  - `page_size`: Maximum messages to return. Default: 20.
  - **Returns:** JSON string with message list.

- **list_chats** — List Feishu chats the bot is a member of.<br/>`list_chats(page_size: int = 20, page_token: str = '') -> str`

  - `page_size`: Maximum chats to return. Default: 20.
  - `page_token`: Pagination token.
  - **Returns:** JSON string with chat list (chat_id, name, description).

- **get_chat** — Get information about a Feishu chat.<br/>`get_chat(chat_id: str) -> str`

  - `chat_id`: Chat ID.
  - **Returns:** JSON string with chat details.

- **get_user_info** — Get Feishu user information.<br/>`get_user_info(user_id: str, user_id_type: str = 'open_id') -> str`

  - `user_id`: User ID.
  - `user_id_type`: ID type ("open_id", "user_id", "union_id"). Default: "open_id".
  - **Returns:** JSON string with user info.

##### `class FeishuAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Feishu/Lark Open Platform tools.

**Constructor:** `FeishuAgent() -> None`

______________________________________________________________________

#### `kiss.channels.gmail_agent` — *Gmail Agent — StatefulSorcarAgent extension with Gmail API tools.*

##### `class GmailChannelBackend(ToolMethodBackend)` — Channel backend for Gmail.

**Constructor:** `GmailChannelBackend() -> None`

- **connect** — Authenticate with Gmail using stored OAuth2 credentials.<br/>`connect() -> bool`

  - **Returns:** True on success, False on failure.

- **find_channel** — Find a Gmail label by name (used as channel ID).<br/>`find_channel(name: str) -> str | None`

  - `name`: Label name to search for.
  - **Returns:** Label ID string, or None if not found.

- **poll_messages** — Poll Gmail inbox for new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

  - `channel_id`: Label ID to poll (use "INBOX" for inbox).
  - `oldest`: History ID or timestamp string for incremental polling.
  - `limit`: Maximum messages to return.
  - **Returns:** Tuple of (messages, updated_oldest). Each message dict has: ts (date), user (from address), text (body).

- **send_message** — Send an email (reply to a thread if thread_ts provided).<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

  - `channel_id`: Recipient email address.
  - `text`: Email body text.
  - `thread_ts`: Thread ID to reply to (optional).

- **wait_for_reply** — Poll a Gmail thread for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

  - `channel_id`: Label ID (unused for Gmail).
  - `thread_ts`: Thread ID to poll.
  - `user_id`: Email address of expected sender.
  - **Returns:** The text of the user's reply.

- **get_profile** — Get the current user's Gmail profile.<br/>`get_profile() -> str`

  - **Returns:** JSON string with email address, messages total, threads total, and history ID.

- **list_messages** — List messages in the user's mailbox.<br/>`list_messages(query: str = '', max_results: int = 20, page_token: str = '', label_ids: str = '') -> str`

  - `query`: Gmail search query (same syntax as Gmail search box). Examples: "is:unread", "from:alice@example.com", "subject:meeting", "newer_than:1d", "has:attachment".
  - `max_results`: Maximum number of messages to return (1-500). Default: 20.
  - `page_token`: Page token for pagination from a previous response.
  - `label_ids`: Comma-separated label IDs to filter by (e.g. "INBOX", "UNREAD", "STARRED").
  - **Returns:** JSON string with message IDs, snippet, and pagination token. Use get_message() with the ID to read full content.

- **get_message** — Get a specific message by ID.<br/>`get_message(message_id: str, format: str = 'full') -> str`

  - `message_id`: The message ID (from list_messages).
  - `format`: Response format. Options: "full" — full message with parsed payload (default). "metadata" — headers only (faster). "raw" — raw RFC 2822 message. "minimal" — just IDs, labels, snippet.
  - **Returns:** JSON string with message headers, body text, labels, and attachment info.

- **send_email** — Send an email message.<br/>`send_email(to: str, subject: str, body: str, cc: str = '', bcc: str = '', html: bool = False) -> str`

  - `to`: Recipient email address(es), comma-separated.
  - `subject`: Email subject line.
  - `body`: Email body text (plain text or HTML).
  - `cc`: CC recipients, comma-separated. Optional.
  - `bcc`: BCC recipients, comma-separated. Optional.
  - `html`: If True, body is treated as HTML. Default: False.
  - **Returns:** JSON string with ok status and the sent message ID.

- **reply_to_message** — Reply to an existing email message.<br/>`reply_to_message(message_id: str, body: str, reply_all: bool = False, html: bool = False) -> str`

  - `message_id`: ID of the message to reply to.
  - `body`: Reply body text (plain text or HTML).
  - `reply_all`: If True, reply to all recipients. Default: False.
  - `html`: If True, body is treated as HTML. Default: False.
  - **Returns:** JSON string with ok status and the reply message ID.

- **create_draft** — Create a draft email.<br/>`create_draft(to: str, subject: str, body: str, cc: str = '', bcc: str = '', html: bool = False) -> str`

  - `to`: Recipient email address(es), comma-separated.
  - `subject`: Email subject line.
  - `body`: Email body text (plain text or HTML).
  - `cc`: CC recipients, comma-separated. Optional.
  - `bcc`: BCC recipients, comma-separated. Optional.
  - `html`: If True, body is treated as HTML. Default: False.
  - **Returns:** JSON string with ok status and draft ID.

- **trash_message** — Move a message to the trash.<br/>`trash_message(message_id: str) -> str`

  - `message_id`: ID of the message to trash.
  - **Returns:** JSON string with ok status.

- **untrash_message** — Remove a message from the trash.<br/>`untrash_message(message_id: str) -> str`

  - `message_id`: ID of the message to untrash.
  - **Returns:** JSON string with ok status.

- **delete_message** — Permanently delete a message (cannot be undone).<br/>`delete_message(message_id: str) -> str`

  - `message_id`: ID of the message to permanently delete.
  - **Returns:** JSON string with ok status.

- **modify_labels** — Modify labels on a message (star, archive, mark read/unread, etc.). Common label IDs: INBOX, UNREAD, STARRED, IMPORTANT, SPAM, TRASH, CATEGORY_PERSONAL, CATEGORY_SOCIAL, CATEGORY_PROMOTIONS. To archive: remove "INBOX". To mark as read: remove "UNREAD". To star: add "STARRED".<br/>`modify_labels(message_id: str, add_label_ids: str = '', remove_label_ids: str = '') -> str`

  - `message_id`: ID of the message to modify.
  - `add_label_ids`: Comma-separated label IDs to add.
  - `remove_label_ids`: Comma-separated label IDs to remove.
  - **Returns:** JSON string with ok status and updated label list.

- **list_labels** — List all labels in the user's mailbox.<br/>`list_labels() -> str`

  - **Returns:** JSON string with label list (id, name, type).

- **create_label** — Create a new label.<br/>`create_label(name: str, text_color: str = '', background_color: str = '') -> str`

  - `name`: Label name (e.g. "Projects/Important"). Use "/" for nested labels.
  - `text_color`: Optional hex text color (e.g. "#000000").
  - `background_color`: Optional hex background color (e.g. "#16a765").
  - **Returns:** JSON string with the new label's id and name.

- **get_attachment** — Download a message attachment.<br/>`get_attachment(message_id: str, attachment_id: str) -> str`

  - `message_id`: ID of the message containing the attachment.
  - `attachment_id`: Attachment ID (from get_message response).
  - **Returns:** JSON string with base64-encoded attachment data and size.

- **get_thread** — Get all messages in an email thread/conversation.<br/>`get_thread(thread_id: str) -> str`

  - `thread_id`: Thread ID (from list_messages or get_message).
  - **Returns:** JSON string with all messages in the thread.

##### `class GmailAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Gmail API tools.

**Constructor:** `GmailAgent() -> None`

- **run** — Run with Gmail-specific system prompt encouraging browser-based auth.<br/>`run(**kwargs: Any) -> str`

______________________________________________________________________

#### `kiss.channels.googlechat_agent` — *Google Chat Agent — StatefulSorcarAgent extension with Google Chat API tools.*

##### `class GoogleChatChannelBackend(ToolMethodBackend)` — Channel backend for Google Chat API.

**Constructor:** `GoogleChatChannelBackend() -> None`

- **connect** — Authenticate with Google Chat.<br/>`connect() -> bool`

- **find_channel** — Find a Google Chat space by display name.<br/>`find_channel(name: str) -> str | None`

- **poll_messages** — Poll a Google Chat space for new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Google Chat message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **list_spaces** — List Google Chat spaces (rooms and DMs).<br/>`list_spaces(page_size: int = 20, page_token: str = '') -> str`

  - `page_size`: Maximum spaces to return. Default: 20.
  - `page_token`: Pagination token from a previous response.
  - **Returns:** JSON string with space list (name, displayName, type).

- **get_space** — Get information about a Google Chat space.<br/>`get_space(space_name: str) -> str`

  - `space_name`: Space resource name (e.g. "spaces/ABCDEF").
  - **Returns:** JSON string with space details.

- **list_members** — List members of a Google Chat space.<br/>`list_members(space_name: str, page_size: int = 20, page_token: str = '') -> str`

  - `space_name`: Space resource name.
  - `page_size`: Maximum members to return. Default: 20.
  - `page_token`: Pagination token.
  - **Returns:** JSON string with member list.

- **list_messages** — List messages in a Google Chat space.<br/>`list_messages(space_name: str, page_size: int = 20, page_token: str = '', filter: str = '') -> str`

  - `space_name`: Space resource name (e.g. "spaces/ABCDEF").
  - `page_size`: Maximum messages to return. Default: 20.
  - `page_token`: Pagination token.
  - `filter`: Optional filter (e.g. 'createTime > "2024-01-01T00:00:00Z"').
  - **Returns:** JSON string with message list.

- **get_message** — Get a specific Google Chat message.<br/>`get_message(message_name: str) -> str`

  - `message_name`: Message resource name (e.g. "spaces/X/messages/Y").
  - **Returns:** JSON string with message details.

- **post_message** — Send a message to a Google Chat space.<br/>`post_message(space_name: str, text: str, thread_key: str = '') -> str`

  - `space_name`: Space resource name (e.g. "spaces/ABCDEF").
  - `text`: Message text.
  - `thread_key`: Optional thread key to reply in an existing thread.
  - **Returns:** JSON string with ok status and message name.

- **update_message** — Update an existing Google Chat message.<br/>`update_message(message_name: str, text: str) -> str`

  - `message_name`: Message resource name.
  - `text`: New message text.
  - **Returns:** JSON string with ok status.

- **delete_message** — Delete a Google Chat message.<br/>`delete_message(message_name: str) -> str`

  - `message_name`: Message resource name.
  - **Returns:** JSON string with ok status.

- **create_space** — Create a new Google Chat space.<br/>`create_space(display_name: str, space_type: str = 'SPACE') -> str`

  - `display_name`: Space display name.
  - `space_type`: Space type ("SPACE" or "GROUP_CHAT"). Default: "SPACE".
  - **Returns:** JSON string with space name and display name.

##### `class GoogleChatAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Google Chat API tools.

**Constructor:** `GoogleChatAgent() -> None`

______________________________________________________________________

#### `kiss.channels.imessage_agent` — *iMessage Agent — StatefulSorcarAgent extension with iMessage tools via AppleScript.*

##### `class IMessageChannelBackend(ToolMethodBackend)` — Channel backend for iMessage via AppleScript.

**Constructor:** `IMessageChannelBackend() -> None`

- **connect** — Check macOS and Messages.app availability.<br/>`connect() -> bool`

- **poll_messages** — Poll iMessage via AppleScript (basic implementation).<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send an iMessage.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Reply waiting is not supported for AppleScript-based iMessage.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **send_imessage** — Send an iMessage or SMS to a recipient.<br/>`send_imessage(recipient: str, text: str, service: str = 'iMessage') -> str`

  - `recipient`: Phone number or Apple ID email to send to.
  - `text`: Message text.
  - `service`: "iMessage" or "SMS". Default: "iMessage".
  - **Returns:** JSON string with ok status.

- **send_attachment** — Send a file attachment via iMessage.<br/>`send_attachment(recipient: str, file_path: str, service: str = 'iMessage') -> str`

  - `recipient`: Phone number or Apple ID email.
  - `file_path`: Absolute path to the file to send.
  - `service`: "iMessage" or "SMS". Default: "iMessage".
  - **Returns:** JSON string with ok status.

- **list_conversations** — List recent iMessage conversations.<br/>`list_conversations() -> str`

  - **Returns:** JSON string with conversation list.

- **get_messages** — Get recent messages with a recipient (basic implementation).<br/>`get_messages(recipient: str, limit: int = 20) -> str`

  - `recipient`: Phone number or email to get messages for.
  - `limit`: Maximum messages to return. Default: 20.
  - **Returns:** JSON string with message list (basic).

##### `class IMessageAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with iMessage tools (macOS only).

**Constructor:** `IMessageAgent() -> None`

______________________________________________________________________

#### `kiss.channels.irc_agent` — *IRC Agent — StatefulSorcarAgent extension with IRC tools.*

##### `class IRCChannelBackend(ToolMethodBackend)` — Channel backend for IRC via raw socket.

**Constructor:** `IRCChannelBackend() -> None`

- **connect** — Connect to IRC server.<br/>`connect() -> bool`

- **join_channel** — Join an IRC channel.<br/>`join_channel(channel_id: str) -> None`

- **poll_messages** — Return buffered IRC messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send an IRC PRIVMSG.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **disconnect** — Close the IRC socket and join the reader thread.<br/>`disconnect() -> None`

- **is_from_bot** — Check if message is from the bot.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **strip_bot_mention** — Remove bot mention from text.<br/>`strip_bot_mention(text: str) -> str`

- **connect_irc** — Connect to an IRC server.<br/>`connect_irc(server: str, port: int = 6667, nick: str = 'KISSBot', realname: str = 'KISS Agent', password: str = '', use_tls: bool = False) -> str`

  - `server`: IRC server hostname or IP.
  - `port`: Server port. Default: 6667.
  - `nick`: Nickname to use. Default: "KISSBot".
  - `realname`: Real name. Default: "KISS Agent".
  - `password`: Server password. Optional.
  - `use_tls`: Use TLS encryption. Default: False.
  - **Returns:** JSON string with ok status.

- **join_irc_channel** — Join an IRC channel.<br/>`join_irc_channel(channel: str) -> str`

  - `channel`: Channel name (e.g. "#general").
  - **Returns:** JSON string with ok status.

- **leave_channel** — Leave an IRC channel.<br/>`leave_channel(channel: str, reason: str = '') -> str`

  - `channel`: Channel name.
  - `reason`: Optional leave reason.
  - **Returns:** JSON string with ok status.

- **post_message** — Send a message to an IRC channel or user.<br/>`post_message(channel_or_nick: str, text: str) -> str`

  - `channel_or_nick`: Target channel (e.g. "#general") or nick.
  - `text`: Message text.
  - **Returns:** JSON string with ok status.

- **send_notice** — Send a NOTICE to an IRC channel or user.<br/>`send_notice(channel_or_nick: str, text: str) -> str`

  - `channel_or_nick`: Target channel or nick.
  - `text`: Notice text.
  - **Returns:** JSON string with ok status.

- **get_topic** — Get the topic of an IRC channel.<br/>`get_topic(channel: str) -> str`

  - `channel`: Channel name.
  - **Returns:** JSON string with ok status (topic comes via server response).

- **set_topic** — Set the topic of an IRC channel.<br/>`set_topic(channel: str, topic: str) -> str`

  - `channel`: Channel name.
  - `topic`: New topic text.
  - **Returns:** JSON string with ok status.

- **kick_user** — Kick a user from an IRC channel.<br/>`kick_user(channel: str, nick: str, reason: str = '') -> str`

  - `channel`: Channel name.
  - `nick`: Nickname to kick.
  - `reason`: Optional kick reason.
  - **Returns:** JSON string with ok status.

- **whois** — Get WHOIS information about a user.<br/>`whois(nick: str) -> str`

  - `nick`: Nickname to look up.
  - **Returns:** JSON string with ok status (data comes via server response).

- **identify_nickserv** — Identify to NickServ.<br/>`identify_nickserv(password: str) -> str`

  - `password`: NickServ password.
  - **Returns:** JSON string with ok status.

##### `class IRCAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with IRC tools.

**Constructor:** `IRCAgent() -> None`

______________________________________________________________________

#### `kiss.channels.line_agent` — *LINE Agent — StatefulSorcarAgent extension with LINE Messaging API tools.*

##### `class LineChannelBackend(ToolMethodBackend)` — Channel backend for LINE Messaging API.

**Constructor:** `LineChannelBackend() -> None`

- **connect** — Authenticate with LINE and start webhook server.<br/>`connect() -> bool`

- **poll_messages** — Drain the webhook message queue.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a LINE push message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **disconnect** — Stop the embedded webhook server and release backend resources.<br/>`disconnect() -> None`

- **push_text_message** — Send a push text message to a LINE user or group.<br/>`push_text_message(to: str, text: str) -> str`

  - `to`: Target user ID, group ID, or room ID.
  - `text`: Message text (up to 5000 characters).
  - **Returns:** JSON string with ok status.

- **reply_message** — Reply to a message using the reply token.<br/>`reply_message(reply_token: str, messages_json: str) -> str`

  - `reply_token`: Reply token from an inbound message event.
  - `messages_json`: JSON array of message objects. Example: '[{"type":"text","text":"Hello!"}]'
  - **Returns:** JSON string with ok status.

- **get_profile** — Get a LINE user's profile.<br/>`get_profile(user_id: str) -> str`

  - `user_id`: LINE user ID.
  - **Returns:** JSON string with user profile (displayName, pictureUrl, statusMessage).

- **get_quota** — Get the LINE messaging quota for the current month.<br/>`get_quota() -> str`

  - **Returns:** JSON string with quota information.

- **leave_group** — Leave a LINE group.<br/>`leave_group(group_id: str) -> str`

  - `group_id`: Group ID to leave.
  - **Returns:** JSON string with ok status.

- **push_image_message** — Send a push image message.<br/>`push_image_message(to: str, image_url: str, preview_url: str) -> str`

  - `to`: Target user ID, group ID, or room ID.
  - `image_url`: URL of the full-size image.
  - `preview_url`: URL of the preview image.
  - **Returns:** JSON string with ok status.

##### `class LineAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with LINE Messaging API tools.

**Constructor:** `LineAgent() -> None`

______________________________________________________________________

#### `kiss.channels.matrix_agent` — *Matrix Agent — StatefulSorcarAgent extension with Matrix protocol tools.*

##### `class MatrixChannelBackend(ToolMethodBackend)` — Channel backend for Matrix via matrix-nio.

**Constructor:** `MatrixChannelBackend() -> None`

- **connect** — Authenticate with Matrix using stored config.<br/>`connect() -> bool`

- **join_channel** — Join a Matrix room.<br/>`join_channel(channel_id: str) -> None`

- **poll_messages** — Poll for new Matrix messages via sync.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Matrix text message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **is_from_bot** — Check if message is from the bot.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **list_rooms** — List joined Matrix rooms.<br/>`list_rooms() -> str`

  - **Returns:** JSON string with room list (id, name, topic).

- **join_room** — Join a Matrix room.<br/>`join_room(room_id_or_alias: str) -> str`

  - `room_id_or_alias`: Room ID (!room:server.org) or alias (#room:server.org).
  - **Returns:** JSON string with ok status and room id.

- **leave_room** — Leave a Matrix room.<br/>`leave_room(room_id: str) -> str`

  - `room_id`: Room ID to leave.
  - **Returns:** JSON string with ok status.

- **send_text_message** — Send a text message to a Matrix room.<br/>`send_text_message(room_id: str, text: str) -> str`

  - `room_id`: Room ID.
  - `text`: Message text.
  - **Returns:** JSON string with ok status and event id.

- **send_notice** — Send a notice (bot message) to a Matrix room.<br/>`send_notice(room_id: str, text: str) -> str`

  - `room_id`: Room ID.
  - `text`: Notice text.
  - **Returns:** JSON string with ok status and event id.

- **get_room_members** — Get members of a Matrix room.<br/>`get_room_members(room_id: str) -> str`

  - `room_id`: Room ID.
  - **Returns:** JSON string with member list.

- **invite_user** — Invite a user to a Matrix room.<br/>`invite_user(room_id: str, user_id: str) -> str`

  - `room_id`: Room ID.
  - `user_id`: User ID to invite (@user:server.org).
  - **Returns:** JSON string with ok status.

- **kick_user** — Kick a user from a Matrix room.<br/>`kick_user(room_id: str, user_id: str, reason: str = '') -> str`

  - `room_id`: Room ID.
  - `user_id`: User ID to kick.
  - `reason`: Optional reason for kick.
  - **Returns:** JSON string with ok status.

- **create_room** — Create a new Matrix room.<br/>`create_room(name: str = '', topic: str = '', is_public: bool = False, alias: str = '') -> str`

  - `name`: Room display name.
  - `topic`: Room topic.
  - `is_public`: Whether the room is publicly joinable. Default: False.
  - `alias`: Optional local alias (without server part).
  - **Returns:** JSON string with room id.

- **get_profile** — Get a Matrix user's profile.<br/>`get_profile(user_id: str) -> str`

  - `user_id`: User ID (@user:server.org).
  - **Returns:** JSON string with display name and avatar.

##### `class MatrixAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Matrix protocol tools.

**Constructor:** `MatrixAgent() -> None`

______________________________________________________________________

#### `kiss.channels.mattermost_agent` — *Mattermost Agent — StatefulSorcarAgent extension with Mattermost REST API tools.*

##### `class MattermostChannelBackend(ToolMethodBackend)` — Channel backend for Mattermost REST API.

**Constructor:** `MattermostChannelBackend() -> None`

- **connect** — Authenticate with Mattermost using stored config.<br/>`connect() -> bool`

- **poll_messages** — Poll Mattermost channel for new posts.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Mattermost post.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **list_teams** — List Mattermost teams.<br/>`list_teams() -> str`

  - **Returns:** JSON string with team list (id, name, display_name).

- **list_channels** — List channels in a Mattermost team.<br/>`list_channels(team_id: str, page: int = 0, per_page: int = 60) -> str`

  - `team_id`: Team ID.
  - `page`: Page number for pagination. Default: 0.
  - `per_page`: Channels per page. Default: 60.
  - **Returns:** JSON string with channel list.

- **get_channel** — Get information about a Mattermost channel.<br/>`get_channel(channel_id: str) -> str`

  - `channel_id`: Channel ID.
  - **Returns:** JSON string with channel details.

- **list_channel_posts** — List posts in a Mattermost channel.<br/>`list_channel_posts(channel_id: str, page: int = 0, per_page: int = 30) -> str`

  - `channel_id`: Channel ID.
  - `page`: Page number. Default: 0.
  - `per_page`: Posts per page. Default: 30.
  - **Returns:** JSON string with post list.

- **create_post** — Create a post in a Mattermost channel.<br/>`create_post(channel_id: str, message: str, root_id: str = '', file_ids: str = '') -> str`

  - `channel_id`: Channel ID.
  - `message`: Post message text.
  - `root_id`: Root post ID if this is a reply.
  - `file_ids`: Comma-separated file IDs to attach.
  - **Returns:** JSON string with ok status and post id.

- **delete_post** — Delete a Mattermost post.<br/>`delete_post(post_id: str) -> str`

  - `post_id`: Post ID to delete.
  - **Returns:** JSON string with ok status.

- **get_user** — Get a Mattermost user's information.<br/>`get_user(user_id_or_username: str) -> str`

  - `user_id_or_username`: User ID or username. Use "me" for current user.
  - **Returns:** JSON string with user details.

- **list_users** — List Mattermost users.<br/>`list_users(page: int = 0, per_page: int = 60, in_team: str = '', in_channel: str = '') -> str`

  - `page`: Page number. Default: 0.
  - `per_page`: Users per page. Default: 60.
  - `in_team`: Optional team ID to filter by.
  - `in_channel`: Optional channel ID to filter by.
  - **Returns:** JSON string with user list.

- **create_direct_message_channel** — Create a direct message channel between two users.<br/>`create_direct_message_channel(user1_id: str, user2_id: str) -> str`

  - `user1_id`: First user ID.
  - `user2_id`: Second user ID.
  - **Returns:** JSON string with channel id.

- **add_reaction** — Add a reaction to a post.<br/>`add_reaction(user_id: str, post_id: str, emoji_name: str) -> str`

  - `user_id`: User ID adding the reaction.
  - `post_id`: Post ID.
  - `emoji_name`: Emoji name (without colons, e.g. "thumbsup").
  - **Returns:** JSON string with ok status.

##### `class MattermostAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Mattermost REST API tools.

**Constructor:** `MattermostAgent() -> None`

______________________________________________________________________

#### `kiss.channels.msteams_agent` — *Microsoft Teams Agent — StatefulSorcarAgent extension with MS Teams Graph API tools.*

##### `class MSTeamsChannelBackend(ToolMethodBackend)` — Channel backend for Microsoft Teams via Graph API.

**Constructor:** `MSTeamsChannelBackend() -> None`

- **connect** — Authenticate with Microsoft Graph API.<br/>`connect() -> bool`

- **poll_messages** — Poll MS Teams channel for new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Teams channel message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **is_from_bot** — Check if a message is from the bot.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **list_teams** — List Microsoft Teams the bot/user is a member of.<br/>`list_teams(limit: int = 20) -> str`

  - `limit`: Maximum teams to return. Default: 20.
  - **Returns:** JSON string with team list (id, displayName, description).

- **get_team** — Get details about a Microsoft Team.<br/>`get_team(team_id: str) -> str`

  - `team_id`: Team ID.
  - **Returns:** JSON string with team details.

- **list_channels** — List channels in a Microsoft Team.<br/>`list_channels(team_id: str) -> str`

  - `team_id`: Team ID.
  - **Returns:** JSON string with channel list (id, displayName, membershipType).

- **list_channel_messages** — List messages in a Teams channel.<br/>`list_channel_messages(team_id: str, channel_id: str, top: int = 20) -> str`

  - `team_id`: Team ID.
  - `channel_id`: Channel ID.
  - `top`: Maximum messages to return. Default: 20.
  - **Returns:** JSON string with message list.

- **post_channel_message** — Post a message to a Teams channel.<br/>`post_channel_message(team_id: str, channel_id: str, content: str, content_type: str = 'html') -> str`

  - `team_id`: Team ID.
  - `channel_id`: Channel ID.
  - `content`: Message content.
  - `content_type`: "html" or "text". Default: "html".
  - **Returns:** JSON string with ok status and message id.

- **reply_to_message** — Reply to a Teams channel message.<br/>`reply_to_message(team_id: str, channel_id: str, message_id: str, content: str) -> str`

  - `team_id`: Team ID.
  - `channel_id`: Channel ID.
  - `message_id`: Parent message ID.
  - `content`: Reply content.
  - **Returns:** JSON string with ok status and reply id.

- **list_chats** — List chats for the authenticated user.<br/>`list_chats(top: int = 20) -> str`

  - `top`: Maximum chats to return. Default: 20.
  - **Returns:** JSON string with chat list.

- **post_chat_message** — Post a message to a Teams chat.<br/>`post_chat_message(chat_id: str, content: str, content_type: str = 'text') -> str`

  - `chat_id`: Chat ID.
  - `content`: Message content.
  - `content_type`: "text" or "html". Default: "text".
  - **Returns:** JSON string with ok status and message id.

- **list_team_members** — List members of a Microsoft Team.<br/>`list_team_members(team_id: str, top: int = 50) -> str`

  - `team_id`: Team ID.
  - `top`: Maximum members to return. Default: 50.
  - **Returns:** JSON string with member list.

##### `class MSTeamsAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Microsoft Teams Graph API tools.

**Constructor:** `MSTeamsAgent() -> None`

______________________________________________________________________

#### `kiss.channels.nextcloud_talk_agent` — *Nextcloud Talk Agent — StatefulSorcarAgent extension with Nextcloud Talk API tools.*

##### `class NextcloudTalkChannelBackend(ToolMethodBackend)` — Channel backend for Nextcloud Talk REST API.

**Constructor:** `NextcloudTalkChannelBackend() -> None`

- **connect** — Authenticate with Nextcloud Talk.<br/>`connect() -> bool`

- **join_channel** — Join a Nextcloud Talk room.<br/>`join_channel(channel_id: str) -> None`

- **poll_messages** — Poll a Nextcloud Talk room for new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Nextcloud Talk message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **is_from_bot** — Check if message is from the bot.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **list_rooms** — List Nextcloud Talk rooms.<br/>`list_rooms() -> str`

  - **Returns:** JSON string with room list (token, displayName, type).

- **get_room** — Get information about a Nextcloud Talk room.<br/>`get_room(token: str) -> str`

  - `token`: Room token.
  - **Returns:** JSON string with room details.

- **create_room** — Create a Nextcloud Talk room.<br/>`create_room(room_type: int = 3, invite: str = '', room_name: str = '') -> str`

  - `room_type`: 1=one-to-one, 2=group, 3=public. Default: 3.
  - `invite`: User ID, group ID, or circle ID to invite.
  - `room_name`: Room display name.
  - **Returns:** JSON string with room token.

- **list_participants** — List participants in a room.<br/>`list_participants(token: str) -> str`

  - `token`: Room token.
  - **Returns:** JSON string with participant list.

- **list_messages** — List messages in a Nextcloud Talk room.<br/>`list_messages(token: str, look_into_future: int = 0, limit: int = 100, last_known_message_id: int = 0) -> str`

  - `token`: Room token.
  - `look_into_future`: 0 for history, 1 for new messages. Default: 0.
  - `limit`: Maximum messages. Default: 100.
  - `last_known_message_id`: Last message ID seen (for pagination).
  - **Returns:** JSON string with message list.

- **post_message** — Post a message to a Nextcloud Talk room.<br/>`post_message(token: str, message: str, reply_to: int = 0) -> str`

  - `token`: Room token.
  - `message`: Message text.
  - `reply_to`: Message ID to reply to. Default: 0 (no reply).
  - **Returns:** JSON string with ok status and message id.

- **set_room_name** — Set the name of a Nextcloud Talk room.<br/>`set_room_name(token: str, name: str) -> str`

  - `token`: Room token.
  - `name`: New room name.
  - **Returns:** JSON string with ok status.

- **delete_message** — Delete a message from a room.<br/>`delete_message(token: str, message_id: int) -> str`

  - `token`: Room token.
  - `message_id`: Message ID to delete.
  - **Returns:** JSON string with ok status.

##### `class NextcloudTalkAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Nextcloud Talk API tools.

**Constructor:** `NextcloudTalkAgent() -> None`

______________________________________________________________________

#### `kiss.channels.nostr_agent` — *Nostr Agent — StatefulSorcarAgent extension with Nostr protocol tools.*

##### `class NostrChannelBackend(ToolMethodBackend)` — Channel backend for Nostr protocol via pynostr.

**Constructor:** `NostrChannelBackend() -> None`

- **connect** — Load Nostr keys from stored config.<br/>`connect() -> bool`

- **poll_messages** — Poll Nostr relays for new events (basic implementation).<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Publish a Nostr note.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Reply waiting is not currently supported for Nostr.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **is_from_bot** — Check if event is from this key.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **publish_note** — Publish a text note (kind 1) to Nostr.<br/>`publish_note(content: str) -> str`

  - `content`: Note content text.
  - **Returns:** JSON string with ok status and event id.

- **publish_reply** — Publish a reply to an existing Nostr event.<br/>`publish_reply(content: str, reply_to_event_id: str) -> str`

  - `content`: Reply content.
  - `reply_to_event_id`: Event ID to reply to.
  - **Returns:** JSON string with ok status and event id.

- **send_dm** — Send an encrypted direct message (NIP-04).<br/>`send_dm(recipient_pubkey: str, content: str) -> str`

  - `recipient_pubkey`: Recipient's public key (hex).
  - `content`: Message content (will be encrypted).
  - **Returns:** JSON string with ok status and event id.

- **get_profile** — Get the current user's Nostr profile.<br/>`get_profile() -> str`

  - **Returns:** JSON string with public key info.

- **set_profile** — Set the Nostr user profile (kind 0).<br/>`set_profile(name: str = '', about: str = '', picture: str = '', nip05: str = '') -> str`

  - `name`: Display name.
  - `about`: Bio/about text.
  - `picture`: Profile picture URL.
  - `nip05`: NIP-05 identifier (user@domain.com).
  - **Returns:** JSON string with ok status and event id.

- **list_relays** — List configured Nostr relays.<br/>`list_relays() -> str`

  - **Returns:** JSON string with relay list.

- **add_relay** — Add a Nostr relay to the configuration.<br/>`add_relay(relay_url: str) -> str`

  - `relay_url`: WebSocket URL of the relay (wss://...).
  - **Returns:** JSON string with ok status.

- **remove_relay** — Remove a Nostr relay from the configuration.<br/>`remove_relay(relay_url: str) -> str`

  - `relay_url`: Relay URL to remove.
  - **Returns:** JSON string with ok status.

##### `class NostrAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Nostr protocol tools.

**Constructor:** `NostrAgent() -> None`

______________________________________________________________________

#### `kiss.channels.phone_control_agent` — *Phone Control Agent — StatefulSorcarAgent extension with Android phone control tools.*

##### `class PhoneControlChannelBackend(ToolMethodBackend)` — Channel backend for Android phone control via REST API.

**Constructor:** `PhoneControlChannelBackend() -> None`

- **connect** — Connect to phone companion app.<br/>`connect() -> bool`

- **poll_messages** — Poll for new SMS messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send an SMS.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply SMS from a specific number.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **send_sms** — Send an SMS message.<br/>`send_sms(to: str, text: str) -> str`

  - `to`: Recipient phone number.
  - `text`: Message text.
  - **Returns:** JSON string with ok status.

- **make_call** — Make a phone call.<br/>`make_call(to: str) -> str`

  - `to`: Phone number to call.
  - **Returns:** JSON string with ok status.

- **end_call** — End the current active call.<br/>`end_call() -> str`

  - **Returns:** JSON string with ok status.

- **list_sms_conversations** — List recent SMS conversations.<br/>`list_sms_conversations(limit: int = 20) -> str`

  - `limit`: Maximum conversations to return. Default: 20.
  - **Returns:** JSON string with conversation list.

- **get_sms_messages** — Get messages in an SMS thread.<br/>`get_sms_messages(thread_id: str, limit: int = 50) -> str`

  - `thread_id`: Thread ID from list_sms_conversations.
  - `limit`: Maximum messages to return. Default: 50.
  - **Returns:** JSON string with message list.

- **get_call_log** — Get recent call log.<br/>`get_call_log(limit: int = 20) -> str`

  - `limit`: Maximum calls to return. Default: 20.
  - **Returns:** JSON string with call list.

- **get_device_info** — Get phone device information.<br/>`get_device_info() -> str`

  - **Returns:** JSON string with device info (model, battery, etc).

- **list_notifications** — List current phone notifications.<br/>`list_notifications() -> str`

  - **Returns:** JSON string with notification list.

- **dismiss_notification** — Dismiss a phone notification.<br/>`dismiss_notification(notification_id: str) -> str`

  - `notification_id`: Notification ID to dismiss.
  - **Returns:** JSON string with ok status.

- **send_notification_reply** — Reply to a phone notification (e.g. WhatsApp, Signal).<br/>`send_notification_reply(notification_id: str, text: str) -> str`

  - `notification_id`: Notification ID to reply to.
  - `text`: Reply text.
  - **Returns:** JSON string with ok status.

##### `class PhoneControlAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Android phone control tools.

**Constructor:** `PhoneControlAgent() -> None`

______________________________________________________________________

#### `kiss.channels.signal_agent` — *Signal Agent — StatefulSorcarAgent extension with Signal CLI tools.*

##### `class SignalChannelBackend(ToolMethodBackend)` — Channel backend for Signal via signal-cli.

**Constructor:** `SignalChannelBackend() -> None`

- **connect** — Load Signal config.<br/>`connect() -> bool`

- **poll_messages** — Receive pending Signal messages via signal-cli.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Signal message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **is_from_bot** — Check if a message is from the bot.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **send_signal_message** — Send a Signal text message.<br/>`send_signal_message(recipient: str, message: str) -> str`

  - `recipient`: Recipient phone number in E.164 format.
  - `message`: Message text to send.
  - **Returns:** JSON string with ok status.

- **receive_messages** — Receive pending Signal messages.<br/>`receive_messages(timeout: int = 5) -> str`

  - `timeout`: Seconds to wait for messages. Default: 5.
  - **Returns:** JSON string with list of received messages.

- **send_attachment** — Send a Signal message with an attachment.<br/>`send_attachment(recipient: str, message: str, file_path: str) -> str`

  - `recipient`: Recipient phone number.
  - `message`: Message text.
  - `file_path`: Local path to the file to attach.
  - **Returns:** JSON string with ok status.

- **list_contacts** — List Signal contacts.<br/>`list_contacts() -> str`

  - **Returns:** JSON string with contact list.

- **list_groups** — List Signal groups.<br/>`list_groups() -> str`

  - **Returns:** JSON string with group list.

##### `class SignalAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Signal CLI tools.

**Constructor:** `SignalAgent() -> None`

______________________________________________________________________

#### `kiss.channels.slack_agent` — *Slack Agent — StatefulSorcarAgent extension with Slack API tools.*

##### `class SlackChannelBackend(ToolMethodBackend)` — Slack channel backend.

**Constructor:** `SlackChannelBackend(workspace: str = 'default') -> None`

- **connect** — Authenticate with Slack using the stored bot token. Uses the workspace set at construction time to load the appropriate token.<br/>`connect() -> bool`

  - **Returns:** True on success, False on failure.

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

- **poll_thread_messages** — Poll a Slack thread for new replies since *oldest*. Used by the poller to detect user replies within active threads. The parent message itself is excluded from the results. Retries up to 3 times on transient network errors with exponential backoff (same strategy as `poll_messages`).<br/>`poll_thread_messages(channel_id: str, thread_ts: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

  - `channel_id`: Channel ID containing the thread.
  - `thread_ts`: Timestamp of the parent message (thread root).
  - `oldest`: Only return messages newer than this timestamp.
  - `limit`: Maximum number of messages to return.
  - **Returns:** Tuple of (reply messages sorted oldest-first, updated oldest timestamp).

- **send_message** — Send a message to a Slack channel, optionally in a thread.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

  - `channel_id`: Channel ID to post to.
  - `text`: Message text (supports Slack mrkdwn formatting).
  - `thread_ts`: If non-empty, reply in this thread.

- **wait_for_reply** — Poll a Slack thread for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

  - `channel_id`: Channel ID containing the thread.
  - `thread_ts`: Timestamp of the parent message (thread root).
  - `user_id`: User ID to wait for a reply from.
  - **Returns:** The text of the user's reply message, or `None` on timeout.

- **disconnect** — Release Slack backend state before stop or reconnect.<br/>`disconnect() -> None`

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

##### `class SlackAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Slack workspace tools.

**Constructor:** `SlackAgent(workspace: str = 'default') -> None`

- **run** — Run with Slack-specific system prompt encouraging browser-based auth.<br/>`run(**kwargs: Any) -> str`

______________________________________________________________________

#### `kiss.channels.sms_agent` — *SMS Agent — StatefulSorcarAgent extension with Twilio SMS tools.*

##### `class SMSChannelBackend(ToolMethodBackend)` — Channel backend for Twilio SMS.

**Constructor:** `SMSChannelBackend() -> None`

- **connect** — Authenticate with Twilio using stored config.<br/>`connect() -> bool`

- **poll_messages** — Poll Twilio for recent inbound messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send an SMS.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific number.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **is_from_bot** — Check if message is from the bot's number.<br/>`is_from_bot(msg: dict[str, Any]) -> bool`

- **send_sms** — Send an SMS message via Twilio.<br/>`send_sms(to: str, body: str) -> str`

  - `to`: Recipient phone number in E.164 format.
  - `body`: Message text (up to 1600 characters).
  - **Returns:** JSON string with ok status and message SID.

- **send_mms** — Send an MMS message with media via Twilio.<br/>`send_mms(to: str, body: str, media_url: str) -> str`

  - `to`: Recipient phone number in E.164 format.
  - `body`: Message text.
  - `media_url`: Publicly accessible URL of the media file.
  - **Returns:** JSON string with ok status and message SID.

- **list_messages** — List Twilio messages.<br/>`list_messages(to: str = '', from_: str = '', limit: int = 20, page_token: str = '') -> str`

  - `to`: Filter by recipient phone number. Optional.
  - `from_`: Filter by sender phone number. Optional.
  - `limit`: Maximum messages to return. Default: 20.
  - `page_token`: Pagination token. Optional.
  - **Returns:** JSON string with message list.

- **get_message** — Get details about a specific Twilio message.<br/>`get_message(message_sid: str) -> str`

  - `message_sid`: Message SID (e.g. "SM...").
  - **Returns:** JSON string with message details.

- **list_phone_numbers** — List Twilio phone numbers on the account.<br/>`list_phone_numbers(limit: int = 20) -> str`

  - `limit`: Maximum numbers to return. Default: 20.
  - **Returns:** JSON string with phone number list.

- **get_account_info** — Get Twilio account information.<br/>`get_account_info() -> str`

  - **Returns:** JSON string with account details.

- **send_whatsapp_message** — Send a WhatsApp message via Twilio.<br/>`send_whatsapp_message(to: str, body: str) -> str`

  - `to`: Recipient WhatsApp number in format "whatsapp:+14155238886".
  - `body`: Message text.
  - **Returns:** JSON string with ok status and message SID.

- **create_call** — Create a Twilio voice call.<br/>`create_call(to: str, url: str, method: str = 'GET') -> str`

  - `to`: Phone number to call.
  - `url`: TwiML URL for the call instructions.
  - `method`: HTTP method for the URL. Default: "GET".
  - **Returns:** JSON string with ok status and call SID.

- **list_calls** — List recent Twilio calls.<br/>`list_calls(to: str = '', from_: str = '', limit: int = 20) -> str`

  - `to`: Filter by recipient phone number. Optional.
  - `from_`: Filter by caller phone number. Optional.
  - `limit`: Maximum calls to return. Default: 20.
  - **Returns:** JSON string with call list.

- **get_call** — Get details about a specific Twilio call.<br/>`get_call(call_sid: str) -> str`

  - `call_sid`: Call SID (e.g. "CA...").
  - **Returns:** JSON string with call details.

- **cancel_message** — Cancel a queued or scheduled Twilio message.<br/>`cancel_message(message_sid: str) -> str`

  - `message_sid`: Message SID to cancel.
  - **Returns:** JSON string with ok status.

##### `class SMSAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Twilio SMS tools.

**Constructor:** `SMSAgent() -> None`

______________________________________________________________________

#### `kiss.channels.synology_chat_agent` — *Synology Chat Agent — StatefulSorcarAgent extension with Synology Chat webhook API.*

##### `class SynologyChatChannelBackend(ToolMethodBackend)` — Channel backend for Synology Chat webhooks.

**Constructor:** `SynologyChatChannelBackend() -> None`

- **connect** — Load Synology config and start webhook server.<br/>`connect() -> bool`

- **poll_messages** — Drain the webhook message queue.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Synology Chat message via incoming webhook.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **disconnect** — Stop the embedded webhook server and release backend resources.<br/>`disconnect() -> None`

- **post_message** — Send a message to Synology Chat via incoming webhook.<br/>`post_message(text: str, user_ids: str = '') -> str`

  - `text`: Message text.
  - `user_ids`: Comma-separated user IDs to send to (optional). If empty, sends to the default channel.
  - **Returns:** JSON string with ok status.

- **send_file_message** — Send a message with a file attachment.<br/>`send_file_message(text: str, file_url: str) -> str`

  - `text`: Message text.
  - `file_url`: URL of the file to attach.
  - **Returns:** JSON string with ok status.

##### `class SynologyChatAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Synology Chat webhook tools.

**Constructor:** `SynologyChatAgent() -> None`

______________________________________________________________________

#### `kiss.channels.telegram_agent` — *Telegram Agent — StatefulSorcarAgent extension with Telegram Bot API tools.*

##### `class TelegramChannelBackend(ToolMethodBackend)` — Channel backend for Telegram Bot API.

**Constructor:** `TelegramChannelBackend() -> None`

- **connect** — Authenticate with Telegram using the stored bot token.<br/>`connect() -> bool`

- **poll_messages** — Poll for new Telegram updates via getUpdates.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Telegram message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **send_text** — Send a text message to a Telegram chat.<br/>`send_text(chat_id: str, text: str, reply_to_message_id: str = '') -> str`

  - `chat_id`: Chat ID (integer as string) or @username.
  - `text`: Message text (supports Markdown).
  - `reply_to_message_id`: Optional message ID to reply to.
  - **Returns:** JSON string with ok status and message_id.

- **send_photo** — Send a photo to a Telegram chat.<br/>`send_photo(chat_id: str, photo_url_or_path: str, caption: str = '') -> str`

  - `chat_id`: Chat ID or @username.
  - `photo_url_or_path`: URL or local file path of the photo.
  - `caption`: Optional caption text.
  - **Returns:** JSON string with ok status and message_id.

- **send_document** — Send a document/file to a Telegram chat.<br/>`send_document(chat_id: str, document_path: str, caption: str = '') -> str`

  - `chat_id`: Chat ID or @username.
  - `document_path`: Local file path to send.
  - `caption`: Optional caption text.
  - **Returns:** JSON string with ok status and message_id.

- **edit_message_text** — Edit an existing message text.<br/>`edit_message_text(chat_id: str, message_id: str, text: str) -> str`

  - `chat_id`: Chat ID where the message is.
  - `message_id`: ID of the message to edit.
  - `text`: New message text.
  - **Returns:** JSON string with ok status.

- **delete_message** — Delete a message.<br/>`delete_message(chat_id: str, message_id: str) -> str`

  - `chat_id`: Chat ID where the message is.
  - `message_id`: ID of the message to delete.
  - **Returns:** JSON string with ok status.

- **pin_message** — Pin a message in a chat.<br/>`pin_message(chat_id: str, message_id: str) -> str`

  - `chat_id`: Chat ID.
  - `message_id`: ID of the message to pin.
  - **Returns:** JSON string with ok status.

- **unpin_message** — Unpin a message (or all messages) in a chat.<br/>`unpin_message(chat_id: str, message_id: str = '') -> str`

  - `chat_id`: Chat ID.
  - `message_id`: ID of specific message to unpin. If empty, unpins all.
  - **Returns:** JSON string with ok status.

- **get_chat** — Get information about a chat.<br/>`get_chat(chat_id: str) -> str`

  - `chat_id`: Chat ID or @username.
  - **Returns:** JSON string with chat info (id, title, type, members_count).

- **get_chat_members_count** — Get the number of members in a chat.<br/>`get_chat_members_count(chat_id: str) -> str`

  - `chat_id`: Chat ID or @username.
  - **Returns:** JSON string with member count.

- **get_chat_member** — Get information about a chat member.<br/>`get_chat_member(chat_id: str, user_id: str) -> str`

  - `chat_id`: Chat ID.
  - `user_id`: User ID.
  - **Returns:** JSON string with member info (user, status).

- **ban_chat_member** — Ban a user from a chat.<br/>`ban_chat_member(chat_id: str, user_id: str) -> str`

  - `chat_id`: Chat ID.
  - `user_id`: User ID to ban.
  - **Returns:** JSON string with ok status.

- **unban_chat_member** — Unban a user from a chat.<br/>`unban_chat_member(chat_id: str, user_id: str) -> str`

  - `chat_id`: Chat ID.
  - `user_id`: User ID to unban.
  - **Returns:** JSON string with ok status.

- **get_updates** — Get recent updates (messages) from the bot.<br/>`get_updates(offset: str = '', limit: int = 10) -> str`

  - `offset`: Update ID offset for pagination.
  - `limit`: Maximum number of updates to return (1-100).
  - **Returns:** JSON string with list of update objects.

- **send_poll** — Send a poll to a chat.<br/>`send_poll(chat_id: str, question: str, options_json: str, is_anonymous: bool = True) -> str`

  - `chat_id`: Chat ID.
  - `question`: Poll question.
  - `options_json`: JSON array of option strings (2-10 options).
  - `is_anonymous`: Whether the poll is anonymous. Default: True.
  - **Returns:** JSON string with ok status and message_id.

- **forward_message** — Forward a message to another chat.<br/>`forward_message(chat_id: str, from_chat_id: str, message_id: str) -> str`

  - `chat_id`: Target chat ID.
  - `from_chat_id`: Source chat ID.
  - `message_id`: ID of the message to forward.
  - **Returns:** JSON string with ok status and message_id.

##### `class TelegramAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Telegram Bot API tools.

**Constructor:** `TelegramAgent() -> None`

______________________________________________________________________

#### `kiss.channels.tlon_agent` — *Tlon/Urbit Agent — StatefulSorcarAgent extension with Tlon/Urbit Eyre HTTP tools.*

##### `class TlonChannelBackend(ToolMethodBackend)` — Channel backend for Tlon/Urbit Eyre HTTP.

**Constructor:** `TlonChannelBackend() -> None`

- **connect** — Authenticate with Urbit ship.<br/>`connect() -> bool`

- **poll_messages** — Poll event queue for messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Tlon/Urbit poke.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **list_groups** — List Urbit groups.<br/>`list_groups() -> str`

  - **Returns:** JSON string with group list.

- **list_channels** — List channels in an Urbit group.<br/>`list_channels(group_path: str) -> str`

  - `group_path`: Group path (e.g. "~sampel/my-group").
  - **Returns:** JSON string with channel list.

- **get_messages** — Get recent messages from a Tlon channel.<br/>`get_messages(group_path: str, channel_name: str, count: int = 20) -> str`

  - `group_path`: Group path.
  - `channel_name`: Channel name within the group.
  - `count`: Number of messages to retrieve. Default: 20.
  - **Returns:** JSON string with messages.

- **post_message** — Post a message to a Tlon channel.<br/>`post_message(group_path: str, channel_name: str, content: str) -> str`

  - `group_path`: Group path (e.g. "~sampel/my-group").
  - `channel_name`: Channel name within the group.
  - `content`: Message content text.
  - **Returns:** JSON string with ok status.

- **get_profile** — Get the current ship's profile.<br/>`get_profile() -> str`

  - **Returns:** JSON string with profile info.

- **poke** — Send a poke to an Urbit app.<br/>`poke(app: str, mark: str, json_body: str) -> str`

  - `app`: Gall agent name (e.g. "groups").
  - `mark`: Mark name (e.g. "groups-action").
  - `json_body`: JSON string of the poke body.
  - **Returns:** JSON string with ok status.

- **scry** — Perform a scry request on an Urbit app.<br/>`scry(app: str, path: str) -> str`

  - `app`: Gall agent name.
  - `path`: Scry path (starting with /).
  - **Returns:** JSON string with scry result.

##### `class TlonAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Tlon/Urbit Eyre HTTP tools.

**Constructor:** `TlonAgent() -> None`

______________________________________________________________________

#### `kiss.channels.twitch_agent` — *Twitch Agent — StatefulSorcarAgent extension with Twitch Helix API + Chat tools.*

##### `class TwitchChannelBackend(ToolMethodBackend)` — Channel backend for Twitch Helix API.

**Constructor:** `TwitchChannelBackend() -> None`

- **connect** — Authenticate with Twitch using stored config.<br/>`connect() -> bool`

- **poll_messages** — Poll for Twitch events (basic REST polling).<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Twitch chat message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Reply waiting is not currently supported for Twitch.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **get_stream_info** — Get live stream information for a Twitch channel.<br/>`get_stream_info(broadcaster_login: str) -> str`

  - `broadcaster_login`: Twitch channel username.
  - **Returns:** JSON string with stream info (game, title, viewer count, etc).

- **get_channel_info** — Get channel information for a Twitch broadcaster.<br/>`get_channel_info(broadcaster_id: str) -> str`

  - `broadcaster_id`: Twitch broadcaster ID.
  - **Returns:** JSON string with channel info.

- **get_user_info** — Get Twitch user information.<br/>`get_user_info(login_or_id: str) -> str`

  - `login_or_id`: Twitch username (login) or user ID.
  - **Returns:** JSON string with user info.

- **get_chatters** — Get current chatters in a Twitch channel.<br/>`get_chatters(broadcaster_id: str, moderator_id: str = '') -> str`

  - `broadcaster_id`: Broadcaster user ID.
  - `moderator_id`: Moderator user ID (optional, defaults to broadcaster).
  - **Returns:** JSON string with chatters list.

- **send_chat_message** — Send a message to a Twitch chat.<br/>`send_chat_message(broadcaster_id: str, sender_id: str, message: str) -> str`

  - `broadcaster_id`: Broadcaster channel ID.
  - `sender_id`: Sender user ID.
  - `message`: Message text.
  - **Returns:** JSON string with ok status.

- **ban_user** — Ban or timeout a Twitch user.<br/>`ban_user(broadcaster_id: str, moderator_id: str, user_id: str, duration: int = 0, reason: str = '') -> str`

  - `broadcaster_id`: Broadcaster channel ID.
  - `moderator_id`: Moderator user ID.
  - `user_id`: User ID to ban.
  - `duration`: Timeout duration in seconds (0 = permanent ban).
  - `reason`: Optional ban reason.
  - **Returns:** JSON string with ok status.

- **search_channels** — Search for Twitch channels by name.<br/>`search_channels(query: str, limit: int = 10) -> str`

  - `query`: Search query.
  - `limit`: Maximum channels to return. Default: 10.
  - **Returns:** JSON string with matching channels.

- **get_clips** — Get clips from a Twitch channel.<br/>`get_clips(broadcaster_id: str, limit: int = 20) -> str`

  - `broadcaster_id`: Broadcaster ID.
  - `limit`: Maximum clips to return. Default: 20.
  - **Returns:** JSON string with clip list.

- **create_clip** — Create a clip from a live stream.<br/>`create_clip(broadcaster_id: str, has_delay: bool = False) -> str`

  - `broadcaster_id`: Broadcaster ID.
  - `has_delay`: Whether to add a 5-second delay. Default: False.
  - **Returns:** JSON string with clip edit URL.

##### `class TwitchAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Twitch Helix API tools.

**Constructor:** `TwitchAgent() -> None`

______________________________________________________________________

#### `kiss.channels.whatsapp_agent` — *WhatsApp Agent — StatefulSorcarAgent extension with WhatsApp Business Cloud API tools.*

##### `class WhatsAppChannelBackend(ToolMethodBackend)` — Channel backend for WhatsApp Business Cloud API.

**Constructor:** `WhatsAppChannelBackend() -> None`

- **connect** — Authenticate with WhatsApp using stored config and start webhook server.<br/>`connect() -> bool`

  - **Returns:** True on success, False on failure.

- **poll_messages** — Drain the webhook message queue and return new messages.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

  - `channel_id`: Recipient phone number (unused — all messages returned).
  - `oldest`: Unused for push-mode channels.
  - `limit`: Maximum messages to return.
  - **Returns:** Tuple of (messages, oldest). Each message dict has at minimum: ts, user (from), text.

- **send_message** — Send a text message to a WhatsApp number.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

  - `channel_id`: Recipient phone number in E.164 format.
  - `text`: Message text.
  - `thread_ts`: Unused for WhatsApp.

- **wait_for_reply** — Block until a message from a specific user is received.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

  - `channel_id`: Unused for WhatsApp.
  - `thread_ts`: Unused for WhatsApp.
  - `user_id`: Phone number to wait for.
  - **Returns:** The text of the user's reply.

- **disconnect** — Stop the embedded webhook server and release backend resources.<br/>`disconnect() -> None`

- **send_text_message** — Send a text message to a WhatsApp number.<br/>`send_text_message(to: str, body: str, preview_url: bool = False) -> str`

  - `to`: Recipient phone number in E.164 format (e.g. "+14155238886"). Include country code, no spaces or dashes.
  - `body`: Message text (up to 4096 characters).
  - `preview_url`: If True, URLs in the body will show a preview. Default: False.
  - **Returns:** JSON string with ok status and message_id.

- **send_template_message** — Send a pre-approved template message. Template messages are required to initiate conversations outside the 24-hour customer service window.<br/>`send_template_message(to: str, template_name: str, language_code: str = 'en_US', components: str = '') -> str`

  - `to`: Recipient phone number in E.164 format.
  - `template_name`: Name of the approved message template.
  - `language_code`: Template language code (e.g. "en_US"). Default: "en_US".
  - `components`: Optional JSON string of template components (header, body, button parameters).
  - **Returns:** JSON string with ok status and message_id.

- **send_media_message** — Send a media message (image, document, audio, video, sticker). Provide either media_id (from upload_media) or link (public URL).<br/>`send_media_message(to: str, media_type: str, media_id: str = '', link: str = '', caption: str = '', filename: str = '') -> str`

  - `to`: Recipient phone number in E.164 format.
  - `media_type`: Type of media. Options: "image", "document", "audio", "video", "sticker".
  - `media_id`: Media ID from a previous upload_media call.
  - `link`: Public URL of the media file. Used if media_id is empty.
  - `caption`: Optional caption (supported for image, video, document).
  - `filename`: Optional filename (for document type).
  - **Returns:** JSON string with ok status and message_id.

- **send_reaction** — React to a message with an emoji.<br/>`send_reaction(to: str, message_id: str, emoji: str) -> str`

  - `to`: Phone number of the message recipient.
  - `message_id`: ID of the message to react to.
  - `emoji`: Emoji character (e.g. "👍", "❤️", "😂").
  - **Returns:** JSON string with ok status and message_id.

- **send_location_message** — Send a location message.<br/>`send_location_message(to: str, latitude: str, longitude: str, name: str = '', address: str = '') -> str`

  - `to`: Recipient phone number in E.164 format.
  - `latitude`: Latitude of the location (e.g. "37.7749").
  - `longitude`: Longitude of the location (e.g. "-122.4194").
  - `name`: Optional name of the location.
  - `address`: Optional address of the location.
  - **Returns:** JSON string with ok status and message_id.

- **send_interactive_message** — Send an interactive message (buttons, lists, or product messages).<br/>`send_interactive_message(to: str, interactive_json: str) -> str`

  - `to`: Recipient phone number in E.164 format.
  - `interactive_json`: JSON string of the interactive object.
  - **Returns:** JSON string with ok status and message_id.

- **send_contact_message** — Send a contact card message.<br/>`send_contact_message(to: str, contacts_json: str) -> str`

  - `to`: Recipient phone number in E.164 format.
  - `contacts_json`: JSON string of contacts array.
  - **Returns:** JSON string with ok status and message_id.

- **mark_as_read** — Mark a received message as read.<br/>`mark_as_read(message_id: str) -> str`

  - `message_id`: ID of the message to mark as read.
  - **Returns:** JSON string with ok status.

- **get_business_profile** — Get the WhatsApp Business profile information.<br/>`get_business_profile() -> str`

  - **Returns:** JSON string with business profile data (about, address, description, email, websites, profile_picture_url).

- **update_business_profile** — Update the WhatsApp Business profile.<br/>`update_business_profile(about: str = '', address: str = '', description: str = '', email: str = '', websites: str = '', vertical: str = '') -> str`

  - `about`: Short description (max 139 characters).
  - `address`: Business address.
  - `description`: Full business description (max 512 characters).
  - `email`: Business email address.
  - `websites`: Comma-separated list of website URLs (max 2).
  - `vertical`: Business category (e.g. "RETAIL", "FOOD", "HEALTH").
  - **Returns:** JSON string with ok status.

- **upload_media** — Upload a media file for later sending.<br/>`upload_media(file_path: str, mime_type: str) -> str`

  - `file_path`: Local path to the file to upload.
  - `mime_type`: MIME type of the file (e.g. "image/jpeg", "application/pdf", "video/mp4", "audio/ogg").
  - **Returns:** JSON string with ok status and media_id (use in send_media_message).

- **get_media_url** — Get the download URL for an uploaded media file.<br/>`get_media_url(media_id: str) -> str`

  - `media_id`: Media ID from upload_media or a received message.
  - **Returns:** JSON string with ok status, url, mime_type, and file_size.

- **delete_media** — Delete an uploaded media file.<br/>`delete_media(media_id: str) -> str`

  - `media_id`: Media ID to delete.
  - **Returns:** JSON string with ok status.

- **list_message_templates** — List available message templates for the WhatsApp Business Account. Requires waba_id to be configured.<br/>`list_message_templates(limit: int = 20, status: str = '') -> str`

  - `limit`: Maximum number of templates to return. Default: 20.
  - `status`: Filter by status ("APPROVED", "PENDING", "REJECTED"). If empty, returns all statuses.
  - **Returns:** JSON string with template list (name, status, category, language).

##### `class WhatsAppAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with WhatsApp Business Cloud API tools.

**Constructor:** `WhatsAppAgent() -> None`

______________________________________________________________________

#### `kiss.channels.zalo_agent` — *Zalo Agent — StatefulSorcarAgent extension with Zalo Official Account API tools.*

##### `class ZaloChannelBackend(ToolMethodBackend)` — Channel backend for Zalo OA API.

**Constructor:** `ZaloChannelBackend() -> None`

- **connect** — Load Zalo config and start webhook server.<br/>`connect() -> bool`

- **poll_messages** — Drain the webhook message queue.<br/>`poll_messages(channel_id: str, oldest: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]`

- **send_message** — Send a Zalo text message.<br/>`send_message(channel_id: str, text: str, thread_ts: str = '') -> None`

- **wait_for_reply** — Poll for a reply from a specific user.<br/>`wait_for_reply(channel_id: str, thread_ts: str, user_id: str, timeout_seconds: float = 300.0) -> str | None`

- **disconnect** — Stop the embedded webhook server and release backend resources.<br/>`disconnect() -> None`

- **send_text_message** — Send a text message to a Zalo user.<br/>`send_text_message(to_user_id: str, text: str) -> str`

  - `to_user_id`: Zalo user ID.
  - `text`: Message text.
  - **Returns:** JSON string with ok status.

- **send_image_message** — Send an image message to a Zalo user.<br/>`send_image_message(to_user_id: str, image_url: str, caption: str = '') -> str`

  - `to_user_id`: Zalo user ID.
  - `image_url`: URL of the image to send.
  - `caption`: Optional image caption.
  - **Returns:** JSON string with ok status.

- **get_follower_profile** — Get a Zalo follower's profile.<br/>`get_follower_profile(user_id: str) -> str`

  - `user_id`: Zalo user ID.
  - **Returns:** JSON string with user profile.

- **get_followers** — Get followers of the Zalo OA.<br/>`get_followers(offset: int = 0, count: int = 50) -> str`

  - `offset`: Pagination offset. Default: 0.
  - `count`: Number of followers to return (max 50). Default: 50.
  - **Returns:** JSON string with follower list.

- **get_oa_info** — Get Zalo Official Account information.<br/>`get_oa_info() -> str`

  - **Returns:** JSON string with OA info (name, id, description, etc).

- **get_recent_messages** — Get recent messages from the OA.<br/>`get_recent_messages(offset: int = 0, count: int = 10) -> str`

  - `offset`: Pagination offset. Default: 0.
  - `count`: Number of messages. Default: 10.
  - **Returns:** JSON string with message list.

- **get_conversation** — Get conversation history with a specific user.<br/>`get_conversation(user_id: str, offset: int = 0, count: int = 20) -> str`

  - `user_id`: Zalo user ID.
  - `offset`: Pagination offset. Default: 0.
  - `count`: Number of messages. Default: 20.
  - **Returns:** JSON string with conversation messages.

- **upload_image** — Upload an image file to Zalo.<br/>`upload_image(file_path: str) -> str`

  - `file_path`: Local path to the image file.
  - **Returns:** JSON string with ok status and attachment_id.

##### `class ZaloAgent(BaseChannelAgent, StatefulSorcarAgent)` — StatefulSorcarAgent extended with Zalo OA API tools.

**Constructor:** `ZaloAgent() -> None`

______________________________________________________________________

#### `kiss.core.models.claude_code_model` — *Claude Code model implementation — uses the `claude` CLI as an LLM backend.*

##### `class ClaudeCodeModel(Model)` — A model that delegates to the Claude Code CLI for LLM completions.

**Constructor:** `ClaudeCodeModel(model_name: str, model_config: dict[str, Any] | None = None, token_callback: TokenCallback | None = None)`

- `model_name`: Full model name including `cc/` prefix (e.g. `cc/opus`).

- `model_config`: Optional configuration. Recognised keys: - `system_instruction` (str): System prompt for the session. - `timeout` (int): Subprocess timeout in seconds (default 300).

- `token_callback`: Optional callback invoked with each streamed text token.

- **initialize** — Initialize the conversation with an initial user prompt.<br/>`initialize(prompt: str, attachments: list[Attachment] | None = None) -> None`

  - `prompt`: The initial user prompt.
  - `attachments`: Not supported — ignored with a warning if provided.

- **generate** — Generate a response using the Claude Code CLI.<br/>`generate() -> tuple[str, Any]`

  - **Returns:** tuple\[str, Any\]: (generated_text, parsed_json_response).

- **generate_and_process_with_tools** — Generate with text-based tool calling via the Claude Code CLI. Tool descriptions are injected into the system prompt. The model's text output is parsed for JSON `tool_calls` blocks, which are returned to the framework for execution — the CLI itself runs in pure LLM mode (`--tools ""`), **not** as an agent.<br/>`generate_and_process_with_tools(function_map: dict[str, Callable[..., Any]], tools_schema: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str, Any]`

  - `function_map`: Dictionary mapping function names to callable functions.
  - `tools_schema`: Ignored (text-based tool calling builds its own prompt).
  - **Returns:** Tuple of `(function_calls, content, response)`.

- **extract_input_output_token_counts_from_response** — Extract token counts from the Claude Code CLI JSON response.<br/>`extract_input_output_token_counts_from_response(response: Any) -> tuple[int, int, int, int]`

  - `response`: The parsed JSON response from the CLI.
  - **Returns:** (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).

- **get_embedding** — Not supported — Claude Code CLI does not provide embeddings.<br/>`get_embedding(text: str, embedding_model: str | None = None) -> list[float]`

______________________________________________________________________

#### `kiss.docker.docker_tools` — *File tools (Read, Write, Edit) that execute inside a Docker container via bash.*

##### `class DockerTools` — File tools that execute inside a Docker container via bash.

**Constructor:** `DockerTools(bash_fn: Callable[[str, str], str]) -> None`

- `bash_fn`: Callable(command, description) -> output string. Executes a bash command inside the Docker container.

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

______________________________________________________________________
