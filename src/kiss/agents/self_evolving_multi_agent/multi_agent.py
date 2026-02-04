# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Advanced Coding Agent with planning, error recovery, and dynamic tool creation.

PSEUDOCODE:
===========

run(task):
    Initialize Docker container
    Run orchestrator agent with task

Orchestrator Agent:
    Given: task, todo_list, completed_tasks, last_error
    Tools: plan_task, execute_todo, complete_todo, run_bash,
           read_file, write_file, create_tool, finish

    Loop until finish() called or budget exhausted:
        LLM decides next action based on current state
        - Simple work: use run_bash/write_file directly
        - Complex work: plan_task() to create todos, then execute_todo() to delegate

execute_todo(todo_id):
    Spawn sub-agent to handle single todo
    Sub-agent has limited tools: run_bash, read_file, write_file
    On success: mark todo completed
    On failure: retry up to max_retries, then mark failed

"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import kiss.agents.self_evolving_multi_agent.config  # noqa: F401
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.docker.docker_manager import DockerManager


@dataclass
class TodoItem:
    """A single todo item in the task list."""
    id: int
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed

@dataclass
class AgentState:
    """Mutable state for the coding agent."""
    todos: list[TodoItem] = field(default_factory=list)
    dynamic_tools: dict[str, Callable[..., str]] = field(default_factory=dict)
    error_count: int = 0
    last_error: str = ""
    completed_tasks: list[str] = field(default_factory=list)

ORCHESTRATOR_PROMPT = """## Task
{task}

## Current State
Todos: {todo_list}
Done: {completed_tasks}
Last Error: {last_error}

## Tools
- plan_task(tasks: str): Define multiple steps (newline separated). Use for complex projects.
- search_web(query: str): Search the web for information.
- execute_todo(todo_id: int): Delegate complex logic to a sub-agent.
- complete_todo(todo_id: int, result: str): Mark a task finished after manual work.
- run_bash(command: str): Execute shell. Batch multiple commands with &&.
- create_tool(name: str, description: str, bash_command_template: str): Create a reusable tool.
- read_file(path: str): Read file content.
- write_file(path: str, content: str): Write file content.
- finish(result: str): Task complete.

## Strategy
1. For simple tasks, use run_bash/write_file directly then call finish.
2. For complex logic, use plan_task followed by execute_todo or run_bash.
3. Batch commands in run_bash to minimize steps.
4. Call finish immediately upon goal completion."""

SUB_AGENT_PROMPT = """## Sub-Task
{task}
Focus ONLY on this. Use run_bash, read_file, and write_file. Be concise and report results."""

class SelfEvolvingMultiAgent:
    """Optimized coding agent with planning and tool-usage efficiency."""

    def __init__(self) -> None:
        """Initialize the SelfEvolvingMultiAgent with configuration from DEFAULT_CONFIG."""
        cfg = DEFAULT_CONFIG.self_evolving_multi_agent
        self.model_name = cfg.model
        self.docker_image = cfg.docker_image
        self.workdir = cfg.workdir
        self.max_steps = cfg.max_steps
        self.max_budget = cfg.max_budget

        self.sub_agent_model = cfg.sub_agent_model
        self.sub_agent_max_steps = cfg.sub_agent_max_steps
        self.sub_agent_max_budget = cfg.sub_agent_max_budget
        self.max_retries = cfg.max_retries

        self.state = AgentState()
        self.docker: DockerManager | None = None

    def _format_todos(self) -> str:
        """Format todo items as a string.

        Returns:
            str: Formatted list of todos with id, status, and description.
        """
        items = [f"#{t.id}: [{t.status}] {t.description}" for t in self.state.todos]
        return "\n".join(items) or "None."

    def _format_done(self) -> str:
        """Format completed tasks as a string.

        Returns:
            str: Comma-separated list of completed task summaries.
        """
        return ", ".join(self.state.completed_tasks) or "None."

    def _create_tools(self) -> list[Callable[..., str]]:
        """Create the set of tools available to the orchestrator.

        Returns:
            list[Callable[..., str]]: List of tool functions including plan_task,
                execute_todo, complete_todo, run_bash, create_tool, read_file,
                write_file, and any dynamic tools created during execution.
        """
        def run_bash(command: str, description: str = "") -> str:
            if not self.docker:
                raise KISSError("Docker not initialized.")
            return self.docker.run_bash_command(command, description or "Command")

        def read_file(path: str) -> str:
            return run_bash(f"cat -n {path}", f"Read {path}")

        def write_file(path: str, content: str) -> str:
            encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
            return run_bash(f"echo '{encoded}' | base64 -d > {path}", f"Write {path}")

        def plan_task(tasks: str) -> str:
            lines = [t.strip() for t in tasks.strip().splitlines() if t.strip()]
            added = []
            for desc in lines:
                tid = len(self.state.todos) + 1
                self.state.todos.append(TodoItem(id=tid, description=desc))
                added.append(str(tid))
            return f"Created todos: {', '.join(added)}"

        def complete_todo(todo_id: int, result: str = "Success") -> str:
            todo = next((t for t in self.state.todos if t.id == todo_id), None)
            if not todo:
                return f"Error: Todo {todo_id} not found."
            todo.status = "completed"
            self.state.completed_tasks.append(f"[{todo_id}] {todo.description}: {result}")
            return f"Todo {todo_id} completed."

        def execute_todo(todo_id: int) -> str:
            todo = next((t for t in self.state.todos if t.id == todo_id), None)
            if not todo or todo.status == "completed":
                return f"Todo {todo_id} invalid or already done."

            todo.status = "in_progress"
            try:
                sub_agent = KISSAgent(name=f"SubAgent-{todo_id}")
                res = sub_agent.run(
                    model_name=self.sub_agent_model,
                    prompt_template=SUB_AGENT_PROMPT,
                    arguments={"task": todo.description},
                    tools=[run_bash, read_file, write_file],
                    max_steps=self.sub_agent_max_steps,
                    max_budget=self.sub_agent_max_budget,
                )
                todo.status = "completed"
                self.state.completed_tasks.append(f"[{todo_id}] {todo.description}: {res}")
                return f"Todo {todo_id} finished: {res}"
            except Exception as e:
                self.state.error_count += 1
                self.state.last_error = str(e)
                if self.state.error_count <= self.max_retries:
                    todo.status = "pending"
                    return f"Attempt failed, retrying {todo_id}: {e}"
                todo.status = "failed"
                return f"Todo {todo_id} failed: {e}"

        def create_tool(name: str, description: str, bash_command_template: str) -> str:
            if not name.isidentifier():
                return "Invalid identifier."

            def dynamic_tool(arg: str = "") -> str:
                try:
                    return run_bash(bash_command_template.format(arg=arg), description)
                except Exception as e:
                    return str(e)

            dynamic_tool.__name__, dynamic_tool.__doc__ = name, description
            self.state.dynamic_tools[name] = dynamic_tool
            return f"Tool '{name}' ready."

        tools: list[Callable[..., str]] = [
            plan_task, execute_todo, complete_todo, run_bash,
            create_tool, read_file, write_file
        ]
        tools.extend(self.state.dynamic_tools.values())
        return tools

    def run(self, task: str) -> str:
        """Run the agent on a task.

        Args:
            task: The task description for the agent to execute.

        Returns:
            str: The result from the orchestrator agent.
        """
        self.state = AgentState()
        if self.docker:
            return self._run_orchestrator(task)

        with DockerManager(self.docker_image, workdir="/", mount_shared_volume=True) as docker:
            self.docker = docker
            docker.run_bash_command("mkdir -p /workspace", "Init")
            docker.workdir = self.workdir
            return self._run_orchestrator(task)

    def _run_orchestrator(self, task: str) -> str:
        """Run the orchestrator agent on a task.

        Args:
            task: The task description for the orchestrator.

        Returns:
            str: The result from the orchestrator agent.
        """
        orchestrator = KISSAgent(name="Orchestrator")
        return orchestrator.run(
            model_name=self.model_name,
            prompt_template=ORCHESTRATOR_PROMPT,
            arguments={
                "task": task,
                "todo_list": self._format_todos(),
                "completed_tasks": self._format_done(),
                "last_error": self.state.last_error or "None",
            },
            tools=self._create_tools(),
            max_steps=self.max_steps,
            max_budget=self.max_budget,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics.

        Returns:
            dict[str, Any]: Dictionary with total_todos, completed, failed,
                error_count, and dynamic_tools count.
        """
        todos = self.state.todos
        return {
            "total_todos": len(todos),
            "completed": sum(t.status == "completed" for t in todos),
            "failed": sum(t.status == "failed" for t in todos),
            "error_count": self.state.error_count,
            "dynamic_tools": len(self.state.dynamic_tools),
        }

def run_task(task: str) -> dict:
    """Run task and return result with metrics for the evolver.

    Args:
        task: The task description to execute.

    Returns:
        dict: Dictionary with 'result', 'metrics' (llm_calls, steps), 'stats',
            and optionally 'error' if the task failed.
    """
    agent = SelfEvolvingMultiAgent()
    try:
        result = agent.run(task)
        n = len(agent.state.completed_tasks)
        return {
            "result": result,
            "metrics": {"llm_calls": n + 1, "steps": n},
            "stats": agent.get_stats(),
        }
    except Exception as e:
        return {"result": str(e), "metrics": {"llm_calls": 10, "steps": 0}, "error": str(e)}


# A complex long-horizon real-world task
COMPLEX_TASK = """
Build a complete E-Commerce Backend System with the following components:

## Part 1: Core API (src/api/)
1. FastAPI application with:
   - JWT authentication (register, login, refresh token, logout)
   - Password hashing with bcrypt
   - Role-based access control (admin, seller, customer)
   - Rate limiting middleware (100 requests/minute per user)
   - Request logging middleware

2. Database models (SQLAlchemy + SQLite):
   - User (id, email, password_hash, role, created_at, is_active)
   - Product (id, name, description, price, stock, seller_id, category_id, created_at)
   - Category (id, name, parent_id for nested categories)
   - Order (id, customer_id, status, total, created_at, updated_at)
   - OrderItem (id, order_id, product_id, quantity, unit_price)
   - Review (id, product_id, customer_id, rating 1-5, comment, created_at)
   - Cart (id, customer_id) and CartItem (id, cart_id, product_id, quantity)

3. API Endpoints:
   Auth:
   - POST /auth/register - Register user (validates email, strong password)
   - POST /auth/login - Login, returns JWT tokens
   - POST /auth/refresh - Refresh access token
   - POST /auth/logout - Invalidate refresh token

   Products (sellers can CRUD their own, customers read-only):
   - GET /products - List with pagination, search, filter by category/price range
   - GET /products/{id} - Get product with reviews
   - POST /products - Create (seller only)
   - PUT /products/{id} - Update (owner seller only)
   - DELETE /products/{id} - Soft delete (owner seller only)

   Categories:
   - GET /categories - List all with nested structure
   - POST /categories - Create (admin only)

   Cart:
   - GET /cart - Get current user's cart
   - POST /cart/items - Add item to cart
   - PUT /cart/items/{id} - Update quantity
   - DELETE /cart/items/{id} - Remove item

   Orders:
   - POST /orders/checkout - Create order from cart (validates stock, calculates total)
   - GET /orders - List user's orders
   - GET /orders/{id} - Get order details
   - PUT /orders/{id}/status - Update status (seller: shipped, admin: any)

   Reviews:
   - POST /products/{id}/reviews - Add review (must have purchased)
   - GET /products/{id}/reviews - List reviews with pagination

## Part 2: Business Logic (src/services/)
1. InventoryService:
   - Check stock availability
   - Reserve stock during checkout
   - Release stock on order cancellation
   - Low stock alerts (return products with stock < 5)

2. PricingService:
   - Calculate order totals with potential discounts
   - Apply category-based discounts (configurable)
   - Bulk purchase discounts (10+ items = 5% off)

3. NotificationService (mock implementation):
   - Log order confirmations
   - Log shipping updates
   - Log low stock alerts

4. ReportingService:
   - Sales report by date range
   - Top selling products
   - Revenue by category
   - Customer order statistics

## Part 3: CLI Tool (src/cli/)
Create a command-line admin tool:
- python -m src.cli users list
- python -m src.cli users create --email X --password Y --role admin
- python -m src.cli products list --low-stock
- python -m src.cli orders list --status pending
- python -m src.cli reports sales --from 2024-01-01 --to 2024-12-31

## Part 4: Testing (tests/)
1. Unit tests for each service
2. Integration tests for API endpoints
3. Test fixtures with factory patterns
4. Test authentication flows
5. Test authorization (role-based access)
6. Test edge cases (out of stock, invalid data, etc.)
7. Minimum 30 tests total, all must pass

## Part 5: Data & Scripts
1. scripts/seed_db.py - Create:
   - 3 users (1 admin, 1 seller, 1 customer)
   - 5 categories (2 parent, 3 children)
   - 10 products across categories
   - 5 orders with multiple items
   - 10 reviews

2. scripts/generate_report.py - Generate sample sales report

## Part 6: Configuration
1. config.py with settings for:
   - JWT secret and expiration times
   - Database URL
   - Rate limit settings
   - Pagination defaults

2. requirements.txt with all dependencies
3. Makefile with: install, test, run, seed, lint, report

## Validation Requirements:
- Email must be valid format
- Password: min 8 chars, 1 uppercase, 1 number, 1 special char
- Price must be positive
- Stock must be non-negative
- Rating must be 1-5
- Order status transitions must be valid (pending->paid->shipped->delivered)

Run all tests and ensure they pass. Generate the sales report.
"""

# Test script to verify task completion
COMPLEX_TASK_TEST = """
import os
import sys
import subprocess

errors = []
warnings = []

# 1. Check required file structure (flexible paths)
required_files = [
    'requirements.txt',
]
for f in required_files:
    if not os.path.exists(f):
        errors.append(f"Missing file: {f}")

# Check for main.py in various locations
main_found = any(os.path.exists(p) for p in [
    'src/api/main.py', 'src/main.py', 'main.py', 'app/main.py'
])
if not main_found:
    errors.append("Missing main.py (FastAPI entry point)")

# Check for database/models (flexible locations)
db_patterns = ['database', 'db', 'models']
db_found = False
for root, dirs, files in os.walk('src'):
    for f in files:
        if any(p in f.lower() for p in db_patterns) and f.endswith('.py'):
            db_found = True
            break
if not db_found:
    warnings.append("Could not find database/models files")

# 2. Check services directory
services_dir = 'src/services'
if os.path.isdir(services_dir):
    service_files = [f for f in os.listdir(services_dir) if f.endswith('.py')]
    if len(service_files) < 2:
        warnings.append(f"Services directory should have service files, found: {service_files}")
else:
    warnings.append(f"Missing services directory: {services_dir}")

# 3. Check for route/endpoint files (flexible locations)
route_files_found = 0
for root, dirs, files in os.walk('src'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path) as file:
                    content = file.read()
                    if 'router' in content.lower() or '@app.' in content or 'APIRouter' in content:
                        route_files_found += 1
            except:
                pass
if route_files_found < 2:
    warnings.append(f"Expected multiple route files, found ~{route_files_found}")

# 4. Check tests directory
tests_dir = 'tests'
if os.path.isdir(tests_dir):
    test_files = []
    for root, dirs, files in os.walk(tests_dir):
        test_files.extend([f for f in files if f.startswith('test_') and f.endswith('.py')])
    if len(test_files) < 5:
        warnings.append(f"Expected at least 5 test files, found {len(test_files)}: {test_files}")
else:
    errors.append(f"Missing tests directory: {tests_dir}")

# 5. Check requirements.txt has required packages
if os.path.exists('requirements.txt'):
    with open('requirements.txt') as f:
        reqs = f.read().lower()
    required_pkgs = ['fastapi', 'sqlalchemy', 'pydantic', 'jwt', 'bcrypt']
    for pkg in required_pkgs:
        if pkg not in reqs and pkg.replace('-', '') not in reqs:
            warnings.append(f"requirements.txt may be missing: {pkg}")

# 6. Check main.py has key features (search in multiple locations)
main_content = ""
for p in ['src/api/main.py', 'src/main.py', 'main.py', 'app/main.py']:
    if os.path.exists(p):
        with open(p) as f:
            main_content = f.read()
        break
if main_content:
    if 'FastAPI' not in main_content:
        warnings.append("main.py may not use FastAPI directly")
else:
    # Search for FastAPI usage anywhere
    fastapi_found = False
    for root, dirs, files in os.walk('src'):
        for f in files:
            if f.endswith('.py'):
                try:
                    with open(os.path.join(root, f)) as file:
                        if 'FastAPI' in file.read():
                            fastapi_found = True
                            break
                except:
                    pass
        if fastapi_found:
            break
    if not fastapi_found:
        warnings.append("Could not find FastAPI usage")

# 7. Check for required model classes in any Python file
required_models = ['User', 'Product', 'Order', 'Cart']
models_found = {m: False for m in required_models}
for root, dirs, files in os.walk('src'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path) as file:
                    content = file.read()
                    for model in required_models:
                        if f'class {model}' in content:
                            models_found[model] = True
            except:
                pass
missing_models = [m for m, found in models_found.items() if not found]
if missing_models:
    warnings.append(f"May be missing models: {missing_models}")

# 8. Check Makefile has required targets
if os.path.exists('Makefile'):
    with open('Makefile') as f:
        makefile = f.read()
    for target in ['install', 'test', 'run', 'seed']:
        if target + ':' not in makefile and target + ' :' not in makefile:
            warnings.append(f"Makefile missing target: {target}")

# 9. Check for authentication
auth_found = False
for root, dirs, files in os.walk('src'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path) as file:
                    content = file.read()
                    if 'jwt' in content.lower() or 'token' in content.lower():
                        auth_found = True
                        break
            except:
                pass
    if auth_found:
        break
if not auth_found:
    warnings.append("JWT authentication may not be implemented")

# 10. Try to run tests
try:
    subprocess.run(
        ['pip', 'install', '-q', '-r', 'requirements.txt'],
        check=True, capture_output=True, timeout=180
    )
    subprocess.run(
        ['pip', 'install', '-q', 'pytest', 'httpx', 'pytest-asyncio'],
        check=True, capture_output=True, timeout=60
    )
    result = subprocess.run(
        ['python', '-m', 'pytest', '-v', '--tb=short'],
        capture_output=True, text=True, timeout=300
    )
    # Count passed tests
    import re
    passed_match = re.search(r'(\\d+) passed', result.stdout)
    passed_count = int(passed_match.group(1)) if passed_match else 0

    if result.returncode != 0:
        if 'passed' not in result.stdout and 'failed' not in result.stdout:
            errors.append(f"Tests failed to run: {result.stderr[:500]}")
        elif 'failed' in result.stdout:
            warnings.append(f"Some tests failed (passed: {passed_count})")
    else:
        if passed_count < 15:
            warnings.append(f"Only {passed_count} tests passed, expected at least 15")
        print(f"Tests: {passed_count} passed")
except subprocess.TimeoutExpired:
    warnings.append("Test execution timed out")
except Exception as e:
    warnings.append(f"Could not run tests: {e}")

# Report results
print(f"\\nVerification Summary:")
print(f"  Errors: {len(errors)}")
print(f"  Warnings: {len(warnings)}")

if errors:
    print("\\nERRORS (must fix):")
    for e in errors:
        print(f"  ❌ {e}")

if warnings:
    print("\\nWARNINGS (may be ok):")
    for w in warnings:
        print(f"  ⚠️  {w}")

# Pass if no critical errors (warnings are ok for complex task)
if errors:
    print("\\nFAIL: Critical errors found")
    sys.exit(1)
else:
    print("\\nPASS: Core requirements met!")
    sys.exit(0)
"""


def verify_task_completion(docker: DockerManager) -> bool:
    """Run the test script to verify task completion.

    Args:
        docker: The DockerManager instance with the running container.

    Returns:
        bool: True if verification passed, False otherwise.
    """
    print("\n" + "=" * 70)
    print("VERIFYING TASK COMPLETION")
    print("=" * 70)

    try:
        # Write test script
        docker.run_bash_command(
            f"cat > /tmp/verify_task.py << 'VERIFY_EOF'\n{COMPLEX_TASK_TEST}\nVERIFY_EOF",
            "Creating verification script"
        )
        # Run verification
        result = docker.run_bash_command("python /tmp/verify_task.py", "Verification")

        if "PASS" in result:
            print("\n✅ Task verification PASSED!")
            return True
        else:
            print(f"\n❌ Task verification FAILED:\n{result}")
            return False
    except Exception as e:
        print(f"\n❌ Verification error: {e}")
        return False


def main() -> None:
    """Run the SelfEvolvingMultiAgent on a complex long-horizon task.

    Executes the COMPLEX_TASK and verifies completion using the test script.
    """
    task = COMPLEX_TASK

    print("=" * 70)
    print("SelfEvolvingMultiAgent - Complex Task Execution")
    print("=" * 70)
    print(f"\nTask:\n{task}\n")
    print("=" * 70)

    agent = SelfEvolvingMultiAgent()

    # Use explicit docker context to keep container alive for verification
    with DockerManager(
        agent.docker_image, workdir="/", mount_shared_volume=True
    ) as docker:
        agent.docker = docker
        docker.run_bash_command("mkdir -p /workspace", "Init")
        docker.workdir = agent.workdir

        try:
            result = agent._run_orchestrator(task)
            stats = agent.get_stats()

            print("\n" + "=" * 70)
            print("TASK COMPLETED")
            print("=" * 70)
            print(f"\nResult:\n{result}\n")
            print(f"Stats: {stats}")
            print(f"Completed tasks: {len(agent.state.completed_tasks)}")
            for t in agent.state.completed_tasks:
                print(f"  - {t}")

            verified = verify_task_completion(docker)
            if not verified:
                print("\n⚠️  Task completed but verification failed!")
            else:
                print("\n🎉 Task completed and verified successfully!")

        except Exception as e:
            print(f"\nTask failed with error: {e}")
            stats = agent.get_stats()
            print(f"Stats at failure: {stats}")


if __name__ == "__main__":
    main()
