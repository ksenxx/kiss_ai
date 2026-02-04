# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Agent Evolver - Evolves the SelfEvolvingMultiAgent for better performance.

This module evolves the SelfEvolvingMultiAgent to optimize for:
1. Fewer LLM calls (efficiency)
2. Lower token/budget usage
3. Accurate completion of long-horizon agentic tasks
"""

from __future__ import annotations

import importlib.resources
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kiss.agents.kiss_evolve.config  # noqa: F401
import kiss.agents.self_evolving_multi_agent.config  # noqa: F401
from kiss.agents.kiss_evolve.kiss_evolve import CodeVariant, KISSEvolve
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.utils import get_config_value
from kiss.docker.docker_manager import DockerManager


def _load_base_agent_code(
    package_name: str,
    agent_file_path: str,
) -> str:
    """Load the base agent code from the specified package and file.

    This works for both:
    - Development installations (editable install with `pip install -e .`)
    - Wheel installations (pip install kiss-*.whl)

    Args:
        package_name: The package name to load from
        agent_file_path: The filename within the package

    Returns:
        The source code of the agent file as a string
    """
    try:
        package_files = importlib.resources.files(package_name)
        agent_file = package_files.joinpath(agent_file_path)
        return agent_file.read_text(encoding="utf-8")
    except (AttributeError, TypeError):
        import importlib.resources as resources

        with resources.open_text(package_name, agent_file_path) as f:
            return f.read()


@dataclass
class EvaluationTask:
    """A task for evaluating self evolving multi agent performance."""

    name: str
    description: str
    test_script: str  # Python script that returns True if task succeeded
    complexity: str = "simple"  # simple, medium, long_horizon


# Long-horizon evaluation tasks that require multi-step planning and execution
EVALUATION_TASKS = [
    # Simple task for baseline
    EvaluationTask(
        name="fibonacci",
        description="""
        Create a Python script that:
        1. Generates the first 15 Fibonacci numbers
        2. Saves them to 'fibonacci.txt', one number per line
        3. The script should be named 'fib.py'
        """,
        test_script="""
import os
if not os.path.exists('fibonacci.txt'):
    print("FAIL: fibonacci.txt not found")
    exit(1)
with open('fibonacci.txt') as f:
    nums = [int(x.strip()) for x in f.readlines() if x.strip()]
expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
if nums == expected:
    print("PASS")
    exit(0)
else:
    print(f"FAIL: Got {nums}")
    exit(1)
""",
        complexity="simple",
    ),
    # Medium complexity - multiple files and data processing
    EvaluationTask(
        name="data_pipeline",
        description="""
        Build a data processing pipeline:
        1. Create 'generator.py' that generates 100 random integers (1-1000)
           and saves to 'raw_data.txt'
        2. Create 'processor.py' that reads 'raw_data.txt', filters numbers > 500,
           sorts them, removes duplicates, and saves to 'processed.txt'
        3. Create 'analyzer.py' that reads 'processed.txt' and writes statistics
           (count, min, max, mean) to 'stats.json'
        4. Run all three scripts in order
        """,
        test_script="""
import os
import json

# Check all files exist
required = [
    'generator.py', 'processor.py', 'analyzer.py',
    'raw_data.txt', 'processed.txt', 'stats.json'
]
for f in required:
    if not os.path.exists(f):
        print(f"FAIL: {f} not found")
        exit(1)

# Verify raw_data.txt has 100 numbers
with open('raw_data.txt') as f:
    raw = [int(x.strip()) for x in f.readlines() if x.strip()]
if len(raw) != 100:
    print(f"FAIL: raw_data.txt should have 100 numbers, got {len(raw)}")
    exit(1)

# Verify processed.txt is filtered, sorted, unique
with open('processed.txt') as f:
    processed = [int(x.strip()) for x in f.readlines() if x.strip()]
if processed != sorted(set([x for x in raw if x > 500])):
    print("FAIL: processed.txt not correctly filtered/sorted/deduped")
    exit(1)

# Verify stats.json
with open('stats.json') as f:
    stats = json.load(f)
expected_keys = ['count', 'min', 'max', 'mean']
for key in expected_keys:
    if key not in stats:
        print(f"FAIL: stats.json missing '{key}'")
        exit(1)
if stats['count'] != len(processed):
    print(f"FAIL: count mismatch")
    exit(1)

print("PASS")
exit(0)
""",
        complexity="medium",
    ),
    # Long-horizon task - full project with testing
    EvaluationTask(
        name="calculator_project",
        description="""
        Build a complete calculator project:
        1. Create 'calculator.py' with a Calculator class that has methods:
           add, subtract, multiply, divide, power, sqrt
           - Division by zero should raise ValueError
           - sqrt of negative should raise ValueError
        2. Create 'test_calculator.py' with unit tests for all Calculator
           methods including edge cases
        3. Create 'main.py' that demonstrates all calculator operations
           and saves results to 'demo_output.txt'
        4. Run the tests and main script
        """,
        test_script="""
import os
import sys

# Check all files exist
required = ['calculator.py', 'test_calculator.py', 'main.py', 'demo_output.txt']
for f in required:
    if not os.path.exists(f):
        print(f"FAIL: {f} not found")
        exit(1)

# Import and test calculator
sys.path.insert(0, '.')
try:
    from calculator import Calculator
    calc = Calculator()

    # Test basic operations
    assert calc.add(2, 3) == 5, "add failed"
    assert calc.subtract(5, 3) == 2, "subtract failed"
    assert calc.multiply(4, 3) == 12, "multiply failed"
    assert calc.divide(10, 2) == 5, "divide failed"
    assert calc.power(2, 3) == 8, "power failed"
    assert abs(calc.sqrt(16) - 4) < 0.001, "sqrt failed"

    # Test error cases
    try:
        calc.divide(1, 0)
        print("FAIL: divide by zero should raise ValueError")
        exit(1)
    except ValueError:
        pass

    try:
        calc.sqrt(-1)
        print("FAIL: sqrt of negative should raise ValueError")
        exit(1)
    except ValueError:
        pass

except Exception as e:
    print(f"FAIL: Calculator error: {e}")
    exit(1)

# Verify demo_output.txt has content
with open('demo_output.txt') as f:
    content = f.read()
if len(content) < 50:
    print("FAIL: demo_output.txt too short")
    exit(1)

print("PASS")
exit(0)
""",
        complexity="long_horizon",
    ),
    # Long-horizon task - web scraper simulation with multiple components
    EvaluationTask(
        name="text_analyzer_suite",
        description="""
        Build a text analysis suite:
        1. Create 'text_utils.py' with functions: word_count, char_count,
           sentence_count, word_frequency, find_longest_word
        2. Create 'file_handler.py' with functions: read_text_file,
           write_json_file, append_to_file
        3. Create 'analyzer.py' that uses both modules to:
           - Generate a sample text file 'sample.txt' with at least 5 paragraphs
             (each paragraph has 3+ sentences)
           - Analyze the text and save complete analysis to 'analysis.json'
             (word_count, char_count, sentence_count, top_10_words, longest_word)
           - Save a summary report to 'report.txt'
        4. Create 'test_utils.py' with tests for text_utils functions
        5. Run tests and analyzer
        """,
        test_script="""
import os
import json
import sys

# Check all files exist
required = [
    'text_utils.py', 'file_handler.py', 'analyzer.py', 'test_utils.py',
    'sample.txt', 'analysis.json', 'report.txt'
]
for f in required:
    if not os.path.exists(f):
        print(f"FAIL: {f} not found")
        exit(1)

# Check sample.txt has enough content
with open('sample.txt') as f:
    text = f.read()
paragraphs = [p.strip() for p in text.split('\\n\\n') if p.strip()]
if len(paragraphs) < 5:
    print(f"FAIL: sample.txt should have at least 5 paragraphs, got {len(paragraphs)}")
    exit(1)

# Check analysis.json
with open('analysis.json') as f:
    analysis = json.load(f)
required_keys = [
    'word_count', 'char_count', 'sentence_count', 'top_10_words', 'longest_word'
]
for key in required_keys:
    if key not in analysis:
        print(f"FAIL: analysis.json missing '{key}'")
        exit(1)

# Verify word_count is reasonable
if analysis['word_count'] < 100:
    print(f"FAIL: word_count too low ({analysis['word_count']})")
    exit(1)

# Test text_utils module
sys.path.insert(0, '.')
try:
    from text_utils import word_count, char_count, sentence_count
    test_text = "Hello world. This is a test."
    assert word_count(test_text) == 6, "word_count failed"
    assert char_count(test_text) >= 20, "char_count failed"
except Exception as e:
    print(f"FAIL: text_utils error: {e}")
    exit(1)

# Check report.txt has content
with open('report.txt') as f:
    report = f.read()
if len(report) < 100:
    print("FAIL: report.txt too short")
    exit(1)

print("PASS")
exit(0)
""",
        complexity="long_horizon",
    ),
    # Very complex long-horizon task - full E-Commerce backend
    EvaluationTask(
        name="ecommerce_backend",
        description="""
        Build a complete E-Commerce Backend System:

        1. Core API (src/api/):
           - FastAPI application with JWT authentication (register, login, refresh)
           - Password hashing with bcrypt
           - Role-based access control (admin, seller, customer)

        2. Database models (SQLAlchemy + SQLite):
           - User (id, email, password_hash, role, created_at)
           - Product (id, name, description, price, stock, seller_id, category_id)
           - Category (id, name, parent_id for nested categories)
           - Order (id, customer_id, status, total, created_at)
           - OrderItem (id, order_id, product_id, quantity, unit_price)
           - Cart (id, customer_id) and CartItem (id, cart_id, product_id, quantity)

        3. API Endpoints:
           - POST /auth/register, POST /auth/login - Authentication
           - GET /products - List with pagination, search, filter by category
           - POST /products - Create product (seller only)
           - GET /cart, POST /cart/items - Cart management
           - POST /orders/checkout - Create order from cart (validates stock)
           - GET /orders - List user's orders

        4. Business Logic (src/services/):
           - InventoryService: Check stock, reserve during checkout
           - PricingService: Calculate totals, bulk discounts (10+ items = 5% off)

        5. Testing (tests/):
           - At least 15 tests covering auth, products, cart, orders
           - All tests must pass

        6. Scripts:
           - scripts/seed_db.py: Create sample users, categories, products
           - requirements.txt with all dependencies
           - config.py with JWT settings

        Run seed script and all tests.
        """,
        test_script="""
import os
import sys
import subprocess

errors = []

# Check for key files
key_patterns = ['main.py', 'requirements.txt']
for pattern in key_patterns:
    found = False
    for root, dirs, files in os.walk('.'):
        if pattern in files:
            found = True
            break
    if not found:
        errors.append(f"Missing: {pattern}")

# Check for database/models
db_found = False
for root, dirs, files in os.walk('src'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path) as file:
                    content = file.read()
                    if 'User' in content and 'Product' in content:
                        db_found = True
                        break
            except:
                pass
if not db_found:
    errors.append("Missing User/Product models")

# Check for auth
auth_found = False
for root, dirs, files in os.walk('src'):
    for f in files:
        if f.endswith('.py'):
            try:
                with open(os.path.join(root, f)) as file:
                    if 'jwt' in file.read().lower():
                        auth_found = True
                        break
            except:
                pass
if not auth_found:
    errors.append("JWT auth not found")

# Install and run tests
try:
    subprocess.run(['pip', 'install', '-q', '-r', 'requirements.txt'],
                   capture_output=True, timeout=180)
    subprocess.run(['pip', 'install', '-q', 'pytest', 'httpx'],
                   capture_output=True, timeout=60)
    result = subprocess.run(['python', '-m', 'pytest', '-v', '--tb=short'],
                            capture_output=True, text=True, timeout=300)
    import re
    match = re.search(r'(\\d+) passed', result.stdout)
    passed = int(match.group(1)) if match else 0
    if passed < 10:
        errors.append(f"Only {passed} tests passed, expected at least 10")
    else:
        print(f"Tests: {passed} passed")
except Exception as e:
    errors.append(f"Test error: {e}")

if errors:
    print("FAIL:")
    for e in errors:
        print(f"  - {e}")
    exit(1)
print("PASS")
exit(0)
""",
        complexity="long_horizon",
    ),
    # Very complex: Full-stack blog platform
    EvaluationTask(
        name="blog_platform",
        description="""
        Build a complete blog platform with the following:

        1. Backend API (src/api/):
           - FastAPI with user authentication (register, login with JWT)
           - SQLAlchemy models: User, Post, Comment, Tag, PostTag (many-to-many)
           - Endpoints:
             * POST /users/register, POST /users/login
             * GET/POST /posts (list with pagination, create)
             * GET/PUT/DELETE /posts/{id} (author only can edit/delete)
             * POST /posts/{id}/comments (authenticated users)
             * GET /posts/{id}/comments
             * GET /tags, GET /tags/{name}/posts
           - Posts support markdown content and multiple tags

        2. Services (src/services/):
           - SearchService: Full-text search in post titles and content
           - StatsService: Post views count, popular posts, author statistics

        3. CLI (src/cli/):
           - Create admin user
           - List all posts with stats
           - Export posts to JSON

        4. Tests (tests/):
           - At least 12 tests for auth, posts, comments, tags
           - Test search functionality

        5. Scripts:
           - seed_db.py: Create 2 users, 5 posts with tags, 10 comments
           - requirements.txt, config.py

        Seed the database and run all tests.
        """,
        test_script="""
import os
import subprocess

errors = []

# Check key components exist
for pattern in ['requirements.txt']:
    if not os.path.exists(pattern):
        errors.append(f"Missing {pattern}")

# Check for models
models_found = {'User': False, 'Post': False, 'Comment': False, 'Tag': False}
for root, dirs, files in os.walk('src'):
    for f in files:
        if f.endswith('.py'):
            try:
                with open(os.path.join(root, f)) as file:
                    content = file.read()
                    for model in models_found:
                        if f'class {model}' in content:
                            models_found[model] = True
            except:
                pass
missing = [m for m, found in models_found.items() if not found]
if missing:
    errors.append(f"Missing models: {missing}")

# Run tests
try:
    subprocess.run(['pip', 'install', '-q', '-r', 'requirements.txt'],
                   capture_output=True, timeout=180)
    subprocess.run(['pip', 'install', '-q', 'pytest', 'httpx'],
                   capture_output=True, timeout=60)
    result = subprocess.run(['python', '-m', 'pytest', '-v', '--tb=short'],
                            capture_output=True, text=True, timeout=300)
    import re
    match = re.search(r'(\\d+) passed', result.stdout)
    passed = int(match.group(1)) if match else 0
    if passed < 8:
        errors.append(f"Only {passed} tests passed, expected at least 8")
    else:
        print(f"Tests: {passed} passed")
except Exception as e:
    errors.append(f"Test error: {e}")

if errors:
    print("FAIL:")
    for e in errors:
        print(f"  - {e}")
    exit(1)
print("PASS")
exit(0)
""",
        complexity="long_horizon",
    ),
    # Very complex: Task scheduler system
    EvaluationTask(
        name="task_scheduler",
        description="""
        Build a distributed task scheduler system:

        1. Core (src/core/):
           - Task model: id, name, cron_expression, handler, status, last_run, next_run
           - TaskQueue: Priority queue for pending tasks
           - CronParser: Parse cron expressions to calculate next run time

        2. Scheduler (src/scheduler/):
           - Scheduler class that:
             * Loads tasks from SQLite database
             * Calculates next run times from cron expressions
             * Executes tasks when due (run Python functions by name)
             * Tracks execution history (TaskExecution model)
             * Handles task failures with retry logic (max 3 retries)

        3. API (src/api/):
           - FastAPI endpoints:
             * GET /tasks - List all tasks with status
             * POST /tasks - Create new scheduled task
             * PUT /tasks/{id} - Update task (pause/resume)
             * DELETE /tasks/{id} - Remove task
             * GET /tasks/{id}/history - Execution history
             * POST /tasks/{id}/run - Manually trigger task

        4. Sample Tasks (src/tasks/):
           - cleanup_task.py: Delete old files
           - report_task.py: Generate daily report
           - health_check.py: Check system status

        5. Tests (tests/):
           - Test cron parser (every minute, hourly, daily, weekly)
           - Test task execution and retry logic
           - Test API endpoints
           - At least 10 tests

        6. Files:
           - requirements.txt, config.py
           - seed_db.py: Create sample scheduled tasks

        Seed database and run tests.
        """,
        test_script="""
import os
import subprocess

errors = []

# Check requirements
if not os.path.exists('requirements.txt'):
    errors.append("Missing requirements.txt")

# Check for scheduler components
scheduler_found = False
cron_found = False
for root, dirs, files in os.walk('src'):
    for f in files:
        if f.endswith('.py'):
            try:
                with open(os.path.join(root, f)) as file:
                    content = file.read()
                    if 'Scheduler' in content or 'scheduler' in content.lower():
                        scheduler_found = True
                    if 'cron' in content.lower():
                        cron_found = True
            except:
                pass
if not scheduler_found:
    errors.append("Scheduler not found")
if not cron_found:
    errors.append("Cron parsing not found")

# Run tests
try:
    subprocess.run(['pip', 'install', '-q', '-r', 'requirements.txt'],
                   capture_output=True, timeout=180)
    subprocess.run(['pip', 'install', '-q', 'pytest', 'httpx'],
                   capture_output=True, timeout=60)
    result = subprocess.run(['python', '-m', 'pytest', '-v', '--tb=short'],
                            capture_output=True, text=True, timeout=300)
    import re
    match = re.search(r'(\\d+) passed', result.stdout)
    passed = int(match.group(1)) if match else 0
    if passed < 6:
        errors.append(f"Only {passed} tests passed, expected at least 6")
    else:
        print(f"Tests: {passed} passed")
except Exception as e:
    errors.append(f"Test error: {e}")

if errors:
    print("FAIL:")
    for e in errors:
        print(f"  - {e}")
    exit(1)
print("PASS")
exit(0)
""",
        complexity="long_horizon",
    ),
    # Very complex: ML pipeline system
    EvaluationTask(
        name="ml_pipeline",
        description="""
        Build a machine learning pipeline system:

        1. Data Layer (src/data/):
           - DataLoader: Load CSV files, handle missing values
           - DataSplitter: Train/test/validation splits
           - FeatureExtractor: Numeric scaling, categorical encoding

        2. Models (src/models/):
           - BaseModel: Abstract class with fit, predict, evaluate methods
           - LinearRegressor: Simple linear regression from scratch
           - DecisionTree: Basic decision tree classifier from scratch
           - ModelRegistry: Save/load models with versioning

        3. Pipeline (src/pipeline/):
           - Pipeline class that chains: load -> preprocess -> train -> evaluate
           - Support for cross-validation
           - Metrics: accuracy, precision, recall, MSE, R2

        4. API (src/api/):
           - FastAPI endpoints:
             * POST /datasets - Upload CSV dataset
             * GET /datasets - List datasets
             * POST /train - Train model on dataset
             * POST /predict - Make predictions
             * GET /models - List trained models with metrics

        5. CLI (src/cli/):
           - Train model from command line
           - Evaluate model on test data
           - Export predictions to CSV

        6. Tests (tests/):
           - Test data loading and preprocessing
           - Test model training and prediction
           - Test pipeline end-to-end
           - At least 10 tests

        7. Sample Data:
           - Create sample_data.csv with 100 rows, 5 features
           - requirements.txt (no sklearn allowed - implement from scratch)

        Run tests to verify pipeline works.
        """,
        test_script="""
import os
import subprocess

errors = []

if not os.path.exists('requirements.txt'):
    errors.append("Missing requirements.txt")

# Check for ML components
components = {'DataLoader': False, 'Pipeline': False, 'predict': False}
for root, dirs, files in os.walk('src'):
    for f in files:
        if f.endswith('.py'):
            try:
                with open(os.path.join(root, f)) as file:
                    content = file.read()
                    for comp in components:
                        if comp in content:
                            components[comp] = True
            except:
                pass
missing = [c for c, found in components.items() if not found]
if missing:
    errors.append(f"Missing components: {missing}")

# Run tests
try:
    subprocess.run(['pip', 'install', '-q', '-r', 'requirements.txt'],
                   capture_output=True, timeout=180)
    subprocess.run(['pip', 'install', '-q', 'pytest', 'httpx', 'numpy', 'pandas'],
                   capture_output=True, timeout=60)
    result = subprocess.run(['python', '-m', 'pytest', '-v', '--tb=short'],
                            capture_output=True, text=True, timeout=300)
    import re
    match = re.search(r'(\\d+) passed', result.stdout)
    passed = int(match.group(1)) if match else 0
    if passed < 6:
        errors.append(f"Only {passed} tests passed, expected at least 6")
    else:
        print(f"Tests: {passed} passed")
except Exception as e:
    errors.append(f"Test error: {e}")

if errors:
    print("FAIL:")
    for e in errors:
        print(f"  - {e}")
    exit(1)
print("PASS")
exit(0)
""",
        complexity="long_horizon",
    ),
]


COMPLEXITY_WEIGHTS = {"simple": 1.0, "medium": 2.0, "long_horizon": 3.0}


def evaluate_agent_code(
    agent_code: str,
    tasks: list[EvaluationTask],
) -> dict[str, Any]:
    """Evaluate agent code on a set of evaluation tasks.

    Compiles and executes the agent code, runs it on each task in a Docker
    container, and computes fitness scores based on accuracy, efficiency,
    and speed.

    Args:
        agent_code: The Python source code of the agent to evaluate.
        tasks: List of EvaluationTask instances to test against.

    Returns:
        Dictionary containing:
        - fitness: Overall fitness score (0.0 to 1.0)
        - metrics: Dict with tasks_passed, tasks_total, total_time,
          avg_time, total_llm_calls, avg_llm_calls, efficiency_score
        - artifacts: Per-task results with passed status, time, llm_calls
        - error: Error message if evaluation failed, None otherwise
    """
    n = len(tasks)
    results: dict[str, Any] = {
        "fitness": 0.0,
        "metrics": {"tasks_passed": 0, "tasks_total": n, "total_time": 0.0,
                    "avg_time": 0.0, "total_llm_calls": 0, "avg_llm_calls": 0.0,
                    "efficiency_score": 0.0},
        "artifacts": {},
        "error": None,
    }

    namespace: dict[str, Any] = {}
    try:
        exec(agent_code, namespace)
        run_task_fn = namespace.get("run_task")
        if not run_task_fn:
            results["error"] = "Agent code does not define run_task function"
            return results
    except Exception as e:
        results["error"] = f"Failed to compile agent code: {e}"
        return results

    passed, total_time, total_llm_calls = 0, 0.0, 0
    complexity_score, max_complexity = 0.0, 0.0

    for task in tasks:
        weight = COMPLEXITY_WEIGHTS.get(task.complexity, 1.0)
        max_complexity += weight
        task_start = time.time()
        task_passed, llm_calls = False, 10  # default penalty

        try:
            result = run_task_fn(task.description)
            if isinstance(result, dict):
                llm_calls = result.get("metrics", {}).get("llm_calls", 5)
            else:
                llm_calls = 5
            total_llm_calls += llm_calls

            with DockerManager(
                "python:3.12-slim", workdir="/workspace", mount_shared_volume=True
            ) as docker:
                script = f"cat > /tmp/test.py << 'EOF'\n{task.test_script}\nEOF"
                docker.run_bash_command(script, "Test setup")
                if "PASS" in docker.run_bash_command("python /tmp/test.py", "Test"):
                    task_passed = True
                    passed += 1
                    complexity_score += weight
        except Exception as e:
            results["artifacts"][task.name] = f"Error: {e}"

        task_time = time.time() - task_start
        total_time += task_time
        results["artifacts"][task.name] = {"passed": task_passed, "time": task_time,
                                            "llm_calls": llm_calls, "complexity": task.complexity}

    # Metrics
    m = results["metrics"]
    m["tasks_passed"], m["total_time"], m["total_llm_calls"] = passed, total_time, total_llm_calls
    m["avg_time"] = total_time / n if n else 0
    m["avg_llm_calls"] = total_llm_calls / n if n else 0

    # Fitness: accuracy (60%) + efficiency (25%) + speed (15%)
    accuracy = complexity_score / max_complexity if max_complexity else 0
    efficiency = max(0, min(1, (15 - m["avg_llm_calls"]) / 12))
    speed = max(0, min(1, (180 - m["avg_time"]) / 150))
    m["efficiency_score"] = efficiency
    results["fitness"] = accuracy * 0.60 + efficiency * 0.25 + speed * 0.15

    return results


def create_code_agent_wrapper(default_model: str) -> Callable[..., str]:
    """Create a code agent wrapper function for KISSEvolve.

    Args:
        default_model: The default LLM model name to use if not specified.

    Returns:
        A wrapper function that takes prompt_template, arguments, and
        optional model_name, and returns the agent's response string.
    """
    def wrapper(
        prompt_template: str,
        arguments: dict[str, str],
        model_name: str | None = None,
    ) -> str:
        return KISSAgent(name="CodeEvolver").run(
            model_name=model_name or default_model,
            prompt_template=prompt_template,
            arguments=arguments,
            is_agentic=True,
            max_steps=100,
            max_budget=10.0,
        )
    return wrapper


class AgentEvolver:
    """Evolves agent code for efficiency and accuracy.

    Uses KISSEvolve to optimize agent code through evolutionary algorithms,
    evaluating candidates on a set of tasks ranging from simple to long-horizon.
    """

    def __init__(
        self,
        package_name: str,
        agent_file_path: str,
        model_name: str | None = None,
        tasks: list[EvaluationTask] | None = None,
        focus_on_efficiency: bool = True,
    ):
        """Initialize the AgentEvolver.

        Args:
            package_name: The Python package containing the agent code.
            agent_file_path: Path to the agent file within the package.
            model_name: LLM model for evolution (uses config default if None).
            tasks: Evaluation tasks (uses EVALUATION_TASKS if None).
            focus_on_efficiency: If True, optimize for fewer LLM calls;
                otherwise optimize for accuracy.
        """
        cfg = DEFAULT_CONFIG.self_evolving_multi_agent  # type: ignore[attr-defined]
        evolve_cfg = DEFAULT_CONFIG.kiss_evolve  # type: ignore[attr-defined]

        self.model_name = get_config_value(model_name, cfg, "evolver_model")
        self.population_size = evolve_cfg.population_size
        self.max_generations = evolve_cfg.max_generations
        self.mutation_rate = evolve_cfg.mutation_rate
        self.elite_size = evolve_cfg.elite_size
        self.tasks = tasks or EVALUATION_TASKS
        self.focus_on_efficiency = focus_on_efficiency
        self.agent_file_path = agent_file_path
        self.base_agent_code = _load_base_agent_code(package_name, agent_file_path)

    def evolve(self) -> CodeVariant:
        """Run the evolutionary optimization to find the best agent code.

        Uses KISSEvolve with configured population size, generations, and
        mutation rate to evolve agent code that maximizes fitness on the
        evaluation tasks.

        Returns:
            The best CodeVariant found during evolution, containing the
            optimized code, fitness score, and metrics.
        """
        print(f"Evolving: model={self.model_name}, pop={self.population_size}, "
              f"gens={self.max_generations}, tasks={len(self.tasks)}")

        efficiency_instructions = """
## Optimization Goals ##
Evolve the multi-agent to be MORE EFFICIENT while maintaining accuracy.

### Key Improvements ###
1. Reduce LLM calls - batch operations, use direct tools for simple tasks
2. Optimize planning - comprehensive upfront plans, group related operations
3. Streamline prompts - concise but complete
4. You can change the agent architecture to be more efficient, but you have
   to use KISSAgent API to create and run agents.
5. You can also modify the agent architecture to be more scalable and robust for
   very very long-horizon tasks.
6. Smart sub-agent usage - only for complex sub-tasks
7. You can also allow sub-agents to plan their own tasks and execute them
   with their own tools and sub-agents.
8. You can search the web for information about various agentic patterns
   that solves long-horizon tasks scalably, efficiently and accurately.

### Code Structure ###
- run_task must return dict with 'result' and 'metrics' (including 'llm_calls')
- Keep error handling lightweight
- Keep tool interface (run_bash, read_file, write_file)
"""
        accuracy_instructions = """
Focus on: task understanding, error recovery, code verification, long-horizon tasks.
"""

        if self.focus_on_efficiency:
            instructions = efficiency_instructions
        else:
            instructions = accuracy_instructions
        evolver = KISSEvolve(
            code_agent_wrapper=create_code_agent_wrapper('gemini-3-pro-preview'),
            initial_code=self.base_agent_code,
            evaluation_fn=lambda code: evaluate_agent_code(code, self.tasks),
            model_names=[(self.model_name, 1.0)],
            population_size=self.population_size,
            max_generations=self.max_generations,
            mutation_rate=self.mutation_rate,
            elite_size=self.elite_size,
            extra_coding_instructions=instructions,
        )

        best = evolver.evolve()
        m = best.metrics or {}
        passed, total = m.get('tasks_passed'), m.get('tasks_total')
        print(f"Done: fitness={best.fitness:.4f}, passed={passed}/{total}, "
              f"avg_llm={m.get('avg_llm_calls', 0):.1f}")
        return best

    def save_best(self, variant: CodeVariant, path: str | None = None) -> None:
        """Save the best variant's code to a file.

        Args:
            variant: The CodeVariant to save.
            path: File path to save to. If None, saves to the artifact
                directory under self_evolving_multi_agent/.

        Returns:
            None.
        """
        if path is None:
            output_dir = Path(DEFAULT_CONFIG.agent.artifact_dir) / "self_evolving_multi_agent"
            output_dir.mkdir(parents=True, exist_ok=True)
            path = str(output_dir / self.agent_file_path)
        Path(path).write_text(variant.code)
        print(f"Saved to {path}")

    def run_baseline_evaluation(self) -> dict[str, Any]:
        """Run evaluation on base agent code to establish a baseline.

        Evaluates the original agent code on all tasks to measure
        the starting point before evolution.

        Returns:
            Dictionary with fitness score and metrics from evaluate_agent_code.
        """
        results = evaluate_agent_code(self.base_agent_code, self.tasks)
        print(f"Baseline: fitness={results['fitness']:.4f}, metrics={results['metrics']}")
        return results


def main() -> None:
    """Evolve SelfEvolvingMultiAgent for efficiency and accuracy.

    Loads configuration, creates an AgentEvolver, runs baseline evaluation,
    evolves the agent, and saves the best variant.

    Returns:
        None.
    """
    config = DEFAULT_CONFIG.self_evolving_multi_agent  # type: ignore[attr-defined]

    evolver = AgentEvolver(
        package_name="kiss.agents.self_evolving_multi_agent",
        agent_file_path="multi_agent.py",
        model_name=config.evolver_model,
        focus_on_efficiency=True,
    )

    baseline = evolver.run_baseline_evaluation()
    best = evolver.evolve()

    improvement = ((best.fitness - baseline['fitness']) / baseline['fitness'] * 100
                   if baseline['fitness'] > 0 else 0)
    print(f"Improvement: {improvement:+.1f}%")
    evolver.save_best(best)


if __name__ == "__main__":
    main()
