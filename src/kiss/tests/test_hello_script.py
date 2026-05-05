import subprocess

def test_hello_script():
    result = subprocess.run(["python", "src/kiss/scripts/hello.py"], capture_output=True, text=True)
    assert result.stdout.strip() == "Hello, world!"
