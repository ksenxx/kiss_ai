#Project Setup Guide
This guide explains how to setup the project.You will need python 3.13 or later.
##Prerequisites
- python>=3.13
-  git
-   docker (optional)

Make sure you have  `uv` installed.  You can install it with:
```
curl -LsSf https://astral.sh/uv/install.sh|sh
```
##Installation steps
1.Clone the repo:
```bash
git clone https://github.com/example/project.git
cd project
```
2.Install dependencies:
```
uv sync
```
3.Run the tests
```bash
uv run pytest
```
If all tests pass you are good to go!
