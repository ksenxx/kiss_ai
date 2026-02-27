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
###Configuration
The config file is located at `config.yaml`.Here are the available options:
|Option|Type|Default|Description|
|---|---|---|---|
|debug|bool|false|Enable debug mode|
|port|int|8080|Server port|
|host|string|localhost|Server host|
|workers|int|4|Number of workers|

You can also set   environment variables:
- `DEBUG=true` to enable debug mode
-`PORT=9090` to change the port
- `HOST=0.0.0.0`  to bind to all interfaces

###Troubleshooting
**Problem**: Server won't start
**Solution**: Check if port is already in use:
```
lsof -i :8080
```
**Problem**:Tests failing with import errors
**Solution**:Make sure you installed all dependencies:
```
uv sync --group dev
```

> Note: if you are on macOS you might need to install xcode command line tools first by running `xcode-select --install`

For more information see the [documentation](https://example.com/docs) or open an  [issue](https://github.com/example/project/issues).
###Contributing
We welcome contributions! Please read our contributing guide before submitting PRs.All PRs must:
* pass CI checks
*  have tests
*   follow the code style
*be reviewed by at least one maintainer

---
Last updated:2025-01-15
