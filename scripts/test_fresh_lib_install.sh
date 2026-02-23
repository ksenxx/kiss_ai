curl -LsSf https://astral.sh/uv/install.sh | sh

mkdir -p kiss_tmp
cd kiss_tmp
uv init --python 3.13
uv add kiss-agent-framework
uv run assistant
cd ..
rm -rf kiss_tmp
