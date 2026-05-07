#!/bin/bash
set -e

cd
if [ -d ~/kiss_ai ]; then
  if [ -d ~/kiss_ai/.git ]; then
    cd ~/kiss_ai
    git pull
  else
    rm -rf ~/kiss_ai
    if command -v git &> /dev/null; then
      git clone https://github.com/ksenxx/kiss_ai.git ~/kiss_ai
    else
      curl -L -o main.zip https://github.com/ksenxx/kiss_ai/archive/refs/heads/main.zip
      unzip main.zip
      rm main.zip
      mv kiss_ai-main ~/kiss_ai
    fi
  fi
else
  if command -v git &> /dev/null; then
    git clone https://github.com/ksenxx/kiss_ai.git ~/kiss_ai
  else
    curl -L -o main.zip https://github.com/ksenxx/kiss_ai/archive/refs/heads/main.zip
    unzip main.zip
    rm main.zip
    mv kiss_ai-main ~/kiss_ai
  fi
fi
cd ~/kiss_ai
./install.sh
export PATH="$HOME/.local/bin:$PATH"
echo "Make sure that you have one of Claude Code, ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY. OPENROUTER_API_KEY, or TOGETHER_API_KEY"
if command -v code &>/dev/null; then
  code
else
  echo "Open a new terminal and run 'code' to launch VS Code."
fi
