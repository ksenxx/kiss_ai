#!/bin/bash
set -e

install_git() {
  if command -v git &> /dev/null; then
    return 0
  fi
  echo "git not found in PATH; attempting to install git..."
  case "$(uname -s)" in
    Darwin)
      if command -v brew &> /dev/null; then
        brew install git
      else
        # xcode-select --install opens a GUI dialog; fall back if brew is absent
        xcode-select --install 2> /dev/null || true
        echo "Please install Homebrew or Xcode Command Line Tools to get git."
      fi
      ;;
    Linux)
      if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y git
      elif command -v dnf &> /dev/null; then
        sudo dnf install -y git
      elif command -v yum &> /dev/null; then
        sudo yum install -y git
      elif command -v pacman &> /dev/null; then
        sudo pacman -Sy --noconfirm git
      elif command -v zypper &> /dev/null; then
        sudo zypper install -y git
      elif command -v apk &> /dev/null; then
        sudo apk add git
      else
        echo "No supported package manager found; cannot install git automatically."
      fi
      ;;
    *)
      echo "Unsupported OS for automatic git install."
      ;;
  esac
  command -v git &> /dev/null
}

install_git || true

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
