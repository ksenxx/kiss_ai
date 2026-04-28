cd
curl -L -o main.zip https://github.com/ksenxx/kiss_ai/archive/refs/heads/main.zip
unzip main.zip
rm main.zip
mv kiss_ai-main ~/kiss_ai
cd ~/kiss_ai
./install.sh
source ~/.zshrc
echo "Make sure that you have one of Claude Code, ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY. OPENROUTER_API_KEY, or TOGETHER_API_KEY"
code
