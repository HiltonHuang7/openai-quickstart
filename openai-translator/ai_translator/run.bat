set OPENAI_API_KEY="your_key"
set GLM_MODEL_URL="your_url"
python ai_translator/main.py --model_type OpenAIModel --openai_api_key $OPENAI_API_KEY --file_format markdown --book tests/test.pdf --openai_model gpt-3.5-turbo --target_language=spanish --api_base_url https://api.xiaoai.plus