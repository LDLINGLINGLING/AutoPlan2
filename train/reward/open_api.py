from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "sk-051021acb6584b4899fcbbe2cf858b15"
openai_api_base = "https://api.deepseek.com/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def request_llm(query,client):
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": query
        }],
        model='deepseek-chat',
        top_p=1,
        max_tokens=4096,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=42,
        temperature=0.7
    )

    return chat_completion.choices[0].message.content