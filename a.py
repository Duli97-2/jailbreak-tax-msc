import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

completion = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[
        {"role": "system", "content": "Respond only with valid JSON: {\"test\": \"hello\"}"},
        {"role": "user", "content": "Say hello in JSON"},
    ],
    max_completion_tokens=5000,
)

print("Full response:")
print(completion)
print("\nMessage object:")
print(completion.choices[0].message)
print("\nContent:")
print(repr(completion.choices[0].message.content))