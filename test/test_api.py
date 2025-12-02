import requests

url = "http://localhost:10083/v1/chat/completions"

response = requests.post(url, json={
    "model": "D:\LeapFaith\models\Qwen2.5-VL-3B-Instruct",
    "messages": [{"role": "user", "content": "写一个特别长的关于将进酒的诗词赏析"}],
    "max_tokens": 2048, 
    "temperature": 0.7
})

print(response.json()["choices"][0]["message"]["content"])