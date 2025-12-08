import requests
import base64
from PIL import Image
import io

url = "http://localhost:10083/v1/chat/completions"
model_path = "D:/LeapFaith/models/Qwen2.5-VL-3B-Instruct"
local_image_path = "D:/LeapFaith/LightLLM/assets/tiger.jpg"

# === 辅助函数：将图片转换为 Base64 ===
def encode_image(image_path):
    
    # 打开并转换图片
    img = Image.open(image_path).convert("RGB")
    
    # 新增：resize到140x140
    img = img.resize((560, 560))
    
    # 保存到字节流
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # 编码为base64
    base64_encoded = base64.b64encode(img_byte_arr).decode('utf-8')
    return base64_encoded



# response = requests.post(url, json={
#     "model": model_path,
#     "messages": [{"role": "user", "content": "写一个关于将进酒的诗词赏析"}],
#     "max_tokens": 2048, 
#     "temperature": 0.7
# })
# print(response.json()["choices"][0]["message"]["content"])


base64_image = encode_image(local_image_path)
response = requests.post(url, json={
    "model": model_path,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "非常非常详细的描述图片内容，并指出左下角文字撰写的图片来源"
                },
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 2048, 
    "temperature": 0.7
})
print(response.json()["choices"][0]["message"]["content"])