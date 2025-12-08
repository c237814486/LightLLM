import io
import os
import re
from tqdm import tqdm
import random
import time
import aiohttp
import base64
import json
from pathlib import Path
from typing import List, Optional, AsyncGenerator, Union, Tuple
import asyncio
from PIL import Image
import numpy as np
import cv2
import tempfile
from concurrent.futures import ThreadPoolExecutor

def load_and_resize_image(image, target_size=(644, 364)) -> Image.Image: 
    resized_image = image.resize(target_size, Image.BICUBIC)
    return resized_image

class MultiModalClient:
    def __init__(self, url: str, default_sampling_params: dict, logger=None):
        self.url = url
        self.default_sampling_params = default_sampling_params
        self._logger = logger or print  
    
    @staticmethod
    def encode_image_to_base64(image:Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    @staticmethod
    def encode_audio_to_base64(audio: Union[np.ndarray, str, bytes]) -> str:
        """将音频数据编码为base64字符串"""
        if isinstance(audio, str):
            with open(audio, 'rb') as f:
                audio_data = f.read()
        elif isinstance(audio, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, audio)
            audio_data = buffer.getvalue()
        elif isinstance(audio, bytes):
            audio_data = audio

        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        
        return base64.b64encode(audio_data).decode("utf-8")
       
    async def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        audios: Optional[List[Union[np.ndarray, str, bytes]]] = None,
        target_sizes:Optional[List[Tuple]] = None
    ) -> AsyncGenerator[str, None]:
        image_dicts = []
        if images:
            
            for i, image in enumerate(images):
                if target_sizes is not None:
                    image = load_and_resize_image(image, target_sizes[i])
                else:
                    image = load_and_resize_image(image, target_size=(644, 364))
                img_b64 = self.encode_image_to_base64(image)
                image_dicts.append({"type": "base64", "data": img_b64})
            
           
        audio_dicts = []
        if audios:
            for audio in audios:
                audio_b64 = self.encode_audio_to_base64(audio)
                audio_dicts.append({"type": "base64", "data": audio_b64})
           
        
        payload = {
                "inputs": prompt,
                "parameters": self.default_sampling_params,
                "multimodal_params": {
                    "images": image_dicts,
                    "audios": audio_dicts
                }
            }
    

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    headers={'Accept': 'text/event-stream'},
                    json=payload
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Request failed with status {response.status}")
                    
                    async for line in response.content:
                      
                        if line:
                            data = line.decode().strip()
                            if data.startswith("data:"):
                                json_data = json.loads(data[5:].strip())
                                token = json_data.get("token", {}).get("text")
                                if token:
                                    yield token
        except Exception as e:
            self._logger(f"Error during request: {e}")

def construct_prompt(query, image_num, have_audio=False, system_prompt="You are a helpful AI assistant."):

    system_prompt_format = "<|im_start|>system\n{}<|im_end|>"
    user_format = "\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    assistant_format = "{content}<|im_end|>" # llm 回复完需要拼上这个

    system = system_prompt_format.format(system_prompt)

    if not have_audio:
        user_prompt = user_format.format(content=query)
    else:
        user_prompt = user_format.format(content="<audio>") # audio token
    if image_num >0:
        image_prompt = "<|vision_start|>" + "<image>" * image_num + "<|vision_end|>"
    else:
        image_prompt = ""
    total_prompt = system + image_prompt + user_prompt
    return total_prompt


def build_inputs_qwen2(prompt_list, system_prompt:str="You are a helpful assistant."):
    prompt = ""
    system_prompt_format = "<|im_start|>system\n{content}<|im_end|>\n"
   
    user_format = "\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    assistant_format = "{content}<|im_end|>" 
    
    system_prompt = system_prompt_format.format(content=system_prompt)
    prompt += system_prompt

    for message in prompt_list:
        content = message["value"]
        if message['role'].lower() in ['human',"user"]:
            prompt += user_format.format(content=content)
        elif message["role"].lower() in ["assistant", "gpt"]:
            prompt += assistant_format.format(content=content)
        else:
            pass
    return prompt


def extract_frames_from_video(video_path, max_frames=8):
    """
    从本地或 S3 视频路径中抽取图像帧（最多 max_frames 张）。
    支持本地路径或 S3（通过 AOSS 读取）。
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)

    frames = []
    count = 0
    success = True
    while success and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        success, frame = cap.read()
        if success:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(image)
            count += frame_interval
    cap.release()

    return frames


async def main():
    client = MultiModalClient(
        url="http://127.0.0.1:10083/generate",
        default_sampling_params={
            "max_new_tokens": 2048,
            "do_sample": True,
            "top_k": 20,
            "top_p": 0.8,
            "temperature": 0.7,
            "repetition_penalty": 1.05
        }
    )

    system_prompt = ""
    prompt = construct_prompt(query="1+1=?", image_num=0, system_prompt=system_prompt)
    first_time = 0
    t1 = time.time()
    print("纯文本测试")
    async for token in client.generate(prompt):
        if first_time == 0:
            first_time = time.time() - t1
        print(token, end="", flush=True)



    print("=== 测试图像推理 ===")
    test_video = "D:/LeapFaith/LightLLM/assets/test_video.mp4"
    target_sizes = [(644, 364), (644, 364),  
                    (644, 364), (644, 364), (644, 364),(644, 364), (644,364), (644,364), (644,364),(644,364),(644,364),(644,364),
                    (1288, 728), (1288, 728), (644,364), (644,364), (644,364), (644,364)]
    
    target_sizes = [(644, 364)]*256
    max_frames = 8  # 你可以改为任意数量

    images = extract_frames_from_video(test_video, max_frames=max_frames)
    prompt = construct_prompt(query="describe the video: ", image_num=len(images))
    first_time = 0
    t1 = time.time()
    async for token in client.generate(prompt, images=images, target_sizes=target_sizes):
        if first_time == 0:
            first_time = time.time() - t1
        print(token, end="", flush=True)
    
    print("\n\n=== 测试多模态推理 (图像+音频) ===")
    audios = ["/mnt/afs/yangdeyu/GameMLLM/lol_v2s/audio_ins_1.wav"]  # 音频文件路径
   
    # 测试多模态推理
    t1 = time.time()
    first_time = 0
    prompt = construct_prompt(query="", image_num=len(images), have_audio=True)
    async for token in client.generate(prompt, images=images, audios=audios, target_sizes=target_sizes):
        if first_time == 0:
            first_time = time.time() - t1
        print(token, end="", flush=True)
    print(first_time)

asyncio.run(main())
