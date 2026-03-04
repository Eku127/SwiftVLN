#!/usr/bin/env python3
"""
批量地标检测脚本
使用 Qwen3-VL-235B 模型分析遥感图像序列，检测独特地标
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
from openai import OpenAI

# 配置
API_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "Qwen3-VL-235B"

PROMPT = """I will provide you with a sequence of remote sensing images taken at various waypoints (corners) along a path. 
Your mission is to find **Unique Landmarks** that are distinct from the repetitive background.

CRITICAL RULES:
1. **Uniqueness**: A feature is NOT a landmark if it appears in almost every image.
2. **Contrast**: A landmark must be a specific object that breaks visual monotony.
3. **Positioning**: If a landmark is found, you MUST specify its relative position in the image (e.g., "top-left", "right side", "center", "bottom-right").
4. **Efficiency**: If no unique landmark is found, set "landmark_description" to null and "has_landmark" to false.

Response Format (JSON):
- "has_landmark": boolean.
- "landmark_description": string or null. If true, include both the object and its position (e.g., "A blue-roofed house at the top-left corner").

Example Output:
{
  "0": {"has_landmark": true, "landmark_description": "A standalone red-roofed warehouse at the bottom-right."},
  "1": {"has_landmark": false, "landmark_description": null}
}

Please analyze all the provided images and return a JSON object with results for each image index."""


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_files(directory: str) -> list:
    """获取目录下所有图片文件，按文件名排序"""
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    image_files = []
    
    for file in Path(directory).iterdir():
        if file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    # 按文件名排序
    image_files.sort(key=lambda x: x.name)
    return image_files


def build_content(image_files: list) -> list:
    """构建请求内容，包含所有图片和 prompt"""
    content = []
    
    # 添加所有图片
    for idx, image_path in enumerate(image_files):
        image_base64 = encode_image_to_base64(str(image_path))
        
        # 判断图片类型
        suffix = image_path.suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif suffix == ".png":
            mime_type = "image/png"
        elif suffix == ".gif":
            mime_type = "image/gif"
        elif suffix == ".webp":
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"
        
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
        })
        print(f"  已加载图片 [{idx}]: {image_path.name}")
    
    # 添加文本 prompt
    content.append({
        "type": "text",
        "text": PROMPT
    })
    
    return content


def detect_landmarks(directory: str, max_tokens: int = 2048, api_url: str = API_BASE_URL) -> dict:
    """检测目录下所有图片中的地标"""
    
    # 获取图片文件
    image_files = get_image_files(directory)
    if not image_files:
        print(f"错误: 目录 {directory} 中没有找到图片文件")
        return {}
    
    print(f"\n找到 {len(image_files)} 张图片:")
    
    # 构建请求内容
    content = build_content(image_files)
    
    # 创建 OpenAI 客户端
    client = OpenAI(
        base_url=api_url,
        api_key="EMPTY"
    )
    
    print(f"\n正在发送请求到 {api_url}...")
    print(f"模型: {MODEL_NAME}")
    print(f"图片数量: {len(image_files)}")
    
    # 发送请求
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": content
        }],
        max_tokens=max_tokens,
        temperature=0.1  # 低温度以获得更确定的输出
    )
    
    # 提取响应
    result_text = response.choices[0].message.content
    
    print(f"\n=== 模型响应 ===")
    print(result_text)
    
    # 尝试解析 JSON
    try:
        # 尝试从响应中提取 JSON
        import re
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            result = json.loads(json_match.group())
            return result
    except json.JSONDecodeError:
        print("\n警告: 无法解析为 JSON，返回原始文本")
    
    return {"raw_response": result_text}


def main():
    parser = argparse.ArgumentParser(description="批量地标检测 - 使用 Qwen3-VL-235B 分析遥感图像")
    parser.add_argument("directory", type=str, help="包含图片的目录路径")
    parser.add_argument("--max-tokens", type=int, default=2048, help="最大输出 tokens (默认: 2048)")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 文件路径")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1", help="API 地址 (默认: http://localhost:8000/v1)")
    
    args = parser.parse_args()
    
    api_url = args.api_url
    
    # 检查目录是否存在
    if not os.path.isdir(args.directory):
        print(f"错误: 目录不存在: {args.directory}")
        sys.exit(1)
    
    print(f"=== 批量地标检测 ===")
    print(f"目录: {args.directory}")
    
    # 执行检测
    result = detect_landmarks(args.directory, args.max_tokens, api_url)
    
    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")
    
    return result


if __name__ == "__main__":
    main()
