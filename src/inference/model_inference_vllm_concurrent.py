"""
vLLM并发客户端模型推理模块

此模块提供了使用vLLM服务进行并发模型推理的功能，大幅提升推理效率。
"""

import requests
import json
import time
import re
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys

# 添加项目根目录和core目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
core_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core')
sys.path.append(project_root)
sys.path.append(core_dir)


class VLLMConcurrentInference:
    """
    vLLM并发模型推理类，通过并发HTTP API调用vLLM服务进行推理。
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:6688",
                 model_name: str = "Qwen3-4B-Base",
                 timeout: int = 180,
                 max_concurrent: int = 8):
        """
        初始化vLLM并发模型推理类。
        
        参数:
            base_url: vLLM服务器的基础URL
            model_name: 服务模型名称
            timeout: 请求超时时间（秒）
            max_concurrent: 最大并发数
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.session = requests.Session()
        
    def load_model(self) -> None:
        """
        检查vLLM服务器连接状态。
        """
        print(f"检查vLLM服务器连接: {self.base_url}")
        start_time = time.time()
        
        try:
            # 检查健康状态
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                print("vLLM服务器连接成功")
            else:
                raise Exception(f"健康检查失败: {response.status_code}")
                
            # 检查模型列表
            models_response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            if models_response.status_code == 200:
                models_data = models_response.json()
                available_models = [model['id'] for model in models_data.get('data', [])]
                print(f"可用模型: {available_models}")
                
                if self.model_name not in available_models:
                    print(f"警告: 指定的模型 '{self.model_name}' 不在可用模型列表中")
                    if available_models:
                        self.model_name = available_models[0]
                        print(f"自动切换到: {self.model_name}")
            else:
                print("无法获取模型列表，但服务器健康检查通过")
                
        except Exception as e:
            print(f"vLLM服务器连接失败: {e}")
            print("请确保vLLM服务器正在运行")
            raise
            
        end_time = time.time()
        print(f"连接检查完成，耗时 {end_time - start_time:.2f} 秒")
        print(f"最大并发数: {self.max_concurrent}")
        
    def generate_single(self, 
                       prompt: Dict[str, str], 
                       max_new_tokens: int = 4096, 
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       presence_penalty: float = 0.0,
                       repetition_penalty: float = 1.0) -> str:
        """
        生成单个回答（同步版本）。
        """
        chat_data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "stream": False
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(chat_data),
            timeout=self.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return ""
        else:
            raise Exception(f"API请求失败: {response.status_code}, {response.text}")
    
    def process_single_task(self, task_data) -> Tuple[int, int, int, str]:
        """
        处理单个推理任务。
        
        参数:
            task_data: (total_id, question_id, sample_id, prompt, max_new_tokens, temperature, top_p, presence_penalty, repetition_penalty)
            
        返回:
            (total_id, question_id, sample_id, response)
        """
        total_id, question_id, sample_id, prompt, max_new_tokens, temperature, top_p, presence_penalty, repetition_penalty = task_data
        
        try:
            response = self.generate_single(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty
            )
            return (total_id, question_id, sample_id, response)
        except Exception as e:
            print(f"  任务失败 - 问题{question_id}/样本{sample_id}: {e}")
            return (total_id, question_id, sample_id, "")
    
    def batch_inference_concurrent(self, 
                                 prompts: List[Dict[str, str]], 
                                 max_new_tokens: int = 4096,
                                 temperature: float = 0.7,
                                 top_p: float = 0.9,
                                 num_samples: int = 5,
                                 presence_penalty: float = 0.0,
                                 repetition_penalty: float = 1.0) -> List[Tuple[int, int, int, str]]:
        """
        并发批量推理。
        
        参数:
            prompts: prompt列表
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            num_samples: 每个问题生成的样本数量
            presence_penalty: 存在惩罚参数
            repetition_penalty: 重复惩罚参数
            
        返回:
            结果列表，每个元素为(total_id, question_id, sample_id, output)的元组
        """
        print(f"开始并发推理: {len(prompts)}个问题 × {num_samples}个样本 = {len(prompts) * num_samples}个任务")
        print(f"最大并发数: {self.max_concurrent}")
        
        # 构建所有任务
        tasks = []
        total_id = 0
        for i, prompt in enumerate(prompts):
            for j in range(num_samples):
                task_data = (total_id, i, j, prompt, max_new_tokens, temperature, top_p, presence_penalty, repetition_penalty)
                tasks.append(task_data)
                total_id += 1
        
        results = []
        completed_tasks = 0
        total_tasks = len(tasks)
        start_time = time.time()
        
        # 使用线程池进行并发处理
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.process_single_task, task): task 
                for task in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    completed_tasks += 1
                    
                    # 显示进度
                    if completed_tasks % 10 == 0 or completed_tasks == total_tasks:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / completed_tasks
                        remaining = (total_tasks - completed_tasks) * avg_time
                        print(f"  进度: {completed_tasks}/{total_tasks} "
                              f"({completed_tasks/total_tasks*100:.1f}%) "
                              f"- 已用时: {elapsed:.1f}s, 预计剩余: {remaining:.1f}s")
                        
                except Exception as e:
                    print(f"  任务异常: {e}")
                    # 添加空结果保持索引一致性
                    task_data = future_to_task[future]
                    results.append((task_data[0], task_data[1], task_data[2], ""))
        
        # 按照 total_id 排序（已经是按顺序的）
        results.sort(key=lambda x: x[0])
        
        total_time = time.time() - start_time
        print("并发推理完成!")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均每个任务: {total_time/total_tasks:.2f}秒")
        print(f"吞吐量: {total_tasks/total_time:.2f}任务/秒")
        
        return results
    
    def extract_answer(self, response: str) -> str:
        """
        从回答中提取最终答案。
        """
        # 尝试使用\\boxed{答案}格式提取答案（新格式）
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_match = re.search(boxed_pattern, response)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # 尝试使用$\\boxed{答案}$格式提取答案（旧格式）
        boxed_pattern_old = r'\$\\boxed\{(.*?)\}\$'
        boxed_match_old = re.search(boxed_pattern_old, response)
        if boxed_match_old:
            return boxed_match_old.group(1).strip()
        
        # 尝试使用####分隔符提取答案
        if "####" in response:
            answer = response.split("####")[-1].strip()
            return answer
        
        # 尝试使用最后一行作为答案
        lines = response.strip().split('\n')
        if lines:
            return lines[-1].strip()
        
        return response.strip()
    
    # 为了兼容性，保持原有的batch_inference方法
    def batch_inference(self, *args, **kwargs):
        """兼容原有接口的批量推理方法"""
        return self.batch_inference_concurrent(*args, **kwargs)
    
    def save_model(self, save_path: str) -> None:
        """
        保存模型配置（vLLM客户端不需要保存模型）。
        """
        print("vLLM客户端模式，无需保存模型")
        print(f"模型由vLLM服务器管理: {self.base_url}")


# 为了兼容性，创建一个别名
ModelInference = VLLMConcurrentInference


# 使用示例
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from data_loader import DataLoader
    
    # 测试并发推理
    print("=== 测试并发推理 ===")
    
    # 初始化并发推理类
    inference = VLLMConcurrentInference(max_concurrent=4)
    inference.load_model()
    
    # 加载测试数据
    # 从当前文件位置到数据文件的正确相对路径
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "input.csv")
    loader = DataLoader(data_path)
    questions = loader.get_questions()
    
    # 准备测试prompts（只用前3个问题）
    test_questions = questions[:5]
    prompts = []
    system_prompt = "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end of the response after a separator ####."
    
    for question in test_questions:
        prompts.append({
            "system": system_prompt,
            "user": question  # 使用实际的问题内容
        })
    
    # 并发推理测试
    start_time = time.time()
    results = inference.batch_inference_concurrent(
        prompts, 
        max_new_tokens=8192, 
        temperature=0.7, 
        num_samples=5
    )
    end_time = time.time()
    
    print("\n测试结果:")
    print(f"总耗时: {end_time - start_time:.2f}秒")
    print(f"生成结果数: {len(results)}")
    
    for i, (total_id, q_id, s_id, response) in enumerate(results):
        print(f"结果{i+1}: 问题{q_id}, 样本{s_id}, 长度{len(response)}")