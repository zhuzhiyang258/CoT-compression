"""
数据加载和预处理模块

此模块提供了加载和预处理金融领域长思维链压缩赛题数据的功能。
"""

import pandas as pd
import csv
from typing import List, Dict, Tuple, Optional

class DataLoader:
    """
    数据加载器类，用于加载和预处理金融领域长思维链压缩赛题数据。
    """
    
    def __init__(self, file_path: str, sep: str = '\t', header=None):
        """
        初始化数据加载器。
        
        参数:
            file_path: 数据文件路径
            sep: 分隔符，默认为制表符
            header: 表头行，默认为None（无表头）
        """
        self.file_path = file_path
        self.sep = sep
        self.header = header
        self.df = None
        self.questions = []
        
    def load_data(self) -> pd.DataFrame:
        """
        加载数据文件。
        
        返回:
            加载的DataFrame对象
        """
        try:
            self.df = pd.read_csv(self.file_path, sep=self.sep, header=self.header)
            print(f"成功加载数据，共{len(self.df)}条记录")
            return self.df
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise
    
    def get_questions(self) -> List[str]:
        """
        获取所有问题文本。
        
        返回:
            问题文本列表
        """
        if self.df is None:
            self.load_data()
        
        self.questions = []
        for i in range(len(self.df)):
            this_row = self.df.iloc[i]
            self.questions.append(this_row[2])  # 问题文本在第3列
        
        return self.questions
    
    def get_question_by_id(self, question_id: int) -> str:
        """
        根据问题ID获取问题文本。
        
        参数:
            question_id: 问题ID
            
        返回:
            问题文本
        """
        if self.df is None:
            self.load_data()
        
        return self.df.iloc[question_id][2]
    
    def prepare_prompt(self, question: str, system_prompt: Optional[str] = None) -> Dict[str, str]:
        """
        准备模型输入的prompt。
        
        参数:
            question: 问题文本
            system_prompt: 系统提示词，如果为None则使用默认提示词
            
        返回:
            包含system和user字段的字典
        """
        if system_prompt is None:
            system_prompt = "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end ofthe response after a separator ####."
        
        return {
            "system": system_prompt,
            "user": question
        }
    
    def prepare_batch_prompts(self, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        准备所有问题的prompt。
        
        参数:
            system_prompt: 系统提示词，如果为None则使用默认提示词
            
        返回:
            包含system和user字段的字典列表
        """
        if not self.questions:
            self.get_questions()
        
        prompts = []
        for question in self.questions:
            prompts.append(self.prepare_prompt(question, system_prompt))
        
        return prompts
    
    def save_results(self, results: List[Tuple[int, int, int, str]], output_path: str) -> None:
        """
        保存模型推理结果到CSV文件。
        
        参数:
            results: 结果列表，每个元素为(total_id, question_id, sample_id, output)的元组
            output_path: 输出文件路径
        """
        # 输出格式：(总编号, 问题编号, 样本编号, 模型原始输出)
        # 无需表头，使用"\t"作为分隔符
        # 将换行符和制表符转换为空格，确保文本在一个单元格中
        formatted_results = []
        for total_id, question_id, sample_id, output in results:
            # 清理输出文本，将换行符和制表符替换为空格
            cleaned_output = output.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
            # 去除多余空格
            cleaned_output = ' '.join(cleaned_output.split())
            formatted_results.append((total_id, question_id, sample_id, cleaned_output))
        
        output_df = pd.DataFrame(formatted_results, columns=['total_id', 'question_id', 'sample_id', 'output'])
        output_df.to_csv(output_path, sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar='\\')
        print(f"结果已保存到 {output_path}")


# 使用示例
if __name__ == "__main__":
    # 初始化数据加载器
    loader = DataLoader('./data/input.csv')
    
    # 加载数据
    df = loader.load_data()
    print(f"数据前5行:\n{df.head()}")
    
    # 获取所有问题
    questions = loader.get_questions()
    print(f"第一个问题:\n{questions[0][:200]}...")
    
    # 准备单个问题的prompt
    prompt = loader.prepare_prompt(questions[0])
    print(f"Prompt示例:\n{prompt}")
    
    # 准备自定义系统提示词的prompt
    custom_system_prompt = "You are a financial expert. Think step by step, but keep your reasoning concise. Return the answer at the end after ####."
    custom_prompt = loader.prepare_prompt(questions[0], custom_system_prompt)
    print(f"自定义Prompt示例:\n{custom_prompt}")

