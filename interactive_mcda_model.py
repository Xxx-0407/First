"""
交互式多准则决策分析模型
通过启发式策略选择问题，学习决策者的偏好函数和类阈值
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize, linprog
import random
from dataclasses import dataclass


@dataclass
class Question:
    """问题数据结构"""
    alternative_idx: int  # 方案索引
    alternative_values: np.ndarray  # 方案的准则值
    categories: List[str]  # 类别列表


@dataclass
class FuzzyAnswer:
    """模糊回答数据结构"""
    lower_category: str  # 下界类别（如'c1'）
    upper_category: str  # 上界类别（如'c2'）
    
    def __str__(self):
        return f"[{self.lower_category}, {self.upper_category}]"


class InteractiveMCDAModel:
    """交互式多准则决策分析模型"""
    
    def __init__(self, n_criteria: int, n_categories: int = 5, 
                 category_names: Optional[List[str]] = None,
                 n_breakpoints: int = 5):
        """
        初始化模型
        
        参数:
            n_criteria: 准则数量
            n_categories: 类别数量（默认5个）
            category_names: 类别名称列表（如['c1', 'c2', 'c3', 'c4', 'c5']）
            n_breakpoints: 每个边际效用函数的断点数量r（默认5个）
        """
        self.n_criteria = n_criteria
        self.n_categories = n_categories
        self.n_breakpoints = n_breakpoints  # 每个准则的断点数量r
        
        if category_names is None:
            self.category_names = [f'c{i+1}' for i in range(n_categories)]
        else:
            self.category_names = category_names
        
        # 待学习的参数（不确定性建模）
        self.lower_marginal_utility_functions = None  # 下界边际效用函数参数
        # 形状: (n_criteria, n_breakpoints) - 每个准则在每个断点的下界效用值
        self.upper_marginal_utility_functions = None  # 上界边际效用函数参数
        # 形状: (n_criteria, n_breakpoints) - 每个准则在每个断点的上界效用值
        self.category_thresholds = None  # 类阈值
        self.breakpoint_x_coords = None  # 断点的横坐标（每个准则的）
        # 形状: (n_criteria, n_breakpoints)
        
        # 人工决策者的真实偏好（用于生成模糊偏好信息）
        self.true_lower_marginal_utilities = None
        self.true_upper_marginal_utilities = None
        self.true_thresholds = None
        
        # 人工决策者的模糊偏好信息（用于自动回答）
        self.synthetic_fuzzy_answers: Optional[Dict[int, FuzzyAnswer]] = None
        
        # 交互历史（现在支持模糊回答）
        self.interaction_history: List[Tuple[Question, FuzzyAnswer]] = []
        # 跟踪每个方案被问的次数
        self.question_count: Dict[int, int] = {}  # {方案索引: 被问次数}
        
    def load_data_from_file(self, file_path: str) -> np.ndarray:
        """
        从文件加载数据
        
        参数:
            file_path: 数据文件路径（CSV格式，每行一个方案，每列一个准则）
        
        返回:
            m x n 的numpy数组，m个方案，n个准则
        """
        try:
            df = pd.read_csv(file_path, header=None)
            data = df.values
            if data.shape[1] != self.n_criteria:
                raise ValueError(f"数据文件中的准则数({data.shape[1]})与模型设定的准则数({self.n_criteria})不匹配")
            return data
        except Exception as e:
            raise ValueError(f"加载数据文件失败: {e}")
    
    def generate_random_data(self, n_alternatives: int, 
                            min_val: float = 0.0, max_val: float = 100.0) -> np.ndarray:
        """
        随机生成数据
        
        参数:
            n_alternatives: 备选方案数量
            min_val: 最小值
            max_val: 最大值
        
        返回:
            m x n 的numpy数组
        """
        return np.random.uniform(min_val, max_val, size=(n_alternatives, self.n_criteria))
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """数据归一化（0-1标准化）"""
        min_vals = data.min(axis=0)
        max_vals = data.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1  # 避免除零
        return (data - min_vals) / ranges
    
    def initialize_breakpoints(self, data: np.ndarray) -> np.ndarray:
        """
        初始化断点的横坐标
        
        参数:
            data: 原始数据（未归一化）
        
        返回:
            breakpoint_x_coords: (n_criteria, n_breakpoints) 数组
            每个准则的断点横坐标，第一个断点是最小值，最后一个断点是最大值
        """
        min_vals = data.min(axis=0)
        max_vals = data.max(axis=0)
        
        breakpoint_x_coords = np.zeros((self.n_criteria, self.n_breakpoints))
        
        for j in range(self.n_criteria):
            # 第一个断点：最小值
            breakpoint_x_coords[j, 0] = min_vals[j]
            # 最后一个断点：最大值
            breakpoint_x_coords[j, -1] = max_vals[j]
            # 中间断点：均匀分布
            if self.n_breakpoints > 2:
                breakpoint_x_coords[j, 1:-1] = np.linspace(
                    min_vals[j], max_vals[j], self.n_breakpoints
                )[1:-1]
        
        return breakpoint_x_coords
    
    def generate_synthetic_decision_maker(self, data: np.ndarray,
                                          breakpoint_x_coords: np.ndarray,
                                          n_modifications: int = 2) -> Dict:
        """
        生成人工决策者的模糊偏好信息
        
        基于偏好模型的约束条件和类阈值的基本约束条件，随机生成上界效用函数和下界效用函数的断点参数值，和类阈值。
        然后修改2-3条偏好信息，营造现实生活中决策者会提供不一致的偏好信息。
        
        参数:
            data: 所有方案的数据（原始数据）
            breakpoint_x_coords: 断点横坐标 (n_criteria, n_breakpoints)
            n_modifications: 要修改的偏好信息数量（默认2-3条）
        
        返回:
            包含真实偏好信息的字典
        """
        n_criteria = self.n_criteria
        n_breakpoints = self.n_breakpoints
        n_categories = self.n_categories
        
        # 1. 生成下界边际效用函数（单调递增，第一个断点为0）
        lower_marginal_utilities = np.zeros((n_criteria, n_breakpoints))
        for j in range(n_criteria):
            # 生成单调递增的随机值
            increments = np.random.uniform(0.01, 0.1, n_breakpoints - 1)
            increments = np.cumsum(increments)
            # 归一化到[0, 1/n_criteria]范围
            if increments[-1] > 0:
                increments = increments / increments[-1] * (1.0 / n_criteria)
            lower_marginal_utilities[j] = np.concatenate([[0], increments])
        
        # 2. 生成上界边际效用函数（单调递增，每个断点 >= 下界对应断点）
        upper_marginal_utilities = np.zeros((n_criteria, n_breakpoints))
        for j in range(n_criteria):
            # 上界从下界开始，添加随机增量
            gaps = np.random.uniform(0.01, 0.05, n_breakpoints)
            upper_marginal_utilities[j] = lower_marginal_utilities[j] + gaps
            # 确保单调递增
            for k in range(1, n_breakpoints):
                if upper_marginal_utilities[j, k] < upper_marginal_utilities[j, k-1]:
                    upper_marginal_utilities[j, k] = upper_marginal_utilities[j, k-1] + 0.01
        
        # 归一化：所有准则的最后一个断点下界效用值之和 = 1
        lower_sum = np.sum(lower_marginal_utilities[:, -1])
        if lower_sum > 0:
            lower_marginal_utilities = lower_marginal_utilities / lower_sum
        
        # 归一化：所有准则的最后一个断点上界效用值之和 = 1 + 某个值（确保上界 > 下界）
        upper_sum = np.sum(upper_marginal_utilities[:, -1])
        if upper_sum > 0:
            upper_marginal_utilities = upper_marginal_utilities / upper_sum * 1.2  # 上界比下界大20%
        
        # 3. 生成类别阈值（单调递增）
        thresholds = np.sort(np.random.uniform(0.1, 0.9, n_categories - 1))
        thresholds = np.clip(thresholds, 0, 1)
        
        # 保存真实偏好
        self.true_lower_marginal_utilities = lower_marginal_utilities.copy()
        self.true_upper_marginal_utilities = upper_marginal_utilities.copy()
        self.true_thresholds = thresholds.copy()
        
        # 4. 生成所有方案的模糊偏好信息
        n_alternatives = data.shape[0]
        fuzzy_answers = {}
        
        for i in range(n_alternatives):
            alt_values = data[i]
            lower_util, upper_util = self.compute_fuzzy_total_utility(
                alt_values, lower_marginal_utilities, upper_marginal_utilities, breakpoint_x_coords
            )
            
            # 根据上下界效用值确定类别范围
            category_to_num = {cat: idx for idx, cat in enumerate(self.category_names)}
            
            # 确定下界类别
            if lower_util < thresholds[0]:
                lower_cat_num = 0
            elif lower_util >= thresholds[-1]:
                lower_cat_num = n_categories - 1
            else:
                for k in range(len(thresholds) - 1):
                    if thresholds[k] <= lower_util < thresholds[k + 1]:
                        lower_cat_num = k + 1
                        break
                else:
                    lower_cat_num = n_categories - 1
            
            # 确定上界类别
            if upper_util < thresholds[0]:
                upper_cat_num = 0
            elif upper_util >= thresholds[-1]:
                upper_cat_num = n_categories - 1
            else:
                for k in range(len(thresholds) - 1):
                    if thresholds[k] <= upper_util < thresholds[k + 1]:
                        upper_cat_num = k + 1
                        break
                else:
                    upper_cat_num = n_categories - 1
            
            # 确保上界 >= 下界
            if upper_cat_num < lower_cat_num:
                upper_cat_num = lower_cat_num
            
            fuzzy_answers[i] = FuzzyAnswer(
                lower_category=self.category_names[lower_cat_num],
                upper_category=self.category_names[upper_cat_num]
            )
        
        # 5. 修改2-3条偏好信息以制造不一致
        n_modifications = min(n_modifications, n_alternatives)
        modified_indices = random.sample(range(n_alternatives), n_modifications)
        
        for idx in modified_indices:
            # 随机修改为不一致的类别范围
            lower_cat_num = random.randint(0, n_categories - 1)
            upper_cat_num = random.randint(lower_cat_num, n_categories - 1)
            
            fuzzy_answers[idx] = FuzzyAnswer(
                lower_category=self.category_names[lower_cat_num],
                upper_category=self.category_names[upper_cat_num]
            )
        
        print(f"\n已生成人工决策者的模糊偏好信息")
        print(f"修改了 {n_modifications} 条偏好信息以制造不一致")
        print(f"真实偏好参数已保存")
        
        return {
            'lower_marginal_utilities': lower_marginal_utilities,
            'upper_marginal_utilities': upper_marginal_utilities,
            'thresholds': thresholds,
            'fuzzy_answers': fuzzy_answers,
            'modified_indices': modified_indices
        }
    
    def compute_marginal_utility(self, criterion_value: float, 
                                 criterion_idx: int,
                                 utility_values: np.ndarray,
                                 breakpoint_x: np.ndarray) -> float:
        """
        计算给定准则值的边际效用（使用分段线性插值）
        
        参数:
            criterion_value: 准则值
            criterion_idx: 准则索引
            utility_values: 该准则在各断点的效用值 (n_breakpoints,)
            breakpoint_x: 该准则的断点横坐标 (n_breakpoints,)
        
        返回:
            边际效用值
        """
        # 找到criterion_value所在的区间
        if criterion_value <= breakpoint_x[0]:
            return utility_values[0]
        if criterion_value >= breakpoint_x[-1]:
            return utility_values[-1]
        
        # 线性插值
        for i in range(len(breakpoint_x) - 1):
            if breakpoint_x[i] <= criterion_value <= breakpoint_x[i + 1]:
                # 线性插值
                ratio = (criterion_value - breakpoint_x[i]) / (
                    breakpoint_x[i + 1] - breakpoint_x[i]
                )
                utility = utility_values[i] + ratio * (
                    utility_values[i + 1] - utility_values[i]
                )
                return utility
        
        return utility_values[-1]
    
    def compute_total_utility(self, alternative_values: np.ndarray,
                             marginal_utilities: np.ndarray,
                             breakpoint_x_coords: np.ndarray) -> float:
        """
        计算方案的总效用（所有准则的边际效用之和）
        
        参数:
            alternative_values: 方案的准则值 (n_criteria,)
            marginal_utilities: 边际效用函数参数 (n_criteria, n_breakpoints)
            breakpoint_x_coords: 断点横坐标 (n_criteria, n_breakpoints)
        
        返回:
            总效用值
        """
        total = 0.0
        for j in range(self.n_criteria):
            marginal = self.compute_marginal_utility(
                alternative_values[j], j,
                marginal_utilities[j], breakpoint_x_coords[j]
            )
            total += marginal
        return total
    
    def compute_fuzzy_total_utility(self, alternative_values: np.ndarray,
                                    lower_marginal_utilities: np.ndarray,
                                    upper_marginal_utilities: np.ndarray,
                                    breakpoint_x_coords: np.ndarray) -> Tuple[float, float]:
        """
        计算方案的模糊总效用（上下界）
        
        参数:
            alternative_values: 方案的准则值 (n_criteria,)
            lower_marginal_utilities: 下界边际效用函数参数 (n_criteria, n_breakpoints)
            upper_marginal_utilities: 上界边际效用函数参数 (n_criteria, n_breakpoints)
            breakpoint_x_coords: 断点横坐标 (n_criteria, n_breakpoints)
        
        返回:
            (下界总效用, 上界总效用)
        """
        lower_total = 0.0
        upper_total = 0.0
        for j in range(self.n_criteria):
            lower_marginal = self.compute_marginal_utility(
                alternative_values[j], j,
                lower_marginal_utilities[j], breakpoint_x_coords[j]
            )
            upper_marginal = self.compute_marginal_utility(
                alternative_values[j], j,
                upper_marginal_utilities[j], breakpoint_x_coords[j]
            )
            lower_total += lower_marginal
            upper_total += upper_marginal
        return lower_total, upper_total
    
    def calculate_heuristic_score(self, data: np.ndarray,
                                  candidate_idx: int,
                                  current_answers: Dict[int, FuzzyAnswer],
                                  lower_marginal_utilities: Optional[np.ndarray],
                                  upper_marginal_utilities: Optional[np.ndarray],
                                  thresholds: Optional[np.ndarray],
                                  breakpoint_x_coords: np.ndarray) -> float:
        """
        计算备选方案的启发式评分（上下界效用值与所属类的上下阈值的距离平方和）
        
        对所有备选方案进行评分，包括已经回答的和没有回答的。
        通过比较方案的上下界总效用与所属类的上下阈值，计算距离平方和作为评分。
        
        参数:
            data: 所有方案的数据（原始数据）
            candidate_idx: 候选方案的索引（可以是已回答的或未回答的）
            current_answers: 当前已有的模糊答案字典 {方案索引: FuzzyAnswer}
            lower_marginal_utilities: 下界边际效用函数参数 (n_criteria, n_breakpoints)
            upper_marginal_utilities: 上界边际效用函数参数 (n_criteria, n_breakpoints)
            thresholds: 当前的类别阈值 (n_categories-1,)
            breakpoint_x_coords: 断点横坐标 (n_criteria, n_breakpoints)
        
        返回:
            距离平方和（值越大，信息量越大）
        """
        
        # 如果还没有学习到参数，使用简单启发式
        if lower_marginal_utilities is None or upper_marginal_utilities is None or thresholds is None:
            # 第一轮：选择与已分类方案距离最远的方案（如果有的话）
            # 或者使用随机值（如果还没有已分类方案）
            if len(current_answers) == 0:
                return random.random()  # 第一轮，随机选择
            
            # 如果有已分类方案，选择距离最远的方案
            candidate_values = data[candidate_idx]
            min_distance = np.inf
            for alt_idx in current_answers.keys():
                alt_values = data[alt_idx]
                distance = np.linalg.norm(candidate_values - alt_values)
                min_distance = min(min_distance, distance)
            
            # 距离越远，信息量越大
            return min_distance if min_distance < np.inf else random.random()
        
        # 计算该方案的上下界总效用
        candidate_values = data[candidate_idx]
        lower_utility, upper_utility = self.compute_fuzzy_total_utility(
            candidate_values, lower_marginal_utilities, upper_marginal_utilities, breakpoint_x_coords
        )
        
        # 确定该方案所属的类别范围（通过比较上下界总效用和阈值）
        category_to_num = {cat: i for i, cat in enumerate(self.category_names)}
        
        # 根据下界总效用确定下界类别
        if lower_utility < thresholds[0]:
            predicted_lower_cat_num = 0
        elif lower_utility >= thresholds[-1]:
            predicted_lower_cat_num = self.n_categories - 1
        else:
            for i in range(len(thresholds) - 1):
                if thresholds[i] <= lower_utility < thresholds[i + 1]:
                    predicted_lower_cat_num = i + 1
                    break
            else:
                predicted_lower_cat_num = self.n_categories - 1
        
        # 根据上界总效用确定上界类别
        if upper_utility < thresholds[0]:
            predicted_upper_cat_num = 0
        elif upper_utility >= thresholds[-1]:
            predicted_upper_cat_num = self.n_categories - 1
        else:
            for i in range(len(thresholds) - 1):
                if thresholds[i] <= upper_utility < thresholds[i + 1]:
                    predicted_upper_cat_num = i + 1
                    break
            else:
                predicted_upper_cat_num = self.n_categories - 1
        
        # 计算上下界效用值与所属类的上下阈值的距离平方和
        distance_squared_sum = 0.0
        
        # 下界效用值与下界类的距离平方和
        if predicted_lower_cat_num == 0:
            distance_squared_sum += (lower_utility - 0.0)**2
            if len(thresholds) > 0:
                distance_squared_sum += (lower_utility - thresholds[0])**2
        elif predicted_lower_cat_num == self.n_categories - 1:
            distance_squared_sum += (lower_utility - thresholds[predicted_lower_cat_num - 1])**2
        else:
            distance_squared_sum += (lower_utility - thresholds[predicted_lower_cat_num - 1])**2 + \
                                   (lower_utility - thresholds[predicted_lower_cat_num])**2
        
        # 上界效用值与上界类的距离平方和
        if predicted_upper_cat_num == 0:
            distance_squared_sum += (upper_utility - 0.0)**2
            if len(thresholds) > 0:
                distance_squared_sum += (upper_utility - thresholds[0])**2
        elif predicted_upper_cat_num == self.n_categories - 1:
            distance_squared_sum += (upper_utility - thresholds[predicted_upper_cat_num - 1])**2
        else:
            distance_squared_sum += (upper_utility - thresholds[predicted_upper_cat_num - 1])**2 + \
                                   (upper_utility - thresholds[predicted_upper_cat_num])**2
        
        return distance_squared_sum
    
    def compute_convergence_score(self, data: np.ndarray,
                                  lower_marginal_utilities: np.ndarray,
                                  upper_marginal_utilities: np.ndarray,
                                  thresholds: np.ndarray,
                                  breakpoint_x_coords: np.ndarray) -> float:
        """
        计算收敛评分s值
        
        s = 所有备选方案（包括已回答的和未回答的）的上界效用值与规划模型得出的所属上界类的上下类阈值的距离平方和
          + 所有备选方案（包括已回答的和未回答的）的下界效用值与规划模型得出的所属下界类的上下类阈值的距离平方和
        
        参数:
            data: 所有方案的数据（原始数据）
            lower_marginal_utilities: 下界边际效用函数参数 (n_criteria, n_breakpoints)
            upper_marginal_utilities: 上界边际效用函数参数 (n_criteria, n_breakpoints)
            thresholds: 类别阈值 (n_categories-1,)
            breakpoint_x_coords: 断点横坐标 (n_criteria, n_breakpoints)
        
        返回:
            s值（收敛评分）
        """
        n_alternatives = data.shape[0]
        category_to_num = {cat: i for i, cat in enumerate(self.category_names)}
        total_score = 0.0
        
        for i in range(n_alternatives):
            alt_values = data[i]
            
            # 计算上下界总效用
            lower_utility, upper_utility = self.compute_fuzzy_total_utility(
                alt_values, lower_marginal_utilities, upper_marginal_utilities, breakpoint_x_coords
            )
            
            # 确定下界类别
            if lower_utility < thresholds[0]:
                predicted_lower_cat_num = 0
            elif lower_utility >= thresholds[-1]:
                predicted_lower_cat_num = self.n_categories - 1
            else:
                for k in range(len(thresholds) - 1):
                    if thresholds[k] <= lower_utility < thresholds[k + 1]:
                        predicted_lower_cat_num = k + 1
                        break
                else:
                    predicted_lower_cat_num = self.n_categories - 1
            
            # 确定上界类别
            if upper_utility < thresholds[0]:
                predicted_upper_cat_num = 0
            elif upper_utility >= thresholds[-1]:
                predicted_upper_cat_num = self.n_categories - 1
            else:
                for k in range(len(thresholds) - 1):
                    if thresholds[k] <= upper_utility < thresholds[k + 1]:
                        predicted_upper_cat_num = k + 1
                        break
                else:
                    predicted_upper_cat_num = self.n_categories - 1
            
            # 下界效用值与下界类的上下类阈值的距离平方和
            if predicted_lower_cat_num == 0:
                lower_score = (lower_utility - 0.0)**2
                if len(thresholds) > 0:
                    lower_score += (lower_utility - thresholds[0])**2
            elif predicted_lower_cat_num == self.n_categories - 1:
                lower_score = (lower_utility - thresholds[predicted_lower_cat_num - 1])**2
            else:
                lower_score = (lower_utility - thresholds[predicted_lower_cat_num - 1])**2 + \
                             (lower_utility - thresholds[predicted_lower_cat_num])**2
            
            # 上界效用值与上界类的上下类阈值的距离平方和
            if predicted_upper_cat_num == 0:
                upper_score = (upper_utility - 0.0)**2
                if len(thresholds) > 0:
                    upper_score += (upper_utility - thresholds[0])**2
            elif predicted_upper_cat_num == self.n_categories - 1:
                upper_score = (upper_utility - thresholds[predicted_upper_cat_num - 1])**2
            else:
                upper_score = (upper_utility - thresholds[predicted_upper_cat_num - 1])**2 + \
                             (upper_utility - thresholds[predicted_upper_cat_num])**2
            
            total_score += lower_score + upper_score
        
        return total_score
    
    def select_best_question(self, data: np.ndarray, 
                            current_answers: Dict[int, FuzzyAnswer],
                            lower_marginal_utilities: Optional[np.ndarray] = None,
                            upper_marginal_utilities: Optional[np.ndarray] = None,
                            thresholds: Optional[np.ndarray] = None,
                            breakpoint_x_coords: Optional[np.ndarray] = None) -> Optional[Question]:
        """
        使用启发式策略选择最有信息量的问题
        
        启发式策略：对所有备选方案（包括已经回答的和没有回答的）计算其上下界效用值与所属类的上下阈值的距离平方和，
        选取值最大的备选方案作为问题。如果方案已被问过2次，则不再对其进行评分和选择。
        
        参数:
            data: 所有方案的数据（原始数据）
            current_answers: 当前已有的模糊答案
            lower_marginal_utilities: 下界边际效用函数参数 (n_criteria, n_breakpoints)
            upper_marginal_utilities: 上界边际效用函数参数 (n_criteria, n_breakpoints)
            thresholds: 当前的类别阈值 (n_categories-1,)
            breakpoint_x_coords: 断点横坐标 (n_criteria, n_breakpoints)
        
        返回:
            选中的问题，如果所有方案都已被问过2次，返回None
        """
        n_alternatives = data.shape[0]
        best_idx = None
        best_score = -np.inf
        
        for i in range(n_alternatives):
            # 排除已经被问过2次的方案
            if self.question_count.get(i, 0) >= 2:
                continue
            
            score = self.calculate_heuristic_score(
                data, i, current_answers,
                lower_marginal_utilities, upper_marginal_utilities, thresholds, breakpoint_x_coords
            )
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx is None:
            # 所有方案都已被问过2次
            return None
        
        return Question(
            alternative_idx=best_idx,
            alternative_values=data[best_idx],
            categories=self.category_names
        )
    
    def ask_question(self, question: Question, 
                    current_answers: Dict[int, FuzzyAnswer],
                    question_number: int) -> FuzzyAnswer:
        """
        向决策者提问（交互接口），支持模糊回答和二次核对
        
        参数:
            question: 问题对象
            current_answers: 当前已有的模糊答案
            question_number: 问题编号
        
        返回:
            决策者的模糊回答（FuzzyAnswer）
        """
        alt_idx = question.alternative_idx
        is_second_time = alt_idx in current_answers
        previous_answer = current_answers.get(alt_idx, None)
        
        print(f"\n问题 {question_number}:")
        print(f"方案 {alt_idx + 1} 的准则值为: {question.alternative_values}")
        
        if is_second_time:
            print(f"【二次核对】该方案之前被分到: {previous_answer}")
            print(f"请选择该方案应该分到哪个类别范围（可以保持原答案或修改）:")
        else:
            print(f"请选择该方案应该分到哪个类别范围（模糊回答，如 [c1, c2]）:")
        
        for i, cat in enumerate(question.categories):
            marker = ""
            if is_second_time and previous_answer:
                if cat == previous_answer.lower_category:
                    marker = " ← 之前的下界"
                elif cat == previous_answer.upper_category:
                    marker = " ← 之前的上界"
            print(f"  {i+1}. {cat}{marker}")
        
        while True:
            try:
                print("\n请输入类别范围:")
                lower_choice = input("  下界类别编号 (1-{}): ".format(len(question.categories)))
                upper_choice = input("  上界类别编号 (1-{}): ".format(len(question.categories)))
                
                lower_idx = int(lower_choice) - 1
                upper_idx = int(upper_choice) - 1
                
                if 0 <= lower_idx < len(question.categories) and \
                   0 <= upper_idx < len(question.categories) and \
                   lower_idx <= upper_idx:
                    lower_category = question.categories[lower_idx]
                    upper_category = question.categories[upper_idx]
                    fuzzy_answer = FuzzyAnswer(lower_category=lower_category, 
                                             upper_category=upper_category)
                    
                    if is_second_time and previous_answer:
                        if str(fuzzy_answer) != str(previous_answer):
                            print(f"注意：答案已从 {previous_answer} 更改为 {fuzzy_answer}，将使用新的偏好信息。")
                    
                    return fuzzy_answer
                else:
                    print("无效的选择，请确保下界 <= 上界，请重新输入")
            except ValueError:
                print("请输入有效的数字")
    
    def compute_utility_coefficients(self, alt_values: np.ndarray,
                                     breakpoint_x_coords: np.ndarray) -> np.ndarray:
        """
        计算方案的总效用关于决策变量的系数向量
        
        参数:
            alt_values: 方案的准则值 (n_criteria,)
            breakpoint_x_coords: 断点横坐标 (n_criteria, n_breakpoints)
        
        返回:
            系数向量 (n_vars,)，其中n_vars = n_criteria * n_breakpoints + (n_categories - 1)
        """
        n_criteria = self.n_criteria
        n_breakpoints = self.n_breakpoints
        n_categories = self.n_categories
        n_vars = n_criteria * n_breakpoints + (n_categories - 1)
        
        coeffs = np.zeros(n_vars)
        
        for j in range(n_criteria):
            x_val = alt_values[j]
            bp_x = breakpoint_x_coords[j]
            
            if x_val <= bp_x[0]:
                # 使用第一个断点
                var_idx = j * n_breakpoints
                coeffs[var_idx] = 1.0
            elif x_val >= bp_x[-1]:
                # 使用最后一个断点
                var_idx = j * n_breakpoints + n_breakpoints - 1
                coeffs[var_idx] = 1.0
            else:
                # 线性插值
                for k in range(n_breakpoints - 1):
                    if bp_x[k] <= x_val <= bp_x[k + 1]:
                        ratio = (x_val - bp_x[k]) / (bp_x[k + 1] - bp_x[k])
                        var_idx_1 = j * n_breakpoints + k
                        var_idx_2 = j * n_breakpoints + k + 1
                        coeffs[var_idx_1] = 1.0 - ratio
                        coeffs[var_idx_2] = ratio
                        break
        
        return coeffs
    
    def build_optimization_model(self, data: np.ndarray, 
                                fuzzy_answers: Dict[int, FuzzyAnswer],
                                breakpoint_x_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据决策者的模糊回答构建规划模型，学习上下界偏好函数和类阈值
        
        目标函数：最小化所有已回答方案的上界效用值与回答的上界类的类阈值的距离平方和
                + 下界效用值与回答的下界类的类阈值的距离平方和
        
        参数:
            data: 所有方案的数据（原始数据，未归一化）
            fuzzy_answers: 决策者的模糊答案 {方案索引: FuzzyAnswer}
            breakpoint_x_coords: 断点横坐标 (n_criteria, n_breakpoints)
        
        返回:
            (lower_marginal_utilities, upper_marginal_utilities, thresholds)
            lower_marginal_utilities: (n_criteria, n_breakpoints) 下界边际效用函数参数
            upper_marginal_utilities: (n_criteria, n_breakpoints) 上界边际效用函数参数
            thresholds: (n_categories-1,) 类别阈值
        """
        n_criteria = self.n_criteria
        n_categories = self.n_categories
        n_breakpoints = self.n_breakpoints
        
        # 将类别映射到数值
        category_to_num = {cat: i for i, cat in enumerate(self.category_names)}
        
        # 变量定义：
        # [u_l_1_1, u_l_1_2, ..., u_l_1_r, ..., u_l_n_r,  # 下界边际效用函数
        #  u_u_1_1, u_u_1_2, ..., u_u_1_r, ..., u_u_n_r,  # 上界边际效用函数
        #  t_1, t_2, ..., t_{n_categories-1}]                # 类别阈值
        n_vars = 2 * n_criteria * n_breakpoints + (n_categories - 1)
        
        # 为每个已回答的方案计算上下界总效用的系数向量
        lower_utility_coeffs_list = []
        upper_utility_coeffs_list = []
        lower_category_nums = []
        upper_category_nums = []
        
        for alt_idx, fuzzy_answer in fuzzy_answers.items():
            alt_values = data[alt_idx]
            coeffs = self.compute_utility_coefficients(alt_values, breakpoint_x_coords)
            lower_utility_coeffs_list.append(coeffs)
            upper_utility_coeffs_list.append(coeffs)
            lower_category_nums.append(category_to_num[fuzzy_answer.lower_category])
            upper_category_nums.append(category_to_num[fuzzy_answer.upper_category])
        
        n_answered = len(fuzzy_answers)
        
        def objective_function(x):
            """
            目标函数：最小化所有已回答方案的上界效用值与回答的上界类的类阈值的距离平方和
                    + 下界效用值与回答的下界类的类阈值的距离平方和
            """
            total_loss = 0.0
            
            # 提取变量
            lower_start = 0
            upper_start = n_criteria * n_breakpoints
            threshold_start = 2 * n_criteria * n_breakpoints
            thresholds = x[threshold_start:]
            
            for i in range(n_answered):
                lower_coeffs = lower_utility_coeffs_list[i]
                upper_coeffs = upper_utility_coeffs_list[i]
                lower_cat_num = lower_category_nums[i]
                upper_cat_num = upper_category_nums[i]
                
                # 计算下界总效用
                lower_utility = np.dot(lower_coeffs[:upper_start], x[lower_start:upper_start])
                
                # 计算上界总效用
                upper_utility = np.dot(upper_coeffs[:upper_start], x[upper_start:threshold_start])
                
                # 下界效用值与下界类的距离平方和
                if lower_cat_num == 0:
                    lower_loss = (lower_utility - 0.0)**2
                    if n_categories > 1:
                        lower_loss += (lower_utility - thresholds[0])**2
                elif lower_cat_num == n_categories - 1:
                    lower_loss = (lower_utility - thresholds[lower_cat_num - 1])**2
                else:
                    lower_loss = (lower_utility - thresholds[lower_cat_num - 1])**2 + \
                                (lower_utility - thresholds[lower_cat_num])**2
                
                # 上界效用值与上界类的距离平方和
                if upper_cat_num == 0:
                    upper_loss = (upper_utility - 0.0)**2
                    if n_categories > 1:
                        upper_loss += (upper_utility - thresholds[0])**2
                elif upper_cat_num == n_categories - 1:
                    upper_loss = (upper_utility - thresholds[upper_cat_num - 1])**2
                else:
                    upper_loss = (upper_utility - thresholds[upper_cat_num - 1])**2 + \
                                 (upper_utility - thresholds[upper_cat_num])**2
                
                total_loss += lower_loss + upper_loss
            
            # 添加正则化项（防止过拟合）
            regularization = 0.001 * np.sum(x[:threshold_start]**2) + \
                           0.01 * np.sum(x[threshold_start:]**2)
            
            return total_loss + regularization
        
        # 约束条件
        constraints = []
        lower_start = 0
        upper_start = n_criteria * n_breakpoints
        threshold_start = 2 * n_criteria * n_breakpoints
        
        # 约束1：下界边际效用函数单调递增
        # u_l_j_k <= u_l_j_{k+1} for all j, k
        for j in range(n_criteria):
            for k in range(n_breakpoints - 1):
                def make_lower_monotonic_constraint(j_idx, k_idx):
                    def constraint(x, j_idx=j_idx, k_idx=k_idx):
                        var_idx_1 = j_idx * n_breakpoints + k_idx
                        var_idx_2 = j_idx * n_breakpoints + k_idx + 1
                        return x[var_idx_2] - x[var_idx_1]  # >= 0
                    return constraint
                constraints.append({
                    'type': 'ineq',
                    'fun': make_lower_monotonic_constraint(j, k)
                })
        
        # 约束2：上界边际效用函数单调递增
        # u_u_j_k <= u_u_j_{k+1} for all j, k
        for j in range(n_criteria):
            for k in range(n_breakpoints - 1):
                def make_upper_monotonic_constraint(j_idx, k_idx):
                    def constraint(x, j_idx=j_idx, k_idx=k_idx):
                        var_idx_1 = upper_start + j_idx * n_breakpoints + k_idx
                        var_idx_2 = upper_start + j_idx * n_breakpoints + k_idx + 1
                        return x[var_idx_2] - x[var_idx_1]  # >= 0
                    return constraint
                constraints.append({
                    'type': 'ineq',
                    'fun': make_upper_monotonic_constraint(j, k)
                })
        
        # 约束3：上界效用函数的每个断点 >= 下界效用函数的对应断点
        # u_u_j_k >= u_l_j_k for all j, k
        for j in range(n_criteria):
            for k in range(n_breakpoints):
                def make_bounds_constraint(j_idx, k_idx):
                    def constraint(x, j_idx=j_idx, k_idx=k_idx):
                        lower_idx = j_idx * n_breakpoints + k_idx
                        upper_idx = upper_start + j_idx * n_breakpoints + k_idx
                        return x[upper_idx] - x[lower_idx]  # >= 0
                    return constraint
                constraints.append({
                    'type': 'ineq',
                    'fun': make_bounds_constraint(j, k)
                })
        
        # 约束4：每个准则的第一个断点下界效用值为0（归一化）
        for j in range(n_criteria):
            def make_lower_zero_constraint(j_idx):
                def constraint(x, j_idx=j_idx):
                    return x[j_idx * n_breakpoints]  # = 0
                return constraint
            constraints.append({
                'type': 'eq',
                'fun': make_lower_zero_constraint(j)
            })
        
        # 约束5：每个准则的第一个断点上界效用值 >= 0
        for j in range(n_criteria):
            def make_upper_zero_constraint(j_idx):
                def constraint(x, j_idx=j_idx):
                    return x[upper_start + j_idx * n_breakpoints]  # >= 0
                return constraint
            constraints.append({
                'type': 'ineq',
                'fun': make_upper_zero_constraint(j)
            })
        
        # 约束6：所有准则的最后一个断点下界效用值之和为1（归一化）
        def lower_normalization_constraint(x):
            total = 0.0
            for j in range(n_criteria):
                total += x[j * n_breakpoints + n_breakpoints - 1]
            return total - 1.0  # = 0
        
        constraints.append({
            'type': 'eq',
            'fun': lower_normalization_constraint
        })
        
        # 约束7：所有准则的最后一个断点上界效用值之和 >= 1（确保上界 >= 下界）
        def upper_normalization_constraint(x):
            total = 0.0
            for j in range(n_criteria):
                total += x[upper_start + j * n_breakpoints + n_breakpoints - 1]
            return total - 1.0  # >= 0
        
        constraints.append({
            'type': 'ineq',
            'fun': upper_normalization_constraint
        })
        
        # 约束4：阈值递增 t_1 < t_2 < ... < t_{n_categories-1}
        for i in range(n_categories - 2):
            def make_threshold_constraint(i_idx):
                def constraint(x, i_idx=i_idx, threshold_start=n_criteria * n_breakpoints):
                    return x[threshold_start + i_idx + 1] - x[threshold_start + i_idx] - 0.01  # >= 0
                return constraint
            constraints.append({
                'type': 'ineq',
                'fun': make_threshold_constraint(i)
            })
        
        # 边界约束
        bounds = []
        # 下界边际效用函数值：非负（因为单调递增且第一个为0）
        for i in range(n_criteria * n_breakpoints):
            bounds.append((0, 1))
        # 上界边际效用函数值：非负
        for i in range(n_criteria * n_breakpoints):
            bounds.append((0, 2))  # 上界可以大于1
        # 阈值：在[0, 1]之间
        for i in range(n_categories - 1):
            bounds.append((0, 1))
        
        # 初始值
        x0 = np.zeros(n_vars)
        # 下界边际效用函数：线性递增
        for j in range(n_criteria):
            x0[j * n_breakpoints:(j + 1) * n_breakpoints] = np.linspace(
                0, 1.0 / n_criteria, n_breakpoints
            )
        # 上界边际效用函数：从下界开始，稍微大一点
        for j in range(n_criteria):
            x0[upper_start + j * n_breakpoints:upper_start + (j + 1) * n_breakpoints] = \
                np.linspace(0, 1.0 / n_criteria, n_breakpoints) * 1.1
        # 阈值：均匀分布
        x0[threshold_start:] = np.linspace(0, 1, n_categories)[1:]
        
        # 求解优化问题
        try:
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            if result.success:
                solution = result.x
                
                # 提取下界边际效用函数参数
                lower_marginal_utilities = solution[lower_start:upper_start].reshape(
                    n_criteria, n_breakpoints
                )
                
                # 提取上界边际效用函数参数
                upper_marginal_utilities = solution[upper_start:threshold_start].reshape(
                    n_criteria, n_breakpoints
                )
                
                # 提取阈值
                thresholds = solution[threshold_start:]
                
                return lower_marginal_utilities, upper_marginal_utilities, thresholds
            else:
                # 如果优化失败，使用默认值
                print(f"优化失败: {result.message}")
                # 默认：线性递增的边际效用函数
                lower_marginal_utilities = np.zeros((n_criteria, n_breakpoints))
                upper_marginal_utilities = np.zeros((n_criteria, n_breakpoints))
                for j in range(n_criteria):
                    lower_marginal_utilities[j] = np.linspace(0, 1.0/n_criteria, n_breakpoints)
                    upper_marginal_utilities[j] = np.linspace(0, 1.0/n_criteria, n_breakpoints) * 1.1
                thresholds = np.linspace(0, 1, n_categories)[1:]
                return lower_marginal_utilities, upper_marginal_utilities, thresholds
                
        except Exception as e:
            print(f"求解优化问题出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认值
            lower_marginal_utilities = np.zeros((n_criteria, n_breakpoints))
            upper_marginal_utilities = np.zeros((n_criteria, n_breakpoints))
            for j in range(n_criteria):
                lower_marginal_utilities[j] = np.linspace(0, 1.0/n_criteria, n_breakpoints)
                upper_marginal_utilities[j] = np.linspace(0, 1.0/n_criteria, n_breakpoints) * 1.1
            thresholds = np.linspace(0, 1, n_categories)[1:]
            return lower_marginal_utilities, upper_marginal_utilities, thresholds
    
    def predict_category(self, alternative_values: np.ndarray,
                        marginal_utilities: np.ndarray,
                        thresholds: np.ndarray,
                        breakpoint_x_coords: np.ndarray) -> str:
        """
        根据学习到的边际效用函数和阈值预测类别
        
        参数:
            alternative_values: 方案的准则值（原始数据）
            marginal_utilities: 边际效用函数参数 (n_criteria, n_breakpoints)
            thresholds: 阈值 (n_categories-1,)
            breakpoint_x_coords: 断点横坐标 (n_criteria, n_breakpoints)
        
        返回:
            预测的类别
        """
        # 计算总效用
        total_utility = self.compute_total_utility(
            alternative_values, marginal_utilities, breakpoint_x_coords
        )
        
        # 根据阈值判断类别
        if total_utility < thresholds[0]:
            return self.category_names[0]
        
        for i in range(len(thresholds) - 1):
            if thresholds[i] <= total_utility < thresholds[i + 1]:
                return self.category_names[i + 1]
        
        return self.category_names[-1]
    
    def run_interactive_learning(self, data: np.ndarray, 
                                max_questions: Optional[int] = None,
                                min_questions: int = 3,
                                use_synthetic_answers: bool = False,
                                convergence_l: int = 3,
                                convergence_threshold: float = 0.01) -> Dict:
        """
        运行交互式学习过程
        
        参数:
            data: 方案数据（原始数据，未归一化）
            max_questions: 最大问题数量（None表示问完所有方案）
            min_questions: 最少问题数量
            use_synthetic_answers: 是否使用人工决策者的偏好信息自动回答
            convergence_l: 连续l轮用于判断收敛（默认3）
            convergence_threshold: 收敛阈值t，连续l轮s值差值小于此值时停止（默认0.01）
        
        返回:
            学习结果字典
        """
        # 初始化断点横坐标
        self.breakpoint_x_coords = self.initialize_breakpoints(data)
        
        # 为了问题选择，使用归一化数据计算相似度
        normalized_data = self.normalize_data(data)
        
        current_answers = {}
        n_alternatives = data.shape[0]
        
        if max_questions is None:
            max_questions = n_alternatives
        
        print("=" * 60)
        print("开始交互式学习过程")
        print(f"方案数量: {n_alternatives}, 准则数量: {self.n_criteria}")
        print(f"类别: {', '.join(self.category_names)}")
        print(f"每个准则的断点数量: {self.n_breakpoints}")
        if use_synthetic_answers:
            print("模式: 自动使用人工决策者的偏好信息")
        else:
            print("模式: 手动输入偏好信息")
        print("=" * 60)
        
        # 交互循环
        question_count = 0
        current_lower_marginal_utilities = None
        current_upper_marginal_utilities = None
        current_thresholds = None
        
        # 初始化问次数跟踪
        self.question_count = {}
        
        # 收敛跟踪：存储连续l轮的s值
        convergence_scores = []
        
        while question_count < max_questions:
            # 选择问题（使用当前的上下界边际效用函数和阈值进行启发式评分）
            question = self.select_best_question(
                data, current_answers,
                current_lower_marginal_utilities, current_upper_marginal_utilities, 
                current_thresholds, self.breakpoint_x_coords
            )
            
            # 如果所有方案都已被问过2次，退出循环
            if question is None:
                print("\n所有备选方案都已被问过2次，交互结束。")
                break
            
            # 更新该方案被问的次数
            alt_idx = question.alternative_idx
            self.question_count[alt_idx] = self.question_count.get(alt_idx, 0) + 1
            
            # 记录之前的答案（如果有）
            previous_answer = current_answers.get(alt_idx, None)
            
            # 获取答案：如果使用人工决策者偏好信息且该方案在偏好信息中，直接使用
            if use_synthetic_answers and self.synthetic_fuzzy_answers is not None:
                if alt_idx in self.synthetic_fuzzy_answers:
                    # 直接使用人工决策者的偏好信息
                    answer = self.synthetic_fuzzy_answers[alt_idx]
                    print(f"\n问题 {question_count + 1}:")
                    print(f"方案 {alt_idx + 1} 的准则值为: {question.alternative_values}")
                    print(f"【自动回答】使用人工决策者的偏好信息: {answer}")
                else:
                    # 该方案不在人工决策者偏好信息中，询问用户
                    answer = self.ask_question(question, current_answers, question_count + 1)
            else:
                # 手动输入模式，询问用户
                answer = self.ask_question(question, current_answers, question_count + 1)
            
            # 记录答案（如果答案改变，覆盖之前的答案）
            if previous_answer is not None and str(answer) != str(previous_answer):
                print(f"答案已从 {previous_answer} 更改为 {answer}，将使用新的偏好信息。")
            
            current_answers[alt_idx] = answer
            self.interaction_history.append((question, answer))
            question_count += 1
            
            # 每一轮都会做规划，来求得这一轮的偏好函数和类阈值
            if len(current_answers) >= min_questions:
                # 学习参数（使用原始数据）
                lower_marginal_utilities, upper_marginal_utilities, thresholds = \
                    self.build_optimization_model(
                        data, current_answers, self.breakpoint_x_coords
                    )
                
                self.lower_marginal_utility_functions = lower_marginal_utilities
                self.upper_marginal_utility_functions = upper_marginal_utilities
                self.category_thresholds = thresholds
                
                # 更新当前轮次的参数（用于下一轮的问题选择）
                current_lower_marginal_utilities = lower_marginal_utilities
                current_upper_marginal_utilities = upper_marginal_utilities
                current_thresholds = thresholds
                
                # 计算当前轮的s值（收敛评分）
                s_value = self.compute_convergence_score(
                    data, lower_marginal_utilities, upper_marginal_utilities,
                    thresholds, self.breakpoint_x_coords
                )
                convergence_scores.append(s_value)
                
                # 保持最近l轮的s值
                if len(convergence_scores) > convergence_l:
                    convergence_scores.pop(0)
                
                # 显示当前学习结果
                print("\n当前学习结果:")
                print("下界边际效用函数参数（每个准则在各断点的效用值）:")
                for j in range(self.n_criteria):
                    print(f"  准则 {j+1}: {lower_marginal_utilities[j]}")
                print("上界边际效用函数参数（每个准则在各断点的效用值）:")
                for j in range(self.n_criteria):
                    print(f"  准则 {j+1}: {upper_marginal_utilities[j]}")
                print(f"类别阈值: {thresholds}")
                print(f"收敛评分 s = {s_value:.6f}")
                
                # 检查停止条件：连续l轮s值差值小于阈值t
                if len(convergence_scores) >= convergence_l:
                    # 计算连续l轮s值的最大差值
                    max_diff = max(convergence_scores) - min(convergence_scores)
                    print(f"连续{convergence_l}轮s值差值: {max_diff:.6f} (阈值: {convergence_threshold})")
                    
                    if max_diff < convergence_threshold:
                        print(f"\n收敛条件满足！连续{convergence_l}轮s值差值({max_diff:.6f})小于阈值({convergence_threshold})，交互结束。")
                        break
        
        # 最终学习
        if len(current_answers) >= min_questions:
            lower_marginal_utilities, upper_marginal_utilities, thresholds = \
                self.build_optimization_model(
                    data, current_answers, self.breakpoint_x_coords
                )
            self.lower_marginal_utility_functions = lower_marginal_utilities
            self.upper_marginal_utility_functions = upper_marginal_utilities
            self.category_thresholds = thresholds
        
        # 统计自动回答和手动输入的数量
        auto_answer_count = 0
        manual_answer_count = 0
        if use_synthetic_answers and self.synthetic_fuzzy_answers is not None:
            for alt_idx in current_answers.keys():
                if alt_idx in self.synthetic_fuzzy_answers:
                    auto_answer_count += 1
                else:
                    manual_answer_count += 1
        
        return {
            'lower_marginal_utility_functions': self.lower_marginal_utility_functions,
            'upper_marginal_utility_functions': self.upper_marginal_utility_functions,
            'thresholds': self.category_thresholds,
            'breakpoint_x_coords': self.breakpoint_x_coords,
            'answers': current_answers,
            'interaction_history': self.interaction_history,
            'auto_answer_count': auto_answer_count,
            'manual_answer_count': manual_answer_count,
            'use_synthetic_answers': use_synthetic_answers
        }


def main():
    """主函数：演示如何使用模型"""
    import sys
    
    print("交互式多准则决策分析模型")
    print("=" * 60)
    
    # 选择数据输入方式
    print("\n请选择数据输入方式:")
    print("1. 从文件加载数据")
    print("2. 随机生成数据")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    # 获取断点数量
    n_breakpoints_input = input("请输入每个边际效用函数的断点数量 r (默认5): ").strip()
    n_breakpoints = int(n_breakpoints_input) if n_breakpoints_input else 5
    
    if choice == "1":
        # 从文件加载
        file_path = input("请输入数据文件路径: ").strip()
        n_criteria = int(input("请输入准则数量 n: ").strip())
        
        model = InteractiveMCDAModel(n_criteria=n_criteria, n_breakpoints=n_breakpoints)
        data = model.load_data_from_file(file_path)
        print(f"成功加载数据: {data.shape[0]} 个方案, {data.shape[1]} 个准则")
        
    elif choice == "2":
        # 随机生成
        n_alternatives = int(input("请输入备选方案数量 m: ").strip())
        n_criteria = int(input("请输入准则数量 n: ").strip())
        
        model = InteractiveMCDAModel(n_criteria=n_criteria, n_breakpoints=n_breakpoints)
        data = model.generate_random_data(n_alternatives)
        print(f"成功生成数据: {data.shape[0]} 个方案, {data.shape[1]} 个准则")
        print("\n生成的数据:")
        print(data)
        
    else:
        print("无效的选择")
        return
    
    # 初始化断点
    model.breakpoint_x_coords = model.initialize_breakpoints(data)
    
    # 选择输入偏好信息的方式
    print("\n请选择输入偏好信息的方式:")
    print("1. 由现实生活中的操作者决策者自己输入")
    print("2. 利用提前生成的人工决策者的偏好信息来直接回答（自动模式）")
    
    input_mode = input("请输入选择 (1 或 2，默认1): ").strip() or "1"
    use_synthetic_answers = False
    
    if input_mode == "2":
        # 自动模式：生成人工决策者偏好信息
        use_synthetic_answers = True
        print("\n正在生成人工决策者的模糊偏好信息...")
        n_modifications = int(input("请输入要修改的偏好信息数量（2-3条，默认2）: ").strip() or "2")
        synthetic_data = model.generate_synthetic_decision_maker(
            data, model.breakpoint_x_coords, n_modifications
        )
        
        # 保存人工决策者的偏好信息
        model.synthetic_fuzzy_answers = synthetic_data['fuzzy_answers']
        
        print("\n生成的人工决策者模糊偏好信息:")
        for alt_idx, fuzzy_answer in synthetic_data['fuzzy_answers'].items():
            print(f"  方案 {alt_idx + 1}: {fuzzy_answer}")
        
        print(f"\n修改的偏好信息索引: {synthetic_data['modified_indices']}")
        print("\n注意：交互过程中将自动使用这些偏好信息，无需手动输入。")
    else:
        # 手动输入模式
        print("\n交互过程中将询问您每个方案的类别范围。")
    
    # 设置收敛参数
    print("\n设置收敛停止条件:")
    convergence_l_input = input("请输入连续轮数 l（用于判断收敛，默认3）: ").strip()
    convergence_l = int(convergence_l_input) if convergence_l_input else 3
    
    convergence_threshold_input = input("请输入收敛阈值 t（连续l轮s值差值小于此值时停止，默认0.01）: ").strip()
    convergence_threshold = float(convergence_threshold_input) if convergence_threshold_input else 0.01
    
    # 运行交互式学习
    max_q = input("\n请输入最大问题数量（直接回车表示问完所有方案）: ").strip()
    max_questions = int(max_q) if max_q else None
    
    results = model.run_interactive_learning(
        data, 
        max_questions=max_questions,
        use_synthetic_answers=use_synthetic_answers,
        convergence_l=convergence_l,
        convergence_threshold=convergence_threshold
    )
    
    # 显示最终结果
    print("\n" + "=" * 60)
    print("学习完成！最终结果:")
    print("=" * 60)
    print("\n下界边际效用函数参数（每个准则在各断点的效用值）:")
    lower_marginal_utils = results['lower_marginal_utility_functions']
    upper_marginal_utils = results['upper_marginal_utility_functions']
    breakpoint_x = results['breakpoint_x_coords']
    for j in range(model.n_criteria):
        print(f"\n准则 {j+1}:")
        print(f"  断点横坐标: {breakpoint_x[j]}")
        print(f"  下界效用值: {lower_marginal_utils[j]}")
        print(f"  上界效用值: {upper_marginal_utils[j]}")
    print(f"\n类别阈值: {results['thresholds']}")
    print(f"\n总共提问: {len(results['interaction_history'])} 个问题")
    print(f"已回答: {len(results['answers'])} 个方案")
    
    if results.get('use_synthetic_answers', False):
        print(f"自动回答: {results.get('auto_answer_count', 0)} 个方案")
        print(f"手动输入: {results.get('manual_answer_count', 0)} 个方案")


if __name__ == "__main__":
    main()

