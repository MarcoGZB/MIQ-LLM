"""
MIQ-LLM Data Processing and Prompt Construction Module.
Includes: Sliding window sample division, piecewise linear RUL label extraction, multi-task Prompt dynamic generation.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

class PHMDataProcessor:
    def __init__(self, window_size: int = 1024, stride: int = 512):
        """
        Initialize the data processor.
        :param window_size: Sliding window size (e.g., intercepting 1024 sampling points each time)
        :param stride: Sliding stride
        """
        self.window_size = window_size
        self.stride = stride

    def generate_piecewise_rul(self, total_time: int, fpt_time: int, current_time: int) -> float:
        """
        Formula: 
        y_t = 1.0 (if t <= t_FPT)
        y_t = (T_total - t) / (T_total - t_FPT) (if t > t_FPT)
        
        :param total_time: Total bearing operating time/cycles (T_total)
        :param fpt_time: First prediction point (FPT), the moment early fault features appear (t_FPT)
        :param current_time: The time node corresponding to the current sliding window (t)
        :return: Normalized health indicator label (0.0 ~ 1.0)
        """
        if current_time <= fpt_time:
            return 1.0
        elif current_time >= total_time:
            return 0.0
        else:
            return (total_time - current_time) / (total_time - fpt_time)

    def extract_samples_and_labels(
        self, 
        raw_signals: np.ndarray, 
        total_time: int, 
        fpt_time: int, 
        fault_type: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        :param raw_signals: Raw 1D continuous vibration signal array
        :param total_time: The total sampling cycles of the bearing
        :param fpt_time: Set early degradation starting point
        :param fault_type: Fault category of the equipment (e.g., 'Inner Race Fault', 'Healthy')
        :return: 
            samples: Blocked temporal signal tensor (N, window_size)
            rul_labels: Corresponding RUL regression labels (N,)
            fault_labels: Corresponding fault classification label list (N,)
        """
        num_samples = (len(raw_signals) - self.window_size) // self.stride + 1
        
        samples = []
        rul_labels = []
        fault_labels = []

        for i in range(num_samples):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            # Extract signal Patch
            window_signal = raw_signals[start_idx:end_idx]
            samples.append(window_signal)
            
            # Calculate the representative time point of the current window (usually the end or midpoint of the window)
            # Assuming one sampling point represents one time unit, conversion must be based on the actual sampling rate
            current_t = end_idx 
            
            # Extract RUL label
            rul = self.generate_piecewise_rul(total_time, fpt_time, current_t)
            rul_labels.append(rul)
            
            # Extract classification labels
            # If in the fully healthy phase (RUL=1.0), it can be forcibly labeled as 'Healthy' to prevent early false alarms
            if rul == 1.0:
                fault_labels.append("Healthy")
            else:
                fault_labels.append(fault_type)

        return np.array(samples), np.array(rul_labels), fault_labels

class PromptConstructor:
    """
    Corresponds to the preparation phase of Feature-Token Fusion in the response letter.
    """
    def __init__(self, task_type: str = "joint"):
        """
        :param task_type: "diagnosis" (diagnosis only), "rul" (prediction only), "joint" (joint tasks)
        """
        self.task_type = task_type
        
    def build_prompt(self, condition_params: Dict[str, str], signal_placeholder: str = "<|signal_patch|>") -> str:
        """
        Dynamically construct system-level prompts (System Prompt) and task inputs.
        
        :param condition_params: Dictionary containing environmental/operating condition parameters (e.g., speed, load)
        :param signal_placeholder: Placeholder to be replaced by continuous signal Tokens during the feature fusion phase
        :return: Formatted complete Prompt string
        """
        # 1. Extract operating condition information (Cross-Condition feature description)
        speed = condition_params.get("speed", "Unknown")
        load = condition_params.get("load", "Unknown")
        
        context_str = (
            f"You are an expert AI assistant specialized in Prognostics and Health Management (PHM). "
            f"The following sequential data is acquired from a rolling bearing operating under a speed of {speed} RPM "
            f"and a load of {load} hp. "
        )
        
        # 2. Insert signal placeholder
        # In the FeatureTokenFusion phase, <|signal_patch|> will be located and replaced by visual/temporal embeddings
        signal_str = f"Here are the processed condition monitoring signals: {signal_placeholder}. "
        
        # 3. Append specific task instructions (Task Instructions)
        if self.task_type == "diagnosis":
            instruction = "Based on the provided signal features, please identify the current fault type of the bearing."
        elif self.task_type == "rul":
            instruction = "Based on the signal degradation pattern, evaluate the current health indicator (a continuous value from 1.0 to 0.0)."
        elif self.task_type == "joint":
            instruction = (
                "Based on the provided sequential signal features, perform two tasks simultaneously:\n"
                "1. Diagnose the specific fault category (e.g., Healthy, Inner Race Fault, Outer Race Fault).\n"
                "2. Evaluate the real-time health indicator (RUL), where 1.0 represents a perfectly healthy state and 0.0 indicates complete failure."
            )
        else:
            instruction = "Analyze the signal and provide an assessment."
            
        return context_str + signal_str + instruction

# ==========================================
# Usage Example
# ==========================================
if __name__ == "__main__":
    # Simulate a raw 1D vibration signal with a length of 10000
    dummy_raw_signal = np.random.randn(10000)
    
    # Define physical parameters
    total_lifetime = 10000       # Total lifetime (number of sampling points)
    fpt_point = 4000             # At the 4000th sampling point, early fault begins to occur
    true_fault = "Inner_Race"    # Actual fault type

    # 1. Data segmentation and label extraction
    processor = PHMDataProcessor(window_size=1024, stride=512)
    samples, rul_labels, cls_labels = processor.extract_samples_and_labels(
        raw_signals=dummy_raw_signal,
        total_time=total_lifetime,
        fpt_time=fpt_point,
        fault_type=true_fault
    )
    
    print(f"Extracted {len(samples)} samples (Batch Size)")
    print(f"First 5 RUL labels: {rul_labels[:5].round(4)}")
    print(f"Last 5 RUL labels: {rul_labels[-5:].round(4)}")
    print(f"First 5 classification labels: {cls_labels[:5]}")
    print("-" * 50)
    
    # 2. Construct text Prompt
    prompt_builder = PromptConstructor(task_type="joint")
    operating_conditions = {"speed": "1797", "load": "1.0"}
    
    # Build the multimodal input text corresponding to the current sample
    final_prompt = prompt_builder.build_prompt(condition_params=operating_conditions)
    
    print("Generated LLM Prompt text:\n")
    print(final_prompt)