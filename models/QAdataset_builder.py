import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Any

# Import the processor and prompt constructor from your existing file
# Ensure data_processor.py is in the same directory
from data_processor import PHMDataProcessor, PromptConstructor

# Attempt to import PyEMD for Empirical Mode Decomposition (IMF1, IMF2 extraction)
try:
    from PyEMD import EMD
    EMD_AVAILABLE = True
except ImportError:
    print("[WARNING] PyEMD not installed. Will use a fallback filter for IMF approximations.")
    EMD_AVAILABLE = False


class MIQ_PHM_Dataset(Dataset):
    def __init__(self, base_dir: str, task_type: str, window_size: int = 1024, stride: int = 512):
        """
        PyTorch Dataset for MIQ-LLM framework.
        """
        self.base_dir = base_dir
        self.task_type = task_type
        self.window_size = window_size
        
        # Initialize core processors
        self.processor = PHMDataProcessor(window_size=window_size, stride=stride)
        
        # Determine prompt task type based on FA or RUL
        prompt_task = "diagnosis" if task_type == 'FA' else "rul"
        self.prompt_builder = PromptConstructor(task_type=prompt_task)
        
        if EMD_AVAILABLE:
            self.emd_decomposer = EMD()

        # Storage for processed dataset items
        self.data_samples = []  # 3-channel Tensors
        self.prompts = []       # Input Prompts
        self.labels = []        # Target Answers / Labels
        
        # Execute the pipeline
        self._build_dataset()

    def _build_dataset(self):
        """
        Traverses the private and public folders and executes Phase 1 to Phase 4.
        """
        private_path = os.path.join(self.base_dir, "private")
        public_path = os.path.join(self.base_dir, "public")
        
        # Define target folders based on task type
        if self.task_type == 'FA':
            target_folders = {
                "private": ["HST-BF"],
                "public": ["BJTU-RAO", "CWRU", "PU"]
            }
        else: # 'RUL'
            target_folders = {
                "private": ["HST-RUL"],
                "public": ["IMS", "PHM2012", "XJTU-SY"]
            }

        # Process Private Datasets
        for folder in target_folders["private"]:
            folder_path = os.path.join(private_path, folder)
            if os.path.exists(folder_path):
                self._process_folder(folder_path, folder)

        # Process Public Datasets
        for folder in target_folders["public"]:
            folder_path = os.path.join(public_path, folder)
            if os.path.exists(folder_path):
                self._process_folder(folder_path, folder)
                
        print(f"[SUCCESS] Built PyTorch {self.task_type} Dataset with {len(self.data_samples)} samples.")

    def _process_folder(self, folder_path: str, dataset_name: str):
        """
        Reads raw data, standardizes it, and constructs inputs/prompts.
        """
        # Dictionary simulating dataset physical properties for Prompt Detail Embedding
        condition_map = {
            "HST-BF": {"speed": "1500", "load": "10kN", "bearing": "SKF BT2-0231"},
            "HST-RUL": {"speed": "2500", "load": "20kN", "bearing": "NTN CRI-2692"},
            "CWRU": {"speed": "1797", "load": "1hp", "bearing": "SKF 6205"},
            "IMS": {"speed": "2000", "load": "6000lb", "bearing": "Rexnord ZA-2115"}
        }
        conditions = condition_map.get(dataset_name, {"speed": "Unknown", "load": "Unknown", "bearing": "Unknown"})

        # MOCKUP: In a real scenario, you would use pandas/scipy to load .csv or .mat files.
        # Here we simulate loading a raw signal and a reference normal signal for the folder.
        raw_signals = np.random.randn(10000)      # Simulating a loaded monitoring signal
        ref_signals = np.random.randn(10000) * 0.5 # Simulating a healthy reference signal
        true_fault = "Inner_Race_Fault"
        total_lifetime, fpt_point = 10000, 4000
        
        # ---------------------------------------------------------
        # Phase 1: Multi-source Signals Standardization & Segmentation
        # ---------------------------------------------------------
        # (Standardization and normalization logic handled internally by your rules)
        ref_rms_mean = np.mean(np.sqrt(np.mean(ref_signals**2)))
        ref_rms_std = np.std(np.sqrt(np.mean(ref_signals**2))) + 1e-8

        # Segment data using data_processor.py
        segments, ruls, faults = self.processor.extract_samples_and_labels(
            raw_signals=raw_signals,
            total_time=total_lifetime,
            fpt_time=fpt_point,
            fault_type=true_fault
        )
        
        for i in range(len(segments)):
            signal_patch = segments[i]
            
            # Normalize signal (Eq. 5 in paper)
            signal_patch = (signal_patch - np.min(signal_patch)) / (np.max(signal_patch) - np.min(signal_patch) + 1e-8)
            
            # Calculate State Information delta_m (Eq. 6 in paper)
            rms_current = np.sqrt(np.mean(signal_patch**2))
            delta_m = abs((rms_current - ref_rms_mean) / ref_rms_std)

            # ---------------------------------------------------------
            # Phase 2: Feature Extraction (3-Channel Construction)
            # ---------------------------------------------------------
            if self.task_type == 'FA':
                # FA Task: [Time-domain, IMF-1, IMF-2]
                if EMD_AVAILABLE:
                    imfs = self.emd_decomposer.emd(signal_patch, max_imf=2)
                    imf1 = imfs[0] if len(imfs) > 0 else np.zeros_like(signal_patch)
                    imf2 = imfs[1] if len(imfs) > 1 else np.zeros_like(signal_patch)
                else:
                    imf1 = np.convolve(signal_patch, np.ones(5)/5, mode='same')  # Fallback
                    imf2 = np.convolve(signal_patch, np.ones(10)/10, mode='same') # Fallback
                
                # Shape: (3, L)
                multi_channel_data = np.stack([signal_patch, imf1, imf2], axis=0)
                
            else:
                # RUL Task: Consecutive 3 time-windows [x_{t-1}, x_t, x_{t+1}]
                # For safety at boundaries, pad if previous/next are not available
                prev_patch = segments[i-1] if i > 0 else signal_patch
                next_patch = segments[i+1] if i < len(segments)-1 else signal_patch
                
                # Shape: (3, L)
                multi_channel_data = np.stack([prev_patch, signal_patch, next_patch], axis=0)
                
            # ---------------------------------------------------------
            # Phase 3: Answering Marks Processing
            # ---------------------------------------------------------
            if self.task_type == 'FA':
                if delta_m <= 3.0:
                    ans_state, ans_fault = "normal", "normal"
                elif 3.0 < delta_m <= 9.0:
                    ans_state, ans_fault = "be ware of fault", faults[i]
                else:
                    ans_state, ans_fault = "diagnosis needed", faults[i]
                    
                target_answer = f"<Fault Type: {ans_fault}> <State: {ans_state}>"
                
            else:
                # RUL Marks
                rul_val = ruls[i]
                if delta_m <= 3.0:
                    fail_prob = "0%"
                elif 3.0 < delta_m <= 9.0:
                    fail_prob = f"{np.clip((delta_m - 3)/6 * 100, 10, 90):.1f}%"
                else:
                    fail_prob = "100%"
                
                certainty = np.clip(1.0 - (np.std(signal_patch) / (np.mean(signal_patch)+1e-8)), 0, 1) * 100
                target_answer = f"<Failure Probability: {fail_prob}> <RUL: {rul_val:.3f}> <Certainty: {certainty:.1f}%>"

            # ---------------------------------------------------------
            # Phase 4: Template-driven Prompt Construction
            # ---------------------------------------------------------
            prompt_text = self.prompt_builder.build_prompt(
                condition_params=conditions,
                signal_placeholder="<|signal_patch|>"
            )
            
            # Store everything
            self.data_samples.append(torch.tensor(multi_channel_data, dtype=torch.float32))
            self.prompts.append(prompt_text)
            self.labels.append(target_answer)

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary containing the PyTorch tensors and strings required for the LLM.
        """
        return {
            "ts_values": self.data_samples[idx], # Shape: (3, L)
            "prompt": self.prompts[idx],         # String
            "answer": self.labels[idx]           # String
        }

# ==========================================
# Execution & Testing
# ==========================================
if __name__ == "__main__":
    # Ensure this path matches the folder structure shown in your image 
    # (e.g., .\LLMCode\MIQ-LLM\datasets)
    base_dataset_dir = os.path.join(".", "MIQ-LLM", "datasets")
    
    # 1. Instantiate the PyTorch Dataset for Fault Assessment (FA)
    print("--- Building Fault Assessment (FA) Dataset ---")
    fa_dataset = MIQ_PHM_Dataset(base_dir=base_dataset_dir, task_type='FA', window_size=1024)
    fa_loader = DataLoader(fa_dataset, batch_size=2, shuffle=True)
    
    # Check outputs
    if len(fa_dataset) > 0:
        sample_batch = next(iter(fa_loader))
        print(f"TS Tensor Shape (FA): {sample_batch['ts_values'].shape}") # Expected: [Batch, 3, 1024]
        print(f"Sample FA Prompt:\n{sample_batch['prompt'][0]}")
        print(f"Sample FA Answer:\n{sample_batch['answer'][0]}\n")
        
    # 2. Instantiate the PyTorch Dataset for RUL Prediction
    print("--- Building RUL Prediction Dataset ---")
    rul_dataset = MIQ_PHM_Dataset(base_dir=base_dataset_dir, task_type='RUL', window_size=1024)
    rul_loader = DataLoader(rul_dataset, batch_size=2, shuffle=True)
    
    # Check outputs
    if len(rul_dataset) > 0:
        sample_batch = next(iter(rul_loader))
        print(f"TS Tensor Shape (RUL): {sample_batch['ts_values'].shape}") # Expected: [Batch, 3, 1024]
        print(f"Sample RUL Prompt:\n{sample_batch['prompt'][0]}")
        print(f"Sample RUL Answer:\n{sample_batch['answer'][0]}")