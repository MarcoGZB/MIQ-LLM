import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List

# Import customized modules from the project
from QAdataset_builder import MIQ_PHM_Dataset
from llm_matcher import LLMBackboneMatcher
from MSP_Former import MSPFormer, FeatureTokenFusion

class MIQ_LLM_Pipeline(nn.Module):
    """
    The end-to-end inference pipeline bridging the time-series data and the LLM[cite: 5, 10].
    """
    def __init__(self, llm_folder_name: str, device: str = 'cuda'):
        super(MIQ_LLM_Pipeline, self).__init__()
        self.device = device
        
        # 1. Initialize the LLM Backbone Matcher[cite: 9]
        print("\n[INFO] Initializing LLM Backbone...")
        self.llm_matcher = LLMBackboneMatcher(model_folder_name=llm_folder_name, device=device)
        self.tokenizer = self.llm_matcher.tokenizer
        self.llm = self.llm_matcher.llm_model
        
        # 2. Initialize the Patch Embedding Layer
        # Projects raw 3-channel signals (e.g., L=1024) into a sequence of continuous tokens.
        # Output shape will be [Batch, 512, 64] -> transposed to [Batch, 64, 512]
        self.patch_len = 64
        self.patch_embedding = nn.Conv1d(
            in_channels=3, 
            out_channels=512, 
            kernel_size=16, 
            stride=16
        ).to(device)
        
        # 3. Initialize the MSP-Former[cite: 10]
        # llm_d_model automatically adapts to the hidden size of the loaded LLM backbone (e.g., 896 or 4096)[cite: 9, 10]
        print("[INFO] Initializing MSP-Former...")
        self.msp_former = MSPFormer(
            d_model=512, 
            e_layers=4, 
            n_heads=8, 
            llm_d_model=self.llm_matcher.hidden_size
        ).to(device)
        
        # Freeze all parameters for pure inference[cite: 5]
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _prepare_prompts(self, raw_prompts: List[str]) -> List[str]:
        """
        Replaces the string placeholder "<|signal_patch|>" with the actual number 
        of PAD tokens required for FeatureTokenFusion[cite: 7, 10].
        """
        pad_str = self.tokenizer.pad_token * self.patch_len
        prepared_prompts = [p.replace("<|signal_patch|>", pad_str) for p in raw_prompts]
        return prepared_prompts

    @torch.no_grad()
    def generate(self, ts_values: torch.Tensor, raw_prompts: List[str], max_new_tokens: int = 150) -> List[str]:
        """
        Executes the multimodal generation process.
        :param ts_values: Raw time-series tensors of shape [Batch, 3, 1024][cite: 6]
        :param raw_prompts: List of text prompts containing placeholders[cite: 6]
        :param max_new_tokens: Maximum tokens for the LLM to generate
        :return: List of generated text answers
        """
        ts_values = ts_values.to(self.device)
        
        # 1. Prepare and tokenize text prompts[cite: 9]
        prepared_prompts = self._prepare_prompts(raw_prompts)
        query_ids, query_embeds, attn_mask = self.llm_matcher(prepared_prompts)
        
        # 2. Patchify the raw time-series data
        # [Batch, 3, 1024] -> [Batch, 512, 64] -> [Batch, 64, 512]
        ts_tokens = self.patch_embedding(ts_values).transpose(1, 2)
        
        # 3. Process through MSP-Former to align with LLM latent space[cite: 10]
        # aligned_features shape: [Batch, 64, llm_d_model]
        aligned_features, _ = self.msp_former(ts_tokens, attn_mask=None)
        
        # 4. Multimodal Fusion via In-place Tensor Replacement[cite: 10]
        inputs_embeds = FeatureTokenFusion.fuse(
            ts_features=aligned_features,
            inputs_embeds=query_embeds,
            input_ids=query_ids,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # 5. Generate text output using the frozen LLM backbone[cite: 5, 9]
        generation_output = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # 6. Decode output IDs back to human-readable strings[cite: 9]
        generated_texts = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        return generated_texts


# ==========================================
# Execution & Testing
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dataset_dir = os.path.join(".", "MIQ-LLM", "datasets")
    
    # Select the model folder (ensure it matches the directory in .\MIQ-LLM\LLMs\)[cite: 9]
    target_llm = "Qwen2.5-0.5B-Instruct" 
    
    try:
        # Initialize the unified pipeline
        pipeline = MIQ_LLM_Pipeline(llm_folder_name=target_llm, device=device)
        
        # Initialize the Dataset and DataLoader (Using RUL task as an example)[cite: 6]
        print(f"\n[INFO] Loading RUL Dataset from {base_dataset_dir}...")
        dataset = MIQ_PHM_Dataset(base_dir=base_dataset_dir, task_type='RUL', window_size=1024)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        if len(dataset) > 0:
            print("\n[INFO] Starting Inference...\n" + "="*50)
            
            # Fetch a single batch for demonstration[cite: 6]
            batch = next(iter(dataloader))
            ts_tensors = batch['ts_values']  # [Batch, 3, 1024]
            prompts = batch['prompt']        # List of strings
            ground_truths = batch['answer']  # List of strings
            
            # Run inference
            predictions = pipeline.generate(ts_tensors, prompts, max_new_tokens=100)
            
            # Display Results
            for i in range(len(predictions)):
                print(f"--- Sample {i+1} ---")
                print(f"[GROUND TRUTH]:\n{ground_truths[i]}\n")
                print(f"[MODEL PREDICTION]:\n{predictions[i]}\n")
                print("-" * 50)
                
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")