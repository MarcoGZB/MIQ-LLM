import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMBackboneMatcher(nn.Module):
    """
    完成文本 Tokenization Input Embeddings。
    """
    def __init__(self, model_folder_name, base_dir=None, device='cuda'):
        """
        :param model_folder_name: "Qwen2.5-0.5B-Instruct"
        :param base_dir: LLMs path
        """
        super(LLMBackboneMatcher, self).__init__()
        self.device = device
        
        if base_dir is None:
            # 兼容 Windows/Linux 路径符号
            self.base_dir = os.path.join(".", "MIQ-LLM", "LLMs")
        else:
            self.base_dir = base_dir
            
        self.model_path = os.path.join(self.base_dir, model_folder_name)
        
        if not os.path.exists(self.model_path):
            print(f"Failed to find model at {self.model_path}, will attempt to load directly as a pre-trained path...")
            self.model_path = model_folder_name
        else:
            print(f"Successfully located large model backbone path: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        self.llm_model.eval()
        for param in self.llm_model.parameters():
            param.requires_grad = False
            
        # 5. 动态导出当前大模型底座的核心微观架构参数
        self.hidden_size = self.llm_model.config.hidden_size
        self.vocab_size = self.llm_model.config.vocab_size
        print(f"Loaded！[(hidden_size): {self.hidden_size} | vocab_size]: {self.vocab_size}]")

    def forward(self, prompt_text, max_length=512):
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        query_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 2. 抽取隐层词嵌入特征
        # 严密对应 TimeLanguageModel.py 中使用的提取技术
        with torch.no_grad():
            query_embeds = self.llm_model.get_input_embeddings()(query_ids)
            
        return query_ids, query_embeds, attention_mask

# ==========================================
# test
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    sample_prompt = "The rolling bearing is currently experiencing significant localized spalling under operating condition 3. Please diagnose the failure type and assess its remaining service life."
    
    try:

        encoder_mini = LLMBackboneMatcher(model_folder_name="Qwen2.5-0.5B-Instruct", device=device)
        q_ids, q_embeds, mask = encoder_mini(sample_prompt)
        print(f"Token IDs: {q_ids.shape}")
        print(f"Embeddings: {q_embeds.shape}") # 预期第3维对应其较小的隐藏维度 (如 896)
    except Exception as e:
        print(f"Failed: {e}\n")


    try:
        # 切换底座只需更改文件夹映射参数
        encoder_large = LLMBackboneMatcher(model_folder_name="Qwen2-7B-Instruct", device=device)
        q_ids_l, q_embeds_l, _ = encoder_large(sample_prompt)
        print(f"Output Embeddings Shape: {q_embeds_l.shape}") # 预期第3维对应标准的 4096 维
    except Exception as e:
        print(f"Failed to load model: {e}")