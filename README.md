# MIQ-LLM
MIQ-LLM work for Paper "A multi-task integrated question-answering framework for highly generalizable fault and RUL assessments based on LLMs with a post-training and fine-tuning strategy"
## Paper Abstract
Effective prognostic and health management (PHM) methods including fault and remaining useful life (RUL) assessments are essential for the reliability and safety of rotating machinery in modern industrial systems. However, traditional approaches face generalizable challenges due to the diversification of industrial applications, meanwhile most existing methods address PHM tasks separately, limiting the demand-side flexibility. We propose a multi-task integrated question-answering (QA) large language model (MIQ-LLM) framework, which enables the high-adaptively generalizable fault and RUL assessments across various scenarios. Firstly, a template-driven signal-textual QA dataset construction pipeline and a multi-task signal prompt transformer (MSP-Former) for integrating signals and instructions. Further, a general post-training and domain-specific fine-tuning strategy is introduced for PHM capability transfer from complete datasets to diversified applications. Additionally, a Fast-adaptive Convergence Balancer (FaCB) is utilized for dynamically adjusting the joint post-training optimization. Experimental results on multiple datasets demonstrate that MIQ LLM surpasses traditional data driven methods and domain generalization baselines in both assessment accuracy and transfer efficiency. Ablation studies validate the generalization ability of MIQ-LLM and confirm the importance of MSP-Former, constructed QA training pairs in enhancing robustness, stability, and interpretability. MIQ-LLM offers a highly generalizable solution for rotating machinery, showcasing the potential of large language models in industrial PHM assessments.
### The post-training stage for fault and RUL assessment of the proposed MIQ-LLM framework
<img width="4200" height="3015" alt="Fig2" src="https://github.com/user-attachments/assets/e4c470ef-b98e-416b-997b-b613330f2cc7" />

### The fine-tuning stage in the proposed MIQ-LLM framework for highly generalizable fault and RUL assessment.
<img width="4200" height="1775" alt="Fig3" src="https://github.com/user-attachments/assets/b38a5f19-15ed-4414-be79-554a997427ad" />

### The architectures of MSP-Former, detailed functional modules and data flows
<img width="4200" height="2326" alt="Fig6" src="https://github.com/user-attachments/assets/e24d5586-c689-40ed-9096-7a80372f1c71" />

### The optimization flowchart of MIQ-LLM for joint post-training and generalizable fine-tuning stage
<img width="3378" height="2772" alt="Fig7" src="https://github.com/user-attachments/assets/7261a4ef-28c9-47c5-9e48-1455306cdb04" />

## Code Version: 2026.07.21
After downloading models and datasets, organize your files as follows:
<pre>
MIQ-LLM/
├── datasets/
|   ├── public/                       # Download and place public datasets 
│   └── private/                      # Smaples of the utilized private datasets
├── LLMs/                             # Base LLM-Instruct models
|   ├── DeepSeek-R1-Distill-Qwen-1.5B/
|   ├── DeepSeek-R1-Distill-Qwen-7B/
|   ├── Qwen2.5-1.5B-Instruct/
|   └── Qwen2.5-7B-Instruct/
├── DAmodels/                         # Domain-adaption models for compare                    
|   ├── data_loader_1d.py/
|   ├── resnet18_1d.py/
|   ├── utils.py/
|   ├── DANN.py/
|   ├── DCORAL.py/
│   └── ERM.py/                      
├── DGmodels/                         # Domain-adaption models for compare                    
|   ├── data_loader_1d.py/
|   ├── resnet18_1d.py/
|   ├── utils.py/
|   ├── DGNIS.py/
│   └── IEDGNet.py/     
├── models/                          # Utilized codes for MIQ-LLM construction
|   ├── data_processor.py/
|   ├── FaCB_Loss.py/
|   ├── inference.py/
|   ├── llm_matcher.py/
|   ├── MSP_Former.py/
│   ├── .../   
|   └── utils/
│         └── .../   
└── yaml/
    └── infer.yaml                   # Inference configuration
</pre>
### Run Inference

We now support **parallel inference** using `accelerate`. This automatically aggregates results from multiple GPUs.

```bash
# Using the automated script (Recommended)
bash scripts/inference.sh

# Or launch manually via accelerate
accelerate launch --config_file accelerate_config.yaml inference.py --config yaml/infer.yaml
```
## Training (related files will be uploaded once the paper is accepted)

We provide a training pipeline using `accelerate`. Ensure your `accelerate_config.yaml` is properly configured for your hardware.

### A. Post-training

Stage A focuses on Post-training stage of the `MSP-Former`.

```bash
# One-click Post-training
bash scripts/run_posttrain.sh
```

### B. Fine-Tuning 

Stage B performs end-to-end Fine-Tuning, Supervised Fine-Tuning for Small Samples and Unsupervised Fine-Tuning for Cross-working conditions.

```bash
# One-click SFT (Requires post-trained MSP-Former weights)
bash scripts/run_sft.sh
# One-click SFT (Requires post-trained MSP-Former weights)
bash scripts/run_uft.sh
```


