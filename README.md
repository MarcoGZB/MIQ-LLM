# MIQ-LLM
MIQ-LLM work for Paper "A multi-task integrated question-answering framework for highly generalizable fault and RUL assessments based on LLMs with a post-training and fine-tuning strategy"
## 1.Paper Abstract
Effective prognostic and health management (PHM) methods including fault and remaining useful life (RUL) assessments are essential for the reliability and safety of rotating machinery in modern industrial systems. However, traditional approaches face generalizable challenges due to the diversification of industrial applications, meanwhile most existing methods address PHM tasks separately, limiting the demand-side flexibility. We propose a multi-task integrated question-answering (QA) large language model (MIQ-LLM) framework, which enables the high-adaptively generalizable fault and RUL assessments across various scenarios. Firstly, a template-driven signal-textual QA dataset construction pipeline and a multi-task signal prompt transformer (MSP-Former) for integrating signals and instructions. Further, a general post-training and domain-specific fine-tuning strategy is introduced for PHM capability transfer from complete datasets to diversified applications. Additionally, a Fast-adaptive Convergence Balancer (FaCB) is utilized for dynamically adjusting the joint post-training optimization. Experimental results on multiple datasets demonstrate that MIQ LLM surpasses traditional data driven methods and domain generalization baselines in both assessment accuracy and transfer efficiency. Ablation studies validate the generalization ability of MIQ-LLM and confirm the importance of MSP-Former, constructed QA training pairs in enhancing robustness, stability, and interpretability. MIQ-LLM offers a highly generalizable solution for rotating machinery, showcasing the potential of large language models in industrial PHM assessments.
### The post-training stage for fault and RUL assessment of the proposed MIQ-LLM framework
<img width="4200" height="3015" alt="Fig2" src="https://github.com/user-attachments/assets/e4c470ef-b98e-416b-997b-b613330f2cc7" />

### The fine-tuning stage in the proposed MIQ-LLM framework for highly generalizable fault and RUL assessment.
<img width="4200" height="1775" alt="Fig3" src="https://github.com/user-attachments/assets/b38a5f19-15ed-4414-be79-554a997427ad" />

### The architectures of MSP-Former, detailed functional modules and data flows
<img width="4200" height="2326" alt="Fig6" src="https://github.com/user-attachments/assets/e24d5586-c689-40ed-9096-7a80372f1c71" />

### The optimization flowchart of MIQ-LLM for joint post-training and generalizable fine-tuning stage
<img width="3378" height="2772" alt="Fig7" src="https://github.com/user-attachments/assets/7261a4ef-28c9-47c5-9e48-1455306cdb04" />

## 2.Code Version: 2026.07.21
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
## 3.Training (related files will be uploaded once the paper is accepted)

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
## 4.Links to public datasets
We have compiled 6 open source datasets utilized in this paper.
| Index 	| Year 	| Dataset Name 	| Component 	| Generation                   	| Working Condition           	| Original data link 	| Alternate data Link 	|
|-------	|------	|--------------	|-----------	|------------------------------	|-----------------------------	|--------------------	|---------------------	|
| 1     	| 2006 	| IMS          	| bearing   	| Run to failure               	| Single working condition    	|[[data link](https://www.nasa.gov/intelligent-systems-division)]                    	| [[data link](https://pan.quark.cn/s/003c8060617d)]                    	|
| 2     	| 2015 	| CWRU         	| bearing   	| artifical                    	| Multiple working conditions 	|[[data link](https://csegroups.case.edu/bearingdatacenter/pages/welcome-case-western-reserve-university-bearing-data-center-website)]                    	|        [[data link](https://pan.quark.cn/s/2b0ceb12ab5a)]                 	|
| 3     	| 2016 	| PU           	| bearing   	| artifical and run to failure 	| Multiple working conditions 	|[[data link](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/)]                    	|                   [[data link](https://pan.quark.cn/s/98940eefefb2)]    	|
| 4     	| 2018 	| XJTU-SY         	| bearing   	| Run to failure               	| Multiple working conditions 	|[[data link](http://biaowang.tech/xjtu-sy-bearing-datasets/)]                 	|        [[data link](https://pan.quark.cn/s/073484fd0bb0)]                     	|
| 5     	| 2012 	| PHM2012        	| bearing   	| Run to failure                    	| Multiple working conditions 	|   /                 	|        [[data link](https://pan.baidu.com/s/10bJXWq_IYZht0B_Pd-Betw?pwd=gr3k)]               	|
| 6     	| 2024 	| BJTU-RAO           	| bearing   	| artifical                    	| Multiple working conditions 	|[[data link](https://github.com/ChaoyingYang/SuperGraph)]              |        [[data link](https://pan.baidu.com/s/1S5F8URapHPArX2mV6FzDfA?pwd=9t74)]                       	|
## 5.Introduction for compared DA and DG methods 
| Category 	| Method 	| Descriptions 	| Paper Link 	|
|-------	|-------	|-------------	|--------------	|
|Domain-adaption |Empirical risk minimization (ERM)|ERM minimizes the classification/prediction errors for source samples.|[[paper link](https://www.sciencedirect.com/science/article/pii/S0951832023006324)]|
|Domain-adaption |Domain-adversarial neural networks (DANN)|DANN employs an adversarial network to match feature distributions. The generator tries to make the discriminator unable to determine which source domain the sample comes from, while the discriminator attempts to classify/predict the source domains of the samples.|[[paper link](https://www.sciencedirect.com/science/article/abs/pii/S0951832022000370)]|
|Domain-adaption |Deep correlation alignment (DCORAL)|DCORAL matches the mean and covariance of feature distributions of all source domains.|[[paper link](https://www.sciencedirect.com/science/article/abs/pii/S0951832026001869)]|
|Domain-generalization |Intrinsic and extrinsic domain generalization network (IEDGNet)|IEDGNet minimizes intrinsic multi-source data variation and mitigates the risk of overfitting.|[[paper link](https://ieeexplore.ieee.org/document/9452118)]|
|Domain-generalization |Domain generalization network combining invariance and specificity (DGNIS)|DGNIS minimizes triplet loss and CORAL loss across source samples and employing a weighted decision strategy.|[[paper link](https://www.sciencedirect.com/science/article/abs/pii/S0888327022001686)]|
## 6.The implementation of Open-source LLM-Backbone
The successful implementation of the LLM-Backbone in MIQ-LLM requires the weights files of each LLM-back from huggingface.io, you can access the utilized LLM-backbones directly by clicking below links. Subsequently, you have to download all the weights and parameter files in the "Files and Version" branch. Then, paste all the downloaded files to the corrseponding folder under the folder: ./LLMs/ . 
|LLM Backbone|Parameters|Descriptions |Source Link |
|-------	 |-------	|------------ |----------- |
|MMoE                            |8.93M  | Functioning as the non-LLM control group and developed by Google Research, the Multi-gate Mixture-of-Experts (MMoE) relies on task-specific gating mechanisms to coordinate distinct expert networks, facilitating basic feature sharing across parallel tasks.  | [[link](https://github.com/ZhichenZhao/pytorch-mmoe)]  |
|GPT-2-1.5B                      |1.5B   |  Acting as a representative early-stage autoregressive transformer and developed by OpenAI, this model leverages broad open-domain pre-training to provide foundational contextual encoding and language generation capabilities. | [[link](https://huggingface.co/openai-community/gpt2/tree/main)]   |
|DeepSeek-R1-Distill-Qwen-1.5B   |1.5B   | Derived from the Qwen architecture and refined via distillation techniques, these models are specifically optimized for logical reasoning and structured computation. They are engineered to balance robust analytical performance with deployability in resource-limited environments.  |  [[link](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/tree/main)] |
|DeepSeek-R1-Distill-Qwen-7B     |7B     | Derived from the Qwen architecture and refined via distillation techniques, these models are specifically optimized for logical reasoning and structured computation. They are engineered to balance robust analytical performance with deployability in resource-limited environments.  |  [[link](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/tree/main)] | |
|LlaMA-3.1-8B-Instruct           |8B     | A highly sophisticated 8-billion-parameter architecture developed by Meta. Benefiting from an expansive 128K context window and massive pre-training, it possesses exceptional capabilities in complex reasoning and numerical comprehension. | [[link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/main)]  |
|Qwen-2.5-7B-Instruct            |8B     |  An advanced 7-billion-parameter model tailored for strict adherence to user instructions. It excels at parsing structured data inputs and executing precise mathematical reasoning, making it highly adaptable to specialized analytical workflows.  | [[link](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)]  |
