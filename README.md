# Explainable artificial intelligence based intelligent fault diagnosis: A systematic review from applications to insights 基于可解释人工智能的智能故障诊断：从应用到洞察的系统回顾

# ![Awesome](https://img.shields.io/badge/Awesome-Yes-brightgreen) ![Last update](https://img.shields.io/badge/Last%20update-20250628-blue) ![Paper number](https://img.shields.io/badge/Paper%20Number-169-orange)

This is a repository about **Explainable intelligent fault diagnosis (XIFD) methods**, including papers, code, datasets etc. 

We will continue to update this repository and hope this repository can benefit your research.

![Image 1](https://github.com/HazeDT/Awesome-Explainable-IFD-Library/blob/main/XIFD.jpg)

# BibTex Citation

If you find this paper and repository useful, please cite our paper☺️.

```
@article{XIFD_LTF,
  title={Explainable artificial intelligence based intelligent fault diagnosis: A systematic review from applications to insights},
  author={Li, Tianfu, Chen Junfan, Liu Tao, Sun Chuang, Zhao Zhibin, Chen Xuefeng, Yan Ruqiang},
  journal={Reliability Engineering & System Safety},
  pages={109964},
  year={2026}
}
```
# Overview

- [1. XIFD](#section-id1)

-   - [1.1 Post-hoc explainability methods](#section-id2)
-   -   -  [1.1.1.Globally explainable methods](#section-id3)
-   -   -  [1.1.2.Locally explainable methods ](#section-id4)
-   - [1.2 Ante-hoc explainability methods](#section-id5)
-   -   - [1.2.1. Attention Mechanism Assisted Intelligent Fault Diagnosis Methods)](#section-id6)
-   -   - [1.2.2. Physics-informed intelligent fault diagnosis methods](#section-id7)
-   -   - [1.2.3. Signal Processing-informed Intelligent Fault Diagnosis Methods](#section-id8)
  
- [2. Data](#section-id9)

- [3.1. Code for Benchmark](#section-id10)

- [3.2. Code for Method Paper](#section-id11)

- [4. Papers for Prognosis](#section-id12)


# Papers
> We list papers, implementation code (the unofficial code is marked with *), etc, in the order of year.


## Post-hoc explainability methods
<a name="section-id2"></a>
> Post-hoc XIFD methods aim to explain how a trained model produce predictions for any decision-making process with a given input by developing additional explainers or techniques, which can be further categorized into local explainability and global explainability depending on the object and destination of the explanation..

| 0      | 1                                                                                        | 2        | 3        |
|:-------|:-----------------------------------------------------------------------------------------|:---------|:---------|
| Class  | Methods                                                                                  | Suit for | Suit for |
|        |                                                                                          | ML       | DL       |
| Global | Knowledge distillation [60]                                                              |  not        |   yes       |
| Global | Activation maximization: AM [61]                                                         |  not        |    yes      |
| Local  | Local approximation method: LIME [62], SP-LIME [63], S-LIME [64], ALIME [65], ILIME [66] |  yes        |   yes       |
| Local  | Gradient based method: Guided-BP [67], Smooth gradients [68], Integrated gradients [69]  |  not        |    yes      |
| Local  | Class activation mapping: CAM [70], Grad-CAM [71], Grad-CAM++ [72], LRP [61],            |   not       |    yes      |
| Local  | SHAP based method: SHAP [73]                                                             |    yes      |   yes       |


### Globally explainable methods
<a name="section-id3"></a>
Global explainability aims to help people understand the overall logic behind the model and its inner working mechanism
#### 2025


### Locally explainable methods
<a name="section-id4"></a>
#### 2025

#### 2024

## Ante-hoc explainability methods
<a name="section-id5"></a>
### Attention Mechanism Assisted Intelligent Fault Diagnosis Methods 
<a name="section-id6"></a>
Attention mechanisms provide a way to help understand what to focus on during model learning, and it allows the model to automatically and selectively focus on important information and ignore unimportant information when dealing with large amounts of input data in order to achieve transparency and visualization of the decision-making process, which in turn enhances the model explainability.
#### 1. Non-self-attention-based


#### 2. Self-attention-based



### Physics-informed intelligent fault diagnosis methods
<a name="section-id7"></a>
Currently, there are two ways to implement PIFD, where the first way is to establish a physical simulation model (PSM) and generate corresponding data to assist in model training, thereby guiding the model to effectively extract fault features. While the second way embeds physics equations as the loss function of the model to guide model training, such as the recently emerging physics-informed neural networks (PINN).
### PSMl-based

### PINN-based


#### 2025

- A Gradient Alignment Federated Domain Generalization Framework for Rotating Machinery Fault Diagnosis [[IOT 2025](https://ieeexplore.ieee.org/abstract/document/10949604)]


- Federated Domain Generalization for Fault Diagnosis: Cross-Client Style Integration and Dual Alignment Representation [[IOT 2025](https://ieeexplore.ieee.org/document/10926881)]
  


### Signal Processing-informed Intelligent Fault Diagnosis Methods [Mainstream in the IFD field]
<a name="section-id8"></a>
### 1)	Wavelet-based SPIFD methods 

### 2)	Sparse representation based-SPIFD methods

### 3)	Other SPIFD methods 
#### 2025

- Dual-contrastive Multi-view Graph Attention Network for Industrial Fault Diagnosis under Domain and Label Shift [[TIM 2025](https://ieeexplore.ieee.org/abstract/document/10891912)]


- Auxiliary-feature-embedded causality-inspired dynamic penalty networks for open-set domain generalization diagnosis scenario [[AEI 2025](https://www.sciencedirect.com/science/article/abs/pii/S1474034625001132)]


- Adaptive reconstruct feature difference network for open set domain generalization fault diagnosis [[EAAI 2025](https://www.sciencedirect.com/science/article/pii/S0952197624020542)]

- A self-improving fault diagnosis method for intershaft bearings with missing training samples [[MSSP 2025](https://www.sciencedirect.com/science/article/pii/S0888327024011592)]


### 2024

- A novel domain-private-suppress meta-recognition network based universal domain generalization for machinery fault diagnosis [[KBS 2024](https://www.sciencedirect.com/science/article/abs/pii/S0950705124014096)]


- Open-set domain generalization for fault diagnosis through data augmentation and a dual-level weighted mechanism [[AEI 2024](https://www.sciencedirect.com/science/article/abs/pii/S1474034624003513)]

- Curriculum learning-based domain generalization for cross-domain fault diagnosis with category shift [[MSSP 2024](https://www.sciencedirect.com/science/article/pii/S0888327024001936)]

### 2023

- A Novel Multidomain Contrastive-Coding-Based Open-Set Domain Generalization Framework for Machinery Fault Diagnosis [[TII 2023](https://ieeexplore.ieee.org/abstract/document/10382502?casa_token=FxKIZnqwoqgAAAAA:vvJI3TjUhHtASvVmDjK8jIGhvt0j7RO1wy0uL-kmiFSapnJOEkcm8YZJA3UpeZsnpUAeAhE)]

  
- A Customized Meta-Learning Framework for Diagnosing New Faults From Unseen Working Conditions With Few Labeled Data [[IEEE/ASME MEC 2023](https://ieeexplore.ieee.org/abstract/document/10214410?casa_token=GWKheX--CFQAAAAA:5n_rqYpoPNHdBYoSqSJJRrTiMf2jyMyO1syc5kEauCASvk9OaUXbNILADKzb-LeFuOTKidk)]

### 2022

- Adaptive open set domain generalization network: Learning to diagnose unknown faults under unknown working conditions [[RESS 2022](https://www.sciencedirect.com/science/article/pii/S0951832022003064)][[Code](https://github.com/CHAOZHAO-1/AOSDGN)]


## Imbalanced Domain Generalization-based Fault Diagnosis (IDGFD)
> Sample number for differnt classes in source domains are different.
<a name="section-id7"></a>



### 2025

- Imbalanced multidomain generalization fault diagnosis based on prototype-guided supervised contrastive learning with dynamic temperature modulation [[SHM 2025](https://journals.sagepub.com/doi/abs/10.1177/14759217251332517)]


- DRSC: Dual-Reweighted Siamese Contrastive Learning Network for Cross-Domain Rotating Machinery Fault Diagnosis With Multi-Source Domain Imbalanced Data [[IoT 2025](https://ieeexplore.ieee.org/abstract/document/10944708)]

- Imbalanced multi-domain generalization method for electro-mechanical actuator fault diagnosis under variable working conditions [[TIM 2025](https://ieeexplore.ieee.org/abstract/document/10938387)]


### 2024

- Adaptive Variational Sampling-embedded Domain Generalization Network for fault diagnosis with intra-inter-domain class imbalance[[RESS 2024](https://www.sciencedirect.com/science/article/abs/pii/S0951832024007786)]

- A two-stage learning framework for imbalanced semi-supervised domain generalization fault diagnosis under unknown operating conditions [[AEI 2024](https://www.sciencedirect.com/science/article/abs/pii/S1474034624005263)]



- Multi-domain Class-imbalance Generalization with Fault Relationship-induced Augmentation for Intelligent Fault Diagnosis [[TIM 2024](https://ieeexplore.ieee.org/document/10606303)]

- Long-tailed multi-domain generalization for fault diagnosis of rotating machinery under variable operating conditions [[SHM 2024](https://journals.sagepub.com/doi/10.1177/14759217241256690)]


### 2023

- Imbalanced Domain Generalization via Semantic-Discriminative Augmentation for Intelligent Fault Diagnosis [[AEI 2023]( https://www.sciencedirect.com/science/article/pii/S1474034623003907?via%3Dihub)][[Code](https://github.com/CHAOZHAO-1/SDAGN)]




## Single Domain Generalization-based Fault Diagnosis (SDGFD)
> source samples are only from a single domain.
<a name="section-id8"></a>


### 2025 



-Multi-style adversarial variational self-distillation in randomized domains for single-domain generalized fault diagnosis [[CII 2025](https://www.sciencedirect.com/science/article/pii/S0166361525000843)]


-A Generic Single-Source Domain Generalization Framework for Fault Diagnosis via Wavelet Packet Augmentation and Pseudo-Domain Generation [[IoT 2025](https://ieeexplore.ieee.org/abstract/document/11015920)]


- Addressing unknown faults diagnosis of transport ship propellers system based on adaptive evolutionary reconstruction metric network [[AEI 2025](https://www.sciencedirect.com/science/article/abs/pii/S1474034625001806)]



- Fault Diagnosis in Rolling Bearings Using Multi-Gaussian Attention and Covariance Loss for Single Domain Generalization [[TIM 2025](https://ieeexplore.ieee.org/abstract/document/10902562)]


- Dual adversarial and contrastive network for single-source domain generalization in fault diagnosis [[AEI 2025](https://www.sciencedirect.com/science/article/pii/S1474034625000333)]

- SDCGAN: A CycleGAN-Based Single-Domain Generalization Method for Mechanical Fault Diagnosis [[RESS 2025](https://www.sciencedirect.com/science/article/pii/S0951832025000572)]

### 2024 

- Uncertainty-guided adversarial augmented domain networks for single domain generalization fault diagnosis [[Measurement 2024](https://www.sciencedirect.com/science/article/abs/pii/S0263224124015598)]

- Prior knowledge embedding convolutional autoencoder: A single-source domain generalized fault diagnosis framework under small samples [[CII 2024](https://www.sciencedirect.com/science/article/abs/pii/S0166361524000976)][[Code](https://github.com/John-520/PKECA)]

- Simulation data-driven attention fusion network with multi-similarity metric: A single-domain generalization diagnostic method for tie rod bolt loosening of a rod-fastening rotor system [[MEASUREMENT 2024](https://www.sciencedirect.com/science/article/abs/pii/S0263224124014507)]

- Single imbalanced domain generalization network for intelligent fault diagnosis of compressors in HVAC systems under unseen working conditions [[Energy & Buildings  2024](https://www.sciencedirect.com/science/article/pii/S0378778824003086?via%3Dihub)]

- Single Source Cross-Domain Bearing Fault  Diagnosis via Multi-Pseudo Domain Augmented  Adversarial Domain-Invariant Learning [[JIOT 2024](https://ieeexplore.ieee.org/abstract/document/10577994)]

- Single domain generalization method based on anti-causal learning for rotating machinery fault diagnosis [[RESS 2024](https://www.sciencedirect.com/science/article/pii/S0951832024003247)]

- DP2Net: A discontinuous physical property-constrained single-source domain generalization network for tool wear state recognition [[MSSP 2024](https://www.sciencedirect.com/science/article/pii/S0888327024003194)]

- Gradient-based domain-augmented meta-learning single-domain generalization for fault diagnosis under variable operating conditions [[SHM 2024](https://journals.sagepub.com/doi/full/10.1177/14759217241230129)]

- HmmSeNet: A Novel Single Domain Generalization Equipment Fault Diagnosis Under Unknown Working Speed Using Histogram Matching Mixup[[TII 2024](https://ieeexplore.ieee.org/abstract/document/10417861/)]

- Support-Sample-Assisted Domain Generalization via Attacks and Defenses: Concepts, Algorithms, and Applications to Pipeline Fault Diagnosis [[TII 2024](https://ieeexplore.ieee.org/abstract/document/10384769?casa_token=dVxouWGvpSYAAAAA:PfiXfJAyfigutyUOLfRvn_OHFO_9YL8IOEl3Kd_rHodFFSEbfYJ4h9PGh5vYKBw0JkceMfw)]

### 2023

- Single domain generalizable and physically interpretable bearing fault diagnosis for unseen working conditions [[ESA 2023](https://www.sciencedirect.com/science/article/abs/pii/S0957417423029573)]

- Multi-scale style generative and adversarial contrastive networks for single domain generalization fault diagnosis [[RESS 2023](https://www.sciencedirect.com/science/article/pii/S0951832023007937?casa_token=jbSOPupOqNEAAAAA:h_9_4oxKe-zEoM0_zHNFt-b7abKR6OTdDRh-C9hEM0XWUZfj6h9DTJM_wJT-kOOITvEcRKwD)]

- An Adversarial Single-Domain Generalization Network for Fault Diagnosis of Wind Turbine Gearboxes [[J MAR SCI ENG 2023](https://www.mdpi.com/2077-1312/11/12/2384)]

### 2022

- Adversarial Mutual Information-Guided Single Domain Generalization Network for Intelligent Fault Diagnosis [[TII 2022](https://ieeexplore.ieee.org/document/9774938)]


# Data
> There are eight open-source dataset and two self-collected dataset for research of domain generalization-based fault diagnosis.
<a name="section-id9"></a>


| Index 	| Year 	| Dataset Name 	| Component 	| Generation                   	| Working Condition           	| Original data link 	| Alternate data Link 	|
|-------	|------	|--------------	|-----------	|------------------------------	|-----------------------------	|--------------------	|---------------------	|
| 1     	| 2006 	| IMS          	| bearing   	| Run to failure               	| Single working condition    	|[[data link](https://www.nasa.gov/intelligent-systems-division)]                    	| [[data link](https://pan.quark.cn/s/003c8060617d)]                    	|
| 2     	| 2013 	| JNU          	| bearing   	| artifical                    	| Multiple working conditions 	|   /                 	|              [[data link](https://pan.quark.cn/s/b2344c54c6d7)]            	|
| 3     	| 2015 	| CWRU         	| bearing   	| artifical                    	| Multiple working conditions 	|[[data link](https://csegroups.case.edu/bearingdatacenter/pages/welcome-case-western-reserve-university-bearing-data-center-website)]                    	|        [[data link](https://pan.quark.cn/s/2b0ceb12ab5a)]                 	|
| 4     	| 2016 	| PU           	| bearing   	| artifical and run to failure 	| Multiple working conditions 	|[[data link](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/)]                    	|                   [[data link](https://pan.quark.cn/s/98940eefefb2)]    	|
| 5     	| 2016 	| SCP          	| bearing   	| artifical                    	| Single working condition    	|      /              	|         [[data link](https://pan.quark.cn/s/6ccea2154a06)]                	| 
| 6     	| 2018 	| XJTU         	| bearing   	| Run to failure               	| Multiple working conditions 	|[[data link](http://biaowang.tech/xjtu-sy-bearing-datasets/)]                 	|        [[data link](https://pan.quark.cn/s/073484fd0bb0)]                     	|
| 7     	| 2018 	| PHM09        	| gearbox   	| artifical                    	| Multiple working conditions 	|   /                 	|        [[data link](https://pan.quark.cn/s/88180e4fccde)]               	|
| 8     	| 2021 	| LW           	| bearing   	| artifical                    	| Multiple working conditions 	|[[data link](https://github.com/ChaoyingYang/SuperGraph)]               	|        [[data link](https://pan.quark.cn/s/7e881548f5a1)]                       	|
| 9     	| 2022 	| HUSTbearing  	| bearing   	| artifical                    	| Multiple working conditions 	|     /               	|         [[data link](https://github.com/CHAOZHAO-1/HUSTbearing-dataset)]               	|       
| 10    	| 2022 	| HUSTgearbox  	| gearbox    	| artifical                    	| Multiple working conditions 	|    /                	|         [[data link](https://github.com/CHAOZHAO-1/HUSTgearbox-dataset)]              	|



# Code for Method Paper
<a name="section-id13"></a>
|  Title  |   Journal  |   Date   |   Code   |   topic   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|[**Conditional Contrastive Domain Generalization For Fault Diagnosis**](https://ieeexplore.ieee.org/abstract/document/9721021) <br> | TIM | 2022 | [Github](https://github.com/mohamedr002/CCDG) | DGFD |
|[**A domain generalization network combing invariance and specificity towards real-time intelligent fault diagnosis**](https://www.sciencedirect.com/science/article/pii/S0888327022001686) <br> | MSSP| 2022 | [Github](https://github.com/CHAOZHAO-1/DGNIS) | DGFD |
|[**Conditional-Adversarial-Domain-Generalization-with-Single-Discriminator**](https://ieeexplore.ieee.org/abstract/document/9399341/) <br> | TIM| 2022 | [Github](https://github.com/hectorLop/Conditional-Adversarial-Domain-Generalization-with-Single-Discriminator) | DGFD |
|[**A federated distillation domain generalization framework for machinery fault diagnosis with data privacy**](https://www.sciencedirect.com/science/article/pii/S0952197623019498) <br> | EAAI | 2024 | [Github](https://github.com/CHAOZHAO-1/FDDG) | FedDGFD |
|[**Federated domain generalization: A secure and robust framework for intelligent fault diagnosis**](https://ieeexplore.ieee.org/abstract/document/10196327) <br> | TII | 2023 | [Github](https://github.com/CHAOZHAO-1/FedDGMC) | FedDGFD |
|[**Imbalanced domain generalization via Semantic-Discriminative augmentation for intelligent fault diagnosis**](https://www.sciencedirect.com/science/article/pii/S1474034623003907) <br> | AEI | 2024 | [Github](https://github.com/CHAOZHAO-1/SDAGN) | IDGFD |
|[**Mutual-assistance semisupervised domain generalization network for intelligent fault diagnosis under unseen working conditions**](https://www.sciencedirect.com/science/article/pii/S0888327022011426) <br> | MSSP | 2023 | [Github](https://github.com/CHAOZHAO-1/MSDGN) | SemiDGFD |
|[**Adaptive open set domain generalization network: Learning to diagnose unknown faults under unknown working conditions**](https://www.sciencedirect.com/science/article/pii/S0951832022003064) <br> | RESS | 2022 | [Github](https://github.com/CHAOZHAO-1/AOSDGN) | OSDGFD |



# Talk
<a name="section-id14"></a>


- [苏州大学沈长青教授：从域适应到域泛化：人工智能驱动的故障诊断模型探索](https://www.bilibili.com/video/BV1V34y1q758/?spm_id_from=333.337.search-card.all.click&vd_source=ec846a76720b6da306d5919873954ab5)


# Contact

If you have any problem, please feel free to contact me.

Name: Tianfu Li

Email address: tianfu.li@kust.edu.cn

