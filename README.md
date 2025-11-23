# Explainable artificial intelligence based intelligent fault diagnosis: A systematic review from applications to insights 基于可解释人工智能的智能故障诊断：从应用到洞察的系统回顾


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
  volume = {267}
  pages={111935},
  year={2026}
}
```
# Task specification for IFD

IFD consists of four main tasks, that is, machine anomaly detection (AD), fault diagnosis (FD), remaining useful life (RUL) prediction, and cross-domain IFD, as shown below. 

![Image 2](https://github.com/HazeDT/Awesome-Explainable-IFD-Library/blob/main/Task%20of%20IFD.jpg)
# Overview

- [1. XIFD](#section-id1)

-   - [1.1 Post-hoc explainability methods](#section-id2)
-   -   -  [1.1.1.Globally explainable methods](#section-id3)
-   -   -  [1.1.2.Locally explainable methods ](#section-id4)
-   - [1.2 Ante-hoc explainability methods](#section-id5)
-   -   - [1.2.1. Attention Mechanism Assisted Intelligent Fault Diagnosis Methods)](#section-id6)
-   -   - [1.2.2. Physics-informed intelligent fault diagnosis methods](#section-id7)
-   -   - [1.2.3. Signal Processing-informed Intelligent Fault Diagnosis Methods](#section-id8)
  
- [2. Open source dataset](#section-id9)

- [3. Code for open Source XIFD methods](#section-id10)



# Papers
> We list papers, implementation code (the unofficial code is marked with *), etc, in the order of year.


## Post-hoc explainability methods
<a name="section-id2"></a>
> Post-hoc XIFD methods aim to explain how a trained model produce predictions for any decision-making process with a given input by developing additional explainers or techniques, which can be further categorized into local explainability and global explainability depending on the object and destination of the explanation.

| Class      | Methods                                                                         | Suit for       | Suit for        |
|:-------|:-----------------------------------------------------------------------------------------|:---------|:---------|
|        |                                                                                          | ML       | DL       |
| Global | Knowledge distillation [[IJCV 2021](https://link.springer.com/article/10.1007/s11263-021-01453-z)]                                                              |  not        |   yes       |
| Global | Activation maximization: AM [[ADLT 2024](https://www.thesciencebrigade.org/adlt/article/view/328)]                                                         |  not        |    yes      |
| Local  | Local approximation method: LIME [[ISMIR 2017](https://www.scopus.com/pages/publications/85063089874)], SP-LIME [[KDD 2016](https://dl.acm.org/doi/10.1145/2939672.2939778)], S-LIME [[KDD 2021](https://dl.acm.org/doi/10.1145/3447548.3467274)], ALIME [[IDEAL 2019](https://doi.org/10.1007/978-3-030-33607-3_49)], ILIME [[ADBIS 2019](https://link.springer.com/chapter/10.1007/978-3-030-28730-6_4)] |  yes        |   yes       |
| Local  | Gradient based method: Guided-BP [[SMARTTECH 2022](https://doi.org/10.1007/978-3-031-17337-6_14)], Smooth gradients [[arXiv 2017](https://arxiv.org/abs/1706.03825)], Integrated gradients [[PMLR 2017](https://proceedings.mlr.press/v70/sundararajan17a.html)]  |  not        |    yes      |
| Local  | Class activation mapping: CAM [[CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html)], Grad-CAM [[ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)], Grad-CAM++ [[WACV 2018](https://ieeexplore.ieee.org/document/8354201)], LRP [[PLoS One 2015](https://doi.org/10.1371/journal.pone.0130140)],            |   not       |    yes      |
| Local  | SHAP based method: SHAP [[NIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)]                                                             |    yes      |   yes       |


### 1) Globally explainable methods
<a name="section-id3"></a>
> Global explainability aims to help people understand the overall logic behind the model and its inner working mechanism.

| Method | Literatures | Usage and Disadvantages |
|--------|-------------|--------------------------|
| KD | Ji [[ASC 2022](https://doi.org/10.1016/j.asoc.2022.109331)], Zhong [[IEEE Sens. J. 2023](https://ieeexplore.ieee.org/abstract/document/10231110)], Sun [[TIM 2023](https://ieeexplore.ieee.org/abstract/document/10269080)], Li [[KBS 2022](https://doi.org/10.1016/j.knosys.2022.108345)] | Can explain the decision-making process of a complex model through a simple model, but ignores the knowledge representation within the complex model. |
| AM | Yang [[MST 2022](https://iopscience.iop.org/article/10.1088/1361-6501/ac41a5)], Jia [[MSSP 2018](https://www.sciencedirect.com/science/article/pii/S0888327018301444)] | Can visualize the input preferences of each neuron, but it does not directly explain why these features lead to the activation of neurons. |


### 2) Locally explainable methods
<a name="section-id4"></a>
> Local explainability aims to deeply analyze the decision-making process of the model for a specific input sample and its neighborhood.

#### a.LIME-based methods

| Method    | Literatures                                                                 | Usage and Disadvantages |
|-----------|-------------------------------------------------------------------------------|---------------------------|
| LIME      | Yao [[ASME 2021](https://asmedigitalcollection.asme.org/GT/proceedings-abstract/GT2021/V004T05A008/1119988)], Al-Zeyadi [[IJCNN 2020](https://ieeexplore.ieee.org/document/9206972)], Sanakkayala [86], Akin [[Micromachines 2022](https://www.mdpi.com/2072-666X/13/9/1471)], Khan [[UT 2024](https://essay.utwente.nl/fileshare/file/101027/Akin_BA_EEMCS.pdf)], Gawde [[DAJ 2024](https://www.sciencedirect.com/science/article/pii/S2772662224000298)], [[Access 2024](https://ieeexplore.ieee.org/document/10440027)], Li [[MST 2024](https://iopscience.iop.org/article/10.1088/1361-6501/ad3666)], Lu [[MST 2022](https://iopscience.iop.org/article/10.1088/1361-6501/ac78c5)], Mai [[DCASE 2022](https://qmro.qmul.ac.uk/jspui/handle/123456789/82013)] | Can explain tables, images, and text data, but can only provide explanations for predictions of a single sample, and the explanations are unstable. |
| SP-LIME   | ——                                                                            | Multiple samples can be explained, and the selected samples need to cover important features, but the algorithm accuracy is low. |
| S-LIME    | ——                                                                            | Can produce stable explanations, not suitable for time series data. |
| ILIME     | ——                                                                            | By selecting the most influential samples for prediction, the explanation accuracy is higher, but it is not applicable to text and image data. |
| GraphLIME | Li [[AEI 2024](https://www.sciencedirect.com/science/article/pii/S1474034624001083)]                                                                       | Can explain the importance of different node features for node classification tasks, but ignores the impact of edges on model performance. And it cannot be used to explain graph classification models. |

#### b.Gradient-based methods

| Method              | Literatures          | Usage and Disadvantages |
|---------------------|-----------------------|---------------------------|
| Guided-BP           | ——                    | The target features are relatively concentrated. |
| Integrated gradients | Li [[TNNLS 2021](https://ieeexplore.ieee.org/document/9411732)], Du [[Sensors 2022](https://www.mdpi.com/1424-8220/22/22/8760)]      | Explain that within CNN, there is less noise in the features. |
| Smooth gradients     | Peng [[ISA Transactions 2022](https://www.sciencedirect.com/science/article/pii/S0019057821003219)]             | Positioning image decision features, unable to quantify contribution. |

#### c.Class activation mapping-based methods

| Method           | Literatures                                                                 | Usage and Disadvantages                                                                 |
|------------------|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| CAM              | Sun [[Access 2020](https://ieeexplore.ieee.org/document/9142228)]                                                                   | Effectively reduces parameters and prevent overfitting, but the original model structure needs to be modified. |
| Grad-CAM         | Zhang [[Sensors 2024](https://www.mdpi.com/1424-8220/24/6/1831)], Chen [[Access 2020](https://ieeexplore.ieee.org/document/9131692)], Lu [[arXiv 2023](https://arxiv.org/abs/2308.10292)], Ren [[TIM 2023](https://ieeexplore.ieee.org/document/10138022)], Menno [[Annual Conference of the PHM Society 2021](https://papers.phmsociety.org/index.php/phmconf/article/view/3047)], Mathew [[Research Square 2024](https://assets-eu.researchsquare.com/files/rs-4707414/v1_covered_6b1f3f49-a2dd-4877-b36b-1a02d7e40c7e.pdf)], Yu [[Measurement 2022](https://www.sciencedirect.com/science/article/pii/S0263224122004778)], Guo [[TIM 2023](https://ieeexplore.ieee.org/abstract/document/10334497)], Senjoba [[Applied Sciences 2024](https://www.mdpi.com/2076-3417/14/9/3621)], Guo [[CMC 2023](https://www.techscience.com/cmc/v77n2/54820)] | Can be applied to different convolutional neural networks for explanation, but the gradient is unstable. |
| Grad-CAM++       | Chen [[IEEE Sens. J. 2023](https://ieeexplore.ieee.org/document/10144572)]                                                                   | Suitable for multi-target object detection explanations, but a lot of background information will be marked. |
| Score-CAM        | Chen [[Building and Environment 2023](https://www.sciencedirect.com/science/article/pii/S0360132323003554)]                                                                   | A gradient free method with good visualization effect                                    |
| Smoothed Score-CAM | Yang [[Neurocomputing 2023](https://www.sciencedirect.com/science/article/pii/S0925231223003806)]                                                                | Introduces an enhanced visual explanation algorithm to smooth the traditional Score-CAM |
| FreGrad-CAM      | Kim [[TII 2020](https://ieeexplore.ieee.org/abstract/document/9153108)]                                                                   | Designed to visualize the learned frequency features.                                   |
| MultiGrad-CAM    | Li [[JMS 2023](https://www.sciencedirect.com/science/article/pii/S0278612523001024)]                                                                    | Designed to address the issue of traditional Grad CAM feature resolution decreasing with increasing network layers |
| GCN—CAM          | Chen [[SAFEPROCESS 2021](https://ieeexplore.ieee.org/abstract/document/9693630)]                                                                  | Designed to visualize the learned features of GNNs.                                      |
| SGG-CAM          | Sun [[Measurement 2022](https://www.sciencedirect.com/science/article/pii/S0263224122000033)]                                                                   | Designed to solve the problem of insufficient centralized and accurate activation response of traditional CAM to fault areas |
| Grad-Absolute-CAM | Li [[Building and Environment 2021](https://www.sciencedirect.com/science/article/pii/S0360132321004595)]                                                                   | Designed to address the issue of traditional Grad-CAM being unable to focus on activating negative feature maps |

#### d.SHAP-based methods

| Method | Literatures                                                                 | Usage and Disadvantages                                                                 |
|--------|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| SHAP   | Groote [[MSSP 2022](https://www.sciencedirect.com/science/article/pii/S0888327021007998)], Ahmad [[Preprints 2021](https://www.preprints.org/frontend/manuscript/0355775a2605b11abe70a6d190e3991d/download_pub)], Li [[TAES 2023](https://ieeexplore.ieee.org/abstract/document/10214393)], Bindingsbø [[Frontiers in Energy Research 2023](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2023.1284676/full)], Moosavi [[Electronics 2024](https://www.mdpi.com/2079-9292/13/9/1721)], Brusa [[Applied Sciences 2023](https://www.mdpi.com/2076-3417/13/4/2038)], Pham [[ATiGB 2022](https://ieeexplore.ieee.org/abstract/document/9984085)], Hasan [Sensors 2021](https://www.mdpi.com/1424-8220/21/12/4070)], Yan [[EAAI 2024](https://www.sciencedirect.com/science/article/pii/S0952197624012041)], Jang [[TII 2023](https://ieeexplore.ieee.org/abstract/document/10032092)], Santos [[MLKE 2024](https://www.mdpi.com/2504-4990/6/1/16)] | Can explain the effect of features on the model’s predictions, but it cannot provide an explanation of the causal relationship between the features and the results. |

#### e.LRP-based XIFD methods

| Method | Literatures                                                                                                                                                                                                 | Usage and Disadvantages                                                                 |
|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| LRP    | Grezmak [[Procedia CIRP 2019](https://www.sciencedirect.com/science/article/pii/S2212827118312873)], [[IEEE Sens. J. 2019](https://ieeexplore.ieee.org/abstract/document/8930493)], Kim [[ESA 2024](https://www.sciencedirect.com/science/article/pii/S0957417423029573)], Wang [[RESS 2023](https://www.sciencedirect.com/science/article/pii/S0951832022006615)], Nie [[JIM 2021](https://link.springer.com/article/10.1007/s10845-020-01608-8)], Herwig [[TI 2023](https://www.sciencedirect.com/science/article/pii/S0301679X23004589)], Han [[JEET 2022](https://link.springer.com/article/10.1007/s42835-022-01207-y)], Xiong [[Building Simulation 2024](https://link.springer.com/article/10.1007/s12273-024-1154-1)], Qu [[SSRN 2023](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3979088)], Parziale [[SSRN 2023](https://journals.sagepub.com/doi/abs/10.1177/14759217241301288)] | Can provide explanations for model decisions, but has high computational costs for complex deep learning models |


## Ante-hoc explainability methods
<a name="section-id5"></a>
### Attention Mechanism Assisted Intelligent Fault Diagnosis Methods 
<a name="section-id6"></a>
Attention mechanisms provide a way to help understand what to focus on during model learning, and it allows the model to automatically and selectively focus on important information and ignore unimportant information when dealing with large amounts of input data in order to achieve transparency and visualization of the decision-making process, which in turn enhances the model explainability.
### 1) Non-self-attention-based
> Currently, many types of attention mechanisms have been developed, such as spatial-attention, channel-attention and mixed-attention.

### 2) Self-attention-based
> self-attention-based XIFD methods can effectively capture long-range dependencies in sequences and display the data of interest in the feature extraction process.


| Method          | Sub-method              | Literatures                                                                                                                                                                                                 | Usage and Disadvantages                                                                 |
|-----------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| Non-self-attention | Traditional attention   | Yang [[ASC 2020](https://www.sciencedirect.com/science/article/pii/S1568494620307675), Chen [[TIM 2021](https://ieeexplore.ieee.org/abstract/document/9611254)]                                                                                                                                                           | Doesn’t need to consider all relationships within the sequence and can be better adapted to a variety of different tasks, but is difficult to effectively capture relationships between the datapoints of the sequence. |
|                 | External attention      | Zhang [[BE 2022](https://www.sciencedirect.com/science/article/pii/S036013232200590X)]                                                                                                          | - |
|                 | Channel attention       | Ren [[TIM 2023](https://ieeexplore.ieee.org/document/10138022)], Wang [[Access 2023](https://ieeexplore.ieee.org/abstract/document/10265257)], Chen [[IEEE Sens. J. 2022](https://ieeexplore.ieee.org/abstract/document/9761239)], Chan [[GLOBECOM 2021](https://ieeexplore.ieee.org/abstract/document/9685864)] | - |
|                 | Path attention          | Zheng [[RESS 2024](https://www.sciencedirect.com/science/article/pii/S095183202300786X)]                                                                                                                            | - |
|                 | CBAM attention          | Li [[TNNLS 2023](https://ieeexplore.ieee.org/abstract/document/10257663)], Zhang [[Sensors 2024](https://www.mdpi.com/1424-8220/24/6/1831)]                                                             | - |
|                 | Mixed attention         | Liu [[Applied Sciences 2022](https://www.mdpi.com/2076-3417/12/16/8388)], Wang [[TII 2019](https://ieeexplore.ieee.org/abstract/document/8911240)], Su [[JMPSCE 2024](https://ieeexplore.ieee.org/abstract/document/10494233), [IET Renewable Power Generation 2022](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rpg2.12336)]], Khaniki [[arXiv 2024](https://arxiv.org/abs/2408.00033)], Xu [[RESS 2022](https://www.sciencedirect.com/science/article/pii/S0951832022003386)], Peng [[JDMD 2023](https://ojs.istp-press.com/dmd/article/view/156)], Zhao [[ESA 2024](https://www.sciencedirect.com/science/article/pii/S0957417424016336)] | - |
| Self-attention   | Self-attention          | Li [[TNNLS 2022](https://ieeexplore.ieee.org/abstract/document/9887963)], Che [[Shock and Vibration 2023](https://onlinelibrary.wiley.com/doi/full/10.1155/2023/1639287)], Wang [[TR 2023](https://ieeexplore.ieee.org/abstract/document/10315955)], Han [[KBS 2024](https://www.sciencedirect.com/science/article/pii/S095070512400995X)], Jiao [[Measurement 2023](https://www.sciencedirect.com/science/article/pii/S0263224122011460)] | Has a large receptive field, fewer parameters, and lower complexity, but requires a large amount of data, making it difficult to explain the specific role of each input element. |
|                 | Multi-head self-attention | Tang [[TIM 2022](https://ieeexplore.ieee.org/abstract/document/9761922)], Liu [[AEI 2022](https://www.sciencedirect.com/science/article/pii/S1474034622001835)], Keshun [[ Nonlinear Dynamics 2024](https://link.springer.com/article/10.1007/s11071-024-10157-1)]                                                         | - |
|                 | Scaling dot-product attention | Ning [[Electronics 2024](https://www.mdpi.com/2079-9292/13/13/2662)]                                                                                                                                        | - |

### Physics-informed intelligent fault diagnosis methods
<a name="section-id7"></a>
Currently, there are two ways to implement PIFD, where the first way is to establish a physical simulation model (PSM) and generate corresponding data to assist in model training, thereby guiding the model to effectively extract fault features. While the second way embeds physics equations as the loss function of the model to guide model training, such as the recently emerging physics-informed neural networks (PINN).
### 1) PSM-based
#### 2025
### 2) PINN-based
#### 2025

- A Gradient Alignment Federated Domain Generalization Framework for Rotating Machinery Fault Diagnosis [[IOT 2025](https://ieeexplore.ieee.org/abstract/document/10949604)]


- Federated Domain Generalization for Fault Diagnosis: Cross-Client Style Integration and Dual Alignment Representation [[IOT 2025](https://ieeexplore.ieee.org/document/10926881)]
  


### Signal Processing-informed Intelligent Fault Diagnosis Methods [Mainstream in the IFD field]
<a name="section-id8"></a>
### 1)	Wavelet-based SPIFD methods 
#### 2025
### 2)	Sparse representation based-SPIFD methods
#### 2025
### 3)	Other SPIFD methods 
#### 2025

- Dual-contrastive Multi-view Graph Attention Network for Industrial Fault Diagnosis under Domain and Label Shift [[TIM 2025](https://ieeexplore.ieee.org/abstract/document/10891912)


# Open source dataset
> There are eight open-source dataset and two self-collected dataset for research of domain generalization-based fault diagnosis.
<a name="section-id9"></a>


| Task                 | Dataset                  | Object                                        | Description                                                                       |
|:------------------|:------------------|:-----------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Anomaly detection | MIMII [30]        | Valve, pump, fan, slide rail             | This dataset is a sound dataset that simulates the sound of components such as valves, pumps, fans, and slides under normal and abnormal conditions. The data was recorded by an 8-channel microphone array and simulated the impact of noise in a real factory.                                                                                                                                                  |
| Anomaly detection | MIMII DG [31]     | Fan, gearbox, bearing, slide rail, valve | The dataset contains sound recordings from five types of industrial machines (fan, gearbox, bearing, slide rail, valve), collected under multiple domain-shift scenarios. Each audio clip is about 10 seconds long, sampled at 16 kHz using a TAMAGO-03 microphone in soundproof or anechoic chambers.                                                                                                            |
| Anomaly detection | ToyADMOS [32]     | Micromachines                            | The dataset is the first large-scale dataset for anomalous sound detection in machine operations, featuring around 540 hours of normal sounds and over 12,000 anomalous samples recorded with four microphones at a 48 kHz sampling rate.                                                                                                                                                                         |
| Anomaly detection | IMAD-DS [33]      | Motor & robotic arm                      | This multi-sensor industrial dataset contains normal and faulty operation data from robotic arms and brushless motors. It includes signals from microphones and accelerometers and introduces domain shifts such as variations in load, speed, and background noise.                                                                                                                                              |
| Anomaly detection | RflyMAD [34]      | Drones                                   | This dataset is used for multi-rotor drone fault detection and health management. It contains data on 11 common faults (such as motor failure, propeller failure, etc.) in six flight states, covering both simulated and real flight scenarios.                                                                                                                                                                  |
| Anomaly detection | PyScrew [35]      | Screw                                    | The dataset collects data from six screw tightening scenarios, including more than 34,000 industrial screw tightening operations, covering various health conditions such as thread wear, surface friction, and assembly failures.                                                                                                                                                                                |
| Diagnosis         | CWRU [36]         | Bearing                                  | The dataset consists of four sub-datasets, each with operating conditions of 0 hp - 1797 rpm, 1 hp - 1772 rpm, 2 hp - 1750 rpm, and 3 hp - 1730 rpm. The motor bearing faults include ball fault, inner ring fault, and outer ring fault.                                                                                                                                                                         |
| Diagnosis         | MFPT [37]         | Bearing                                  | This dataset consists of four sets of bearing vibration data. In the first sub-dataset, it contains three baseline conditions. In the second sub-dataset, it contains three outer race fault conditions. In the third sub-dataset, it contains seven outer race fault conditions with seven different loads. In the fourth sub-dataset, it contains seven inner race fault conditions with seven different loads. |
| Diagnosis         | PU [38]           | Bearing                                  | The dataset contains 32 sub-files, including 26 faulty bearings and 6 healthy bearings. The faulty bearings include 12 artificial damages caused by EDM and 14 real damages caused by accelerated life tests.                                                                                                                                                                                                     |
| Diagnosis         | JNU [39]          | Bearing                                  | The dataset contains four types of bearing faults, including normal state, ball fault, inner race fault, and outer race fault. Vibration data were collected under three different working conditions, with the motor speeds set to 600, 800, and 1000 rpm, respectively.                                                                                                                                         |
| Diagnosis         | HIT [40]          | Bearing                                  | This dataset is for aero-engine inter-shaft bearing failure. The test bench consists of a modified aero-engine, a motor drive system and a lubricating oil system. The experiment collected data of one outer ring failure and two bearing inner ring failures at high- and low-pressure rotors at 28 different speeds.                                                                                           |
| Diagnosis         | VATM [41]         | Bearing& rotor                           | This dataset is a multi-sensor dataset that collects vibration, acoustic, temperature and drive current data of bearing inner and outer rings, shaft misalignment, rotor imbalance and other faults under three different torque load conditions.                                                                                                                                                                 |
| Diagnosis         | HUSTBearing [21]  | Bearing                                  | This dataset collects 9 different failure modes, including 2 groups of bearing failure data at 4 different speeds.                                                                                                                                                                                                                                                                                                |
| Diagnosis         | XJTUSuprgear [23] | Gear                                     | This dataset collects the failure data of spur gears with four different degrees of tooth root cracks under three different working conditions, that is, 900rpm, 1200rpm, and 0-1200rpm-0.                                                                                                                                                                                                                        |
| Diagnosis         | SEU [42]          | Gearbox                                  | The dataset includes four fault types: broken tooth, missing tooth, tooth root crack, and tooth surface wear; the bearing dataset includes four fault types: ball fault, inner ring fault, outer ring fault, and mixed fault.                                                                                                                                                                                     |
| Diagnosis         | XJTUGearbox [23]  | Gearbox                                  | This dataset collects fault datasets of 4 types of gear faults (tooth surface wear, missing teeth, tooth root cracks and broken teeth) and 4 types of bearing faults (inner ring, outer ring, rolling element and mixed faults).                                                                                                                                                                                  |
| Diagnosis         | WT-Gearbox [43]   | Gearbox                                  | This dataset collects broken teeth, tooth surface wear, tooth root cracks, and missing tooth faults in the gearbox. Eight working conditions are considered for each fault. In addition to the X and Y axis vibration signals, the main shaft encoding signal is also collected to consider the impact of equipment disassembly and assembly on the monitoring signal.                                            |
| Diagnosis         | UoC [44]          | Gearbox                                  | In this dataset, the pinion gear mounted on the input shaft was introduced into nine different healthy conditions, including root crack, healthy, spalled, missing teeth, and sharpening, with five different severity levels.                                                                                                                                                                                    |
| Prognosis         | CMAPSS [45]       | Aero-engine                              | This dataset is open-source aviation engine performance degradation data from NASA and consists of four sub-datasets, which are engine performance degradation data under different operating conditions and failure mode combinations                                                                                                                                                                            |
| Prognosis         | N-CMAPSS [46]     | Aero-engine                              | This dataset was also generated by simulation using the CMAPSS software developed by NASA and ETH. It contains 8 subsets, simulating the performance degradation data of 128 engines under 7 different failure modes.                                                                                                                                                                                             |
| Prognosis         | PHM2010 [47]      | Tool                                     | This dataset is the open-source tool wear data of the 2010 PHM competition at different speeds, feed rates, and cutting depths. It consists of 6 sub datasets, each containing 315 samples.                                                                                                                                                                                                                       |
| Prognosis         | IMS [48]          | Bearing                                  | This dataset is a dataset of a bearing run-to-failure experiment. The dataset consists of three subsets, each of which contains the performance degradation data of four bearings.                                                                                                                                                                                                                                |
| Prognosis         | FEMTO-ST [49]     | Bearing                                  | This dataset is the open-source bearing performance degradation data of the 2012 PHM competition, which includes bearing data under three different working conditions.                                                                                                                                                                                                                                           |
| Prognosis         | XJTU-SY [50]      | Bearing                                  | This dataset contains the full-life cycle vibration signals of 15 rolling bearings under three working conditions, and clearly marks the failure location of each bearing.                                                                                                                                                                                                                                        |
| Prognosis         | GearLifeCyle [51] | Gear                                     | This dataset is the full-life vibration data of gears generated by gear fatigue tests conducted using the FZG gear contact fatigue test bench, and includes performance degradation data collected under four operating conditions.                                                                                                                                                                               |



# Code for open Source XIFD methods
<a name="section-id10"></a>
|  Title  |   Journal  |   Date   |   Code   |   Task   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|[**Conditional Contrastive Domain Generalization For Fault Diagnosis**](https://ieeexplore.ieee.org/abstract/document/9721021) <br> | TIM | 2022 | [Github](https://github.com/mohamedr002/CCDG) | Diagnosis |





# Contact

If you have any problem, please feel free to contact me.

Name: Tianfu Li

Email address: tianfu.li@kust.edu.cn

