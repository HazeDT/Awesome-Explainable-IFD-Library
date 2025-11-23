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
| Global | Knowledge distillation [60]                                                              |  not        |   yes       |
| Global | Activation maximization: AM [61]                                                         |  not        |    yes      |
| Local  | Local approximation method: LIME [62], SP-LIME [63], S-LIME [64], ALIME [65], ILIME [66] |  yes        |   yes       |
| Local  | Gradient based method: Guided-BP [67], Smooth gradients [68], Integrated gradients [69]  |  not        |    yes      |
| Local  | Class activation mapping: CAM [70], Grad-CAM [71], Grad-CAM++ [72], LRP [61],            |   not       |    yes      |
| Local  | SHAP based method: SHAP [73]                                                             |    yes      |   yes       |


### 1) Globally explainable methods
<a name="section-id3"></a>
Global explainability aims to help people understand the overall logic behind the model and its inner working mechanism
#### 2025


### 2) Locally explainable methods
<a name="section-id4"></a>
#### 2025

#### 2024

## Ante-hoc explainability methods
<a name="section-id5"></a>
### Attention Mechanism Assisted Intelligent Fault Diagnosis Methods 
<a name="section-id6"></a>
Attention mechanisms provide a way to help understand what to focus on during model learning, and it allows the model to automatically and selectively focus on important information and ignore unimportant information when dealing with large amounts of input data in order to achieve transparency and visualization of the decision-making process, which in turn enhances the model explainability.
### 1) Non-self-attention-based
#### 2025

### 2) Self-attention-based
#### 2025


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

