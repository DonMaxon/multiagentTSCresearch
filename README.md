# From Single Agent to Multi-Agent: Improving Traffic Signal Control

</p>

<p align="center">

| **[1 Introduction](#introduction)** 
| **[2 Requirements](#requirements)**
| **[3 Usage](#usage)**
| **[4 Baselines](#baselines)**
| **[5 LightGPT Training](#lightgpt-training)** 
| **[6 Code structure](#code-structure)** 
| **[7 Datasets](#datasets)**
| **[8 Citation](#citation)**


</p>

<a id="introduction"></a>
## 1 Introduction

Official code for article "[From Single Agent to Multi-Agent: Improving Traffic Signal Control]([https://arxiv.org/abs/2312.16044](https://arxiv.org/abs/2406.13693))".

Due to accelerating urbanization, the importance of solving the signal control problem increases.
This paper analyzes various existing methods and suggests options for increasing the number of agents
to reduce the average travel time. Experiments were carried out with 2 datasets. The results show that
in some cases, the implementation of multiple agents can improve existing methods. For a fine-tuned
large language model approach there’s small enhancement on all metrics.

<a id="requirements"></a>
## 2 Requirements

`python>=3.9`,`tensorflow-cpu=2.8.0`, `cityflow`, `pandas=1.5.0`, `numpy=1.26.2`, `wandb`,  `transformers=4.36.2`, `peft=0.7.1`, `accelerate=0.25.0`, `datasets=2.16.1`, `fire`

[`cityflow`](https://github.com/cityflow-project/CityFlow.git) needs a Linux environment, and we run the code on Ubuntu.

<a id="usage"></a>

## 3 Usage

Parameters are well-prepared, and you can run the code directly.

- For example, to run `Advanced-MPLight`:
```shell
python run_advanced_mplight.py --dataset hangzhou \
                               --traffic_file anon_4_4_hangzhou_real.json \
                               --proj_name TSCS
```
- To run GPT-3.5/GPT-4 with LLMLight, you need to set your key in `./models/chatgpt.py`:

```python
headers = {
    "Content-Type": "application/json",
    "Authorization": "YOUR_KEY_HERE"
}
```

Then, run LLMLight by:

```shell
python run_chatgpt.py --prompt Commonsense \
                      --dataset hangzhou \
                      --traffic_file anon_4_4_hangzhou_real.json \
                      --gpt_version gpt-4 \
                      --proj_name TSCS
```
You can either choose `Commonsense` or `Wait Time Forecast` as the `prompt` argument.

- To run open-sourced LLMs with LLMLight:

```shell
python run_open_LLM.py --llm_model LLM_MODEL_NAME_ONLY_FOR_LOG \
                       --llm_path LLM_PATH \
                       --dataset hangzhou \
                       --traffic_file anon_4_4_hangzhou_real.json \
                       --proj_name TSCS
```
<a id="baselines"></a>

## 4 Baselines

- **Heuristic Methods**:
    - FixedTime, Maxpressure, EfficientMaxPressure
- **DNN-RL**:
    - PressLight, MPLight, CoLight, AttendLight, EfficientMPLight, EfficientPressLight, Efficient-Colight
- **Adv-DNN-RL**:
    - Advanced-MaxPressure, Advanced-MPLight, Advanced-Colight
- **LLMLight+LLM**:
  - `gpt-3.5-turbo-0613`, `gpt-4-0613`, `llama-2-13b-chat-hf`, `llama-2-70b-chat-hf`
- **LLMLight+LightGPT**:
    - The model trained on Jinan 1 is available at https://huggingface.co/USAIL-HKUSTGZ/LLMLight-LightGPT

<a id="code-structure"></a>

## 5 Code structure

- `models`: contains all the models used in our article.
- `utils`: contains all the methods to simulate and train the models.
- `frontend`: contains visual replay files of different agents.
- `errors`: contains error logs of ChatGPT agents.
- `{LLM_MODEL}_logs`: contains dialog log files of a LLM.
- `prompts`: contains base prompts of ChatGPT agents.
- `finetune`: contains codes for LightGPT training.

<a id="datasets"></a>
## 6 Datasets

<table>
    <tr>
        <td> <b> Road networks </b> </td> <td> <b> Intersections </b> </td> <td> <b> Road network arg </b> </td> <td> <b> Traffic files </b> </td>
    </tr>
    <tr> <!-- Jinan -->
        <th rowspan="4"> Jinan </th> <th rowspan="4"> 3 X 4 </th> <th rowspan="4"> jinan </th>  <td> anon_3_4_jinan_real </td> 
    </tr>
  	<tr>
      <td> anon_3_4_jinan_real_2000 </td>
  	</tr>
  	<tr>
      <td> anon_3_4_jinan_real_2500 </td>
    </tr>
    <tr>
      <td> anon_3_4_jinan_synthetic_24000_60min </td>
    </tr>
  	<tr> <!-- Hangzhou -->
        <th rowspan="3"> Hangzhou </th> <th rowspan="3"> 4 X 4 </th> <th rowspan="3"> hangzhou </th> <td> anon_4_4_hangzhou_real </td>
    </tr>
  	<tr>
      <td> anon_4_4_hangzhou_real_5816 </td>
    </tr>
    <tr>
      <td> anon_4_4_hangzhou_synthetic_32000_60min </td>
    </tr>
  <tr> <!-- Newyork -->
        <th rowspan="2"> New York </th> <th rowspan="2"> 28 X 7 </th> <th rowspan="2"> newyork_28x7 </th> <td> anon_28_7_newyork_real_double </td>
    </tr>
  	<tr>
      <td> anon_28_7_newyork_real_triple </td>
    </tr>
</table>

<a id="citation"></a>

## 7 Citation

```
@misc{tislenko2024singleagentmultiagentimproving,
      title={From Single Agent to Multi-Agent: Improving Traffic Signal Control}, 
      author={Maksim Tislenko and Dmitrii Kisilev},
      year={2024},
      eprint={2406.13693},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2406.13693}, 
}
```
