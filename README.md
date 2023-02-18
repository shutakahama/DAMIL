# Domain Adaptive Multiple Instance Learning for Instance-level Prediction
Implementation of DAMIL (Domain Adaptive Multiple Instance Learning for Instance-level Prediction)  
Accepted by ISBI 2023 [Paper Link (Arxiv)](TBD)  

# Getting Started

```
poetry install
cd src/
poetry run python -m main_damil [LOG_OUTPUT_DIR]
```

# Note

- Experimental logs and plots are saved in `result_log/` directory.
- In our paper, we conducted experiments with Digit dataset, Visda Dataset, and Pathological dataset. In this repository, Only Digit dataset is available.

# Citation
If you use this code for your research, please cite our papers.

```
@article{takahama2023damil,
  title={Domain Adaptive Multiple Instance Learning for Instance-level Prediction of Pathological Images},
  author={Shusuke, Takahama and Yusuke, Kurose and Yusuke, Mukuta and Hiroyuki, Abe and Akihiko, Yoshizawa and Tetsuo, Ushiku and Masashi, Fukayama and Masanobu, Kitagawa and Masaru, Kitsuregawa and Tatsuya, Harada},
  journal={TBD},
  year={2023}
}
```
