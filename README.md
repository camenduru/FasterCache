<div align="center">
<h1>FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality</h1></div>



<div align="center">
    <a href="https://scholar.google.com/citations?user=FkkaUgwAAAAJ&hl=en" target="_blank">Zhengyao Lv</a><sup>1</sup> |
    <a href="https://chenyangsi.github.io/" target="_blank">Chenyang Si</a><sup>2‡</sup> |
    <a href="" target="_blank">Junhao Song</a><sup>3</sup> |
    <a href="" target="_blank">Zhenyu Yang</a><sup>3</sup> |
    <a href="https://mmlab.siat.ac.cn/yuqiao" target="_blank">Yu Qiao</a><sup>3</sup> |
    <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu</a><sup>2†</sup>    |
    <a href="https://i.cs.hku.hk/~kykwong/" target="_blank">Kwan-Yee K. Wong</a><sup>1†</sup>
</div>
<div align="center">
    <sup>1</sup>The University of Hong Kong &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
    <sup>2</sup>S-Lab, Nanyang Technological University <br>
    <sup>3</sup>Shanghai Artificial Intelligence Laboratory
</div>
<div align="center">(‡: Project lead; †: Corresponding authors)</div>

​<p align="center">
    <a href="https://arxiv.org/abs/2410.19355">Paper</a> | 
    <a href="https://vchitect.github.io/FasterCache/">Project Page</a>
</p>

<p align="center">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FFasterCache&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Github+visitors&edge_flat=false"/></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fvchitect.github.io%2FFasterCache%2F&count_bg=%23C83D5D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Pages+visitors&edge_flat=false"/></a>
</p>



## About

We present ***FasterCache***, a novel training-free strategy designed to accelerate the inference of video diffusion models with high-quality generation. For more details and visual results, go checkout our [Project Page](https://vchitect.github.io/FasterCache/).

https://github.com/user-attachments/assets/035c50c2-7b74-4755-ac1e-e5aa1cffba2a

## Usage

### Installation

Run the following instructions to create an Anaconda environment.

```
conda create -n fastercache python=3.10 -y
conda activate fastercache
git clone https://github.com/Vchitect/FasterCache
cd FasterCache
pip install -e .
```

### Inference

We currently support [Open-Soa 1.2](https://github.com/hpcaitech/Open-Sora), [Open-Sora-Plan 1.1](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [Latte](https://github.com/Vchitect/Latte), [CogvideoX-2B](https://github.com/THUDM/CogVideo), and [Vchitect 2.0](https://github.com/Vchitect/Vchitect-2.0). You can achieve accelerated sampling by executing the scripts we provide.

- **Open-Sora**

For single-GPU inference on Open-Sora, run the following command:
```
bash scripts/opensora/fastercache_sample_opensora.sh
```

For multi-GPU inference on Open-Sora, run the following command:

```
bash scripts/opensora/fastercache_sample_multi_device_opensora.sh
```

- **Open-Sora-Plan**

For single-GPU inference on Open-Sora-Plan, run the following command:
```
bash scripts/opensora_plan/fastercache_sample_opensoraplan.sh
```

For multi-GPU inference on Open-Sora-Plan, run the following command:

```
bash scripts/opensora_plan/fastercache_sample_multi_device_opensoraplan.sh
```

- **Latte**


For single-GPU inference on Latte, run the following command:
```
bash scripts/latte/fastercache_sample_latte.sh
```

For multi-GPU inference on Latte, run the following command:

```
bash scripts/latte/fastercache_sample_multi_device_latte.sh
```

- **CogVideoX**

For inference on CogVideoX, run the following command:
```
bash scripts/cogvideox/fastercache_sample_cogvideox.sh
```

- **Vchitect 2.0**

For inference on Vchitect 2.0, run the following command:
```
bash scripts/vchitect/fastercache_sample_vchitect.sh
```

## BibTeX

```
@inproceedings{lv2024fastercache,
  title={FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality},
  author={Lv, Zhengyao and Si, Chenyang and Song, Junhao and Yang, Zhenyu and Qiao, Yu and Liu, Ziwei and Kwan-Yee K. Wong},
  booktitle={arxiv},
  year={2024}
}
```

## Acknowledgement

This repository borrows code from [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), [Vchitect-2.0](https://github.com/Vchitect/Vchitect-2.0), and [CogVideo](https://github.com/THUDM/CogVideo),.Thanks for their contributions!
