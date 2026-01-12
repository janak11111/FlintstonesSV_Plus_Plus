
# 🚀 FlintstonesSV++: Improving Story Narration using Visual Scene Graph  

## Overview  

FlintstonesSV++ is a dataset and framework designed to improve story narration by leveraging **Visual Scene Graphs** to enhance the quality of scene descriptions. It also provides tools to finetune story visualization models for better narrative coherence and visual storytelling.  

---

### FlintstonesSV++ Dataset  

🤗🤗🤗 [Huggingface Link](https://huggingface.co/datasets/Janak12/FlintstonesSV_Plus_Plus)

```bash
# Code to download the dataset 
from datasets import load_dataset

dataset = load_dataset("Janak12/FlintstonesSV_Plus_Plus")
```  

---

## Content  

- [Installation](#installation)  
- [FlintstonesSV++ Dataset](#flintstonessv-dataset)  
- [Generate Visual Scene Graph](#generate-visual-scene-graph)  
- [Improve Story Narration](#improve-story-narration)  
- [Finetune Story Visualization Models](#finetune-story-visualization-models)  

---

## Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/janak11111/FlintstonesSV_Plus_Plus
   cd FlintstonesSV_Plus_Plus
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Create and activate the conda environment:
   ```
   conda create -n story python=3.10 -y
   conda activate story
   ```
   

---

## FlintstonesSV++ Dataset  

Download and load the dataset:  

```bash
# Run in Python environment
from datasets import load_dataset

dataset = load_dataset("Janak12/FlintstonesSV_Plus_Plus")
```  

---

## Generate Visual Scene Graph  

```bash
# Run the scene graph generation script
cd Scripts
python VSG_Generation.py
```  

---

## Improve Story Narration  

```bash
# Run the story narration improvement script
cd Scripts
python Scene_Narrative_Generation.py 
```  

---

## Finetune Story Visualization Models  

### Run the fine-tuning script for SDXL Model
```bash
CUDA_VISIBLE_DEVICES=0  accelerate launch Text_to_Image_mage_finetuning_SDXL_lora.py \
    --pretrained_model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0'  \
    --pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix'\
    --data_json_file=ADD JSON TRAINING DATA PATH
    --train_data_dir='temp' \
    --caption_column="text" \
    --image_column="image" \
    --rank=4 \
    --train_text_encoder \
    --resolution=512 \
    --seed=42 \
    --random_flip \
    --train_batch_size=4 \
    --num_train_epochs=10 \
    --checkpointing_steps=1000 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=0 \
    --allow_tf32 \
    --mixed_precision='fp16' \
    --output_dir='/Checkpoints' \
    --validation_prompt="Fred is in the living room. Fred walks by Wilma and then takes a plate of food from his hands." \
    --num_validation_images=4 \
    --report_to="wandb"
```  

---

### Run the fine-tuning script for other Models
```bash
CUDA_VISIBLE_DEVICES=1  accelerate launch /home/jankap/story_visulization/Finetuning/Scripts/text_2_image_finetuning_lora.py \
    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
    --train_data_dir='temp' \
    --caption_column="text" \
    --data_json_file=ADD JSON TRAINING DATA PATH
    --image_column="image" \
    --rank=4 \
    --resolution=512 \
    --seed=42 \
    --random_flip \
    --train_batch_size=4 \
    --num_train_epochs=10 \
    --checkpointing_steps=1000 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --allow_tf32 \
    --mixed_precision='fp16' \
    --output_dir='Checkpoints' \
    --validation_prompt="Fred is in the living room. Fred walks by Wilma and then takes a plate of food from his hands." \
    --num_validation_images=4 \
    --report_to="wandb"
```

---

### Inference script for Text to Image Generation
```bash
CUDA_VISIBLE_DEVICES=0 python Text_to_Image_Inference_sdxl_lora.py \
    --lora_path /path/to/lora/checkpoint-24000 \
    --test_json /path/to/flintstoneSV++_test.json \
    --output_dir ./outputs
```


---
## 📝 Paper  

**Title:** [FlintstonesSV++: Improving Story Narration using Visual Scene Graph](https://ceur-ws.org/Vol-3964/paper3.pdf))  
**Accepted at:** Text2Story Workshop, ECIR Conference 2025, Lucca, Italy.  
**Authors**: *Janak Kapuriya*, *Paul Buitelaar*  
**Organization:** Insight Research Ireland Center for Data Analytics, Data Science Institute, University of Galway, Ireland.  

---

## 🤗 Contribution  

We welcome contributions! Feel free to submit issues or pull requests.  

---

## 📬 Get in Touch  

Feel free to reach out if you have any questions or suggestions!  

**Janak Kapuriya**  
📧 Email: [janakkumar.kapuriya@insight-centre.org](mailto:janakkumar.kapuriya@insight-centre.org)  


## Citation

If you find this project useful, please cite our work:

```
@article{kapuriya2025flintstonessv++,
  title={FlintstonesSV++: Improving Story Narration using Visual Scene Graph},
  author={Kapuriya, Janak and Buitelaar, Paul},
  year={2025}
}
```
