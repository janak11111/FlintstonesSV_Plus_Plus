
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

```bash
# Run the fine-tuning script
python finetune_visualization_model.py --config config/finetune_config.yaml
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
