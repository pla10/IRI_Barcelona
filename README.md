# Learning Priors of Human Motion With Vision Transformers

This repository provides the implementation for the paper:  
**"Learning Priors of Human Motion With Vision Transformers"**  
Published in *COMPSAC 2024*. DOI: [10.1109/COMPSAC61105.2024.00060](https://doi.org/10.1109/COMPSAC61105.2024.00060)

## Overview

The paper introduces a novel neural architecture based on Vision Transformers (ViTs) to learn priors of human motion. This method offers a powerful solution for predicting typical human trajectories and behaviors in various settings. Applications include:
- Urban mobility analysis
- Robot navigation
- Crowd behavior modeling

### Features
- Vision Transformer-based motion analysis
- Supports generalization to diverse environments
- Python implementation with sample datasets

## Installation

Clone this repository and install the necessary Python dependencies using:
```bash
git clone https://github.com/pla10/IRI_Barcelona.git
cd IRI_Barcelona
pip install -r requirements.txt
```

## Usage
- Prepare Dataset: Provide motion trajectory datasets in the expected format.
- Train Model: Use the provided training scripts to train the Vision Transformer.
- Evaluation: Evaluate the model's predictions using the included test scripts.

## Reference
If you find this work helpful, please cite:

```sql
@INPROCEEDINGS{falqueto2024learning,
  author={Falqueto, Placido and Sanfeliu, Alberto and Palopoli, Luigi and Fontanelli, Daniele},
  booktitle={2024 IEEE 48th Annual Computers, Software, and Applications Conference (COMPSAC)}, 
  title={Learning Priors of Human Motion With Vision Transformers}, 
  year={2024},
  volume={},
  number={},
  pages={382-389},
  keywords={Computer vision;Accuracy;Navigation;Urban areas;Computer architecture;Transformers;Trajectory;vision transformers;human motion prediction;semantic scene understanding;masked autoencoders;occupancy priors},
  doi={10.1109/COMPSAC61105.2024.00060}}

```

## License
This project is licensed under MIT License.

## Acknowledgments
Co-funded by the European Union. Views and opinions expressed are
however those of the author(s) only and do not necessarily reflect
those of the European Union or the European Commission. Neither the
European Union nor the granting authority can be held responsible for
them (EU - HE Magician – Grant Agreement 101120731). Moreover, this
work was partially funded by the European Commission grant number
101016906 (Project CANOPIES).
