# Age predictor from single selfie

## Installation

```bash
git clone https://github.com/Kichkun/age_model.git
cd age_model
pip install .
```

**Age recognition** 
```python
from age_model import AgeModel
import cv2
img = cv2.imread('img4.jpg')
demography = AgeModel.analyze(img)
#demographies = DeepFace.analyze([img, img, img]) #analyzing multiple faces same time
print("Age: ", demography["age"])
```

<p align="center"><img src="https://makeameme.org/media/templates/250/the_most_interesting_man_in_the_world.jpg" width="30%" height="20%"></p>

```bash
{'age': 50.75593731981769}
```

**Pretrained weights** can be downloaded from
[here](https://drive.google.com/file/d/1NmS_6TgHNjXfkZpMyqs5y47Bn70-unNi/view?usp=sharing) and should be placed in "models" folder
