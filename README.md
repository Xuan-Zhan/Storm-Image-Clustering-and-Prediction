# Elin - The Day After Tomorrow

Predicting tropical storm behaviour through Deep Learning.

![cover_image](https://drive.google.com/uc?export=view&id=1EEZ4J-i4xY8xImuRZ6p2J1VFeHXaLtWq)

## Background and Objectives

The FEMA (Federal Emergency Management Agency) in the US have released an open competition to improve their emergency protocols under hurricane threats.​ This project aims to 

* Given data from one storm, where some satellite images are available, to **generate 3 future image predictions** based on these existing images for that given storm.
* Train a network to **predict wind speeds** based on the training data provided.

## Getting Started

To run the code in this repo, please follow the steps below.

1. Clone this repository

```bash
git clone https://github.com/ese-msc-2023/acds-the-day-after-tomorrow-elin.git
```

2. Run commands below to install necessary libraries:

```bash
cd the-directory-name-where-the-repo-is
pip install -r requirements.txt
```

We mainly run our project on Google Colab. To run the notebooks, if necessary, make sure you modify the relevant paths. All notebooks that have been run in Colab have an explanation as to which code needs to be changed to run within Colab. 

### Image Generation API
As part of the image generation, we have built an API that can allow you to run inference in model weights (in the `models` folder), specifcy a sequence of images (min. 14) and generate a specified amount of images that continue the sequence provided. For its usage, find it at the bottom of the **[SurpriseStormImage.ipynb](https://github.com/ese-msc-2023/acds-the-day-after-tomorrow-elin/blob/main/notebooks/SurpriseStormImage.ipynb)** Notebook. Alternatively, you can create your own Python file and run inference on the model as follows:

```python
import torch

# load the images
image_directory = 'your-image-directory'
image_paths = [os.path.join(image_directory, f) for f in get_last_images(image_directory)]
initial_images = [load_and_preprocess_image(path) for path in image_paths]

# load the model
model = torch.load_sate_dict(torch.load('../models/128_image_model.pth'))

# predict the next images
predict_next_images(model, initial_images)
```
The code above yields the image sequence below demonstrated as a GIF, which happen to be the continued 3 frames from the unseen surprise storm.

![tst_storm](https://github.com/ese-msc-2023/acds-the-day-after-tomorrow-elin/assets/142490406/67a13234-da91-4b3f-b288-9f8b02515bda)

## Repository Structure
- **workflows:** Configuration folder containing `.yml` files for GitHub Actions.

- **elin_generatedimages**: Contains three image files `tst_252.jpg`, `tst_253.jpg` and `tst_254.jpg`, representing the results of image predictions.

- **models:** Contains two files, where `128_image_model.pth` and `64_image_model.pth` respectively represent models trained on images resized to 128x128 and 64x64 dimensions.

- **notebooks:**

  - `EDA.ipynb`: Perform exploratory data analysis to determine the number of storm images, examine label distribution, analyse time series, and explore other relevant factors.

  - `ImageGenerationWorkflow.ipynb`: Provides a step-by-step workflow for generating images, including code, explanations, and visualizations.

  - `WindPredictionWorkflow.ipynb`: Provides a step-by-step workflow for speed prediction, including code, explanations, and visualizations.

  - `SurpriseStormImage.ipynb`: Shows the results of running the image generation model on the surprisestorm dataset.

  - `SurpriseStormWind.ipynb`: Shows the results of running the speed prediction model on the surprisestorm dataset.

- **prediction:** Comprises a series of `.py` files with classes and functions intended for repeated invocation, supporting the execution of code within the notebook.

- **tests:** Contains a folder called bhk for test data, and a test file `test_prediction.py` to validate the functionality and performance of the models.

## Wind Speed Prediction Model
Below is a link to the final weights of the windspeed prediction model. The file is over a GB in size so adding it to the repository is unnecessary. If you have any issues with downloading or accessing the file please contact [Antony Krymski](#Authors).<br>
https://drive.google.com/file/d/1-Y5q_6vkOF3gMr3fCXDXpElT0Be5WUoc/view?usp=sharing

## Results & Architectures

| Task                   | Model    | Layers | Learning Rate | Batch Size |  Image Size | Loss Function   |
|------------------------|----------|--------|---------------|------------| ------------- |-----------------|
| Image Generation       | ConvLSTM | 6 ConvLSTM layers<br>6 BatchNorm2d layers<br>2 Fully connected layers       |  2e-4         | 8          |(128, 128) | BCE Loss         |
| Wind Speed Prediction  | ConvLSTM | 3 ConvLSTM layers<br>3 BatchNorm2d layers<br>2 Fully connected layers       |     1e-4          |  32          | (64, 64) | MSE Loss               |


## Future Plans

- [ ] Time intervals of the storms are unequal. For wind speed, interpolation can be employed to obtain values with equal time intervals, while for images, the image from the previous time step can be reused.
     
- [ ] There are blank spaces in the images, some are lines, and some are blocks. The solutions we have considered include removing all images from the beginning or end up to the missing image, or employing machine learning methods to fill in the blank space.

- [ ] Incorporate labels and features into the image generation process to enhance its ability to generalize effectively.

- [ ] Utilize an alternative model for image generation.
      

## Data
We train and test models on the dataset which contains information below:

- Satellite images of tropical storms​.
- 30 storms around the Atlantic and East Pacific Oceans (precise locations undisclosed)​.
- Each with varied number of time samples and irregular time sampling.​
- Features: storm id, ocean (1 or 2), and time stamp.​
- Label: wind speed.

|    | image_id | storm_id | relative_time | ocean | wind_speed |
| -- | -------- | -------- | ------------- | ----- | ---------- |
| 0  | abs_000  | abs      | 0             | 2     | 43         |
| 1  | abs_001  | abs      | 1800          | 2     | 44         |
| 2  | abs_002  | abs      | 5400          | 2     | 45         |
| 3  | abs_003  | abs      | 17999         | 2     | 52         |
| 4  | abs_004  | abs      | 19799         | 2     | 53         |

You can access the dataset by clicking the link: https://drive.google.com/drive/folders/1tFqoQl-sdK6qTY1vNWIoAdudEsG6Lirj?usp=drive_link

## Authors
- Antony Krymski: antony.krymski23@imperial.ac.uk
- Artemiy Malyshau: artemiy.malyshau23@imperial.ac.uk
- Ce Huang: ce.huang23@imperial.ac.uk
- Haoran Zhang: haoran.zhang23@imperial.ac.uk
- Jing-Han HUANG: jing-han.huang23@imperial.ac.uk
- Xuan Zhan: xuan.zhan23@imperial.ac.uk
- Xueling Gao: xueling.gao23@imperial.ac.uk
- Yunjie Li: yunjie.li23@imperial.ac.uk

## Reference
See the [references.md](https://github.com/ese-msc-2023/acds-the-day-after-tomorrow-elin/blob/main/references.md) file

## License
Please find the MIT License for this repository [here](https://github.com/ese-msc-2023/acds-the-day-after-tomorrow-elin?tab=MIT-1-ov-file).
