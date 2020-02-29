## The Sales Unit Prediction Model (SKU Level)
### Purpose
1) To predict customers' demands regarding specific SKUs.
2) The model should predict the number of sales, 9 days in advance. 

### Dataset
1) Each rows of dataset has 3 weeks (21 days) features (Time-Series)
2) Entire dataset size is about 160k.
3) There are 3 inputs.<br/>
3-1) Basic Numeric Features<br/>
3-2) Feature Engineered Features : Likes Moving Average & STD<br/>
3-3) Date Features
4) There are 2 outpus.<br/>
4-1) Auxiliary output (The Number of Sales Unit 1 day in advance)<br/>
4-2) Main output (The Number of Sales Unit 9 days in advance / The model's target)

### Model
![ex_screenshot](./image/model_fig.png)
