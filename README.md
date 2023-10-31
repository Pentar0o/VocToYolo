# VocToYoloWithYaml

## Description
This project contains a Python script `VocToYoloWithYaml.py` which is designed to convert VOC (Visual Object Classes) annotations to YOLO (You Only Look Once) format. The script also splits the dataset into training and validation sets, and generates a `data.yaml` file containing class names and other information.

## Installation
To install and run this project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/Pentar0o/VocToYolo.git
   ```
2. Navigate to the project directory:
   ```
   cd VocToYolo
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To use this script, run the following command:
```
python VocToYoloWithYaml.py --dataset <path_to_dataset> --output <path_to_output_directory> --split <train_val_split_ratio>
```
Replace `<path_to_dataset>`, `<path_to_output_directory>`, and `<train_val_split_ratio>` with your specific paths and split ratio respectively.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
