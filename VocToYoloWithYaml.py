import xml.etree.ElementTree as ET
import yaml
from pathlib import Path
import argparse
import shutil

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x,y,w,h

def convert_voc_to_yolo(voc_file_path, yolo_path, classes):
    filename = Path(voc_file_path)
    try:
        tree = ET.parse(voc_file_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        with open(yolo_path/f'{filename.stem}.txt', 'w') as f:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                classes.add(cls)
                cls_id = list(classes).index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert((width, height), b)
                f.write(f"{cls_id} {' '.join(map(str, bb))}\n")
    except Exception as e:
        print(f"Error processing file {voc_file_path}: {e}")

def main(dataset_path, output_path, train_val_split):
    classes_names = set()
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Create necessary directories
    (output_path/'labels'/'train').mkdir(parents=True, exist_ok=True)
    (output_path/'labels'/'val').mkdir(parents=True, exist_ok=True)
    (output_path/'images'/'train').mkdir(parents=True, exist_ok=True)
    (output_path/'images'/'val').mkdir(parents=True, exist_ok=True)

    image_files = list((dataset_path/'JPEGImages').glob('*.jpg'))
    
    # Split data into train and validation sets
    train_files = image_files[:int(len(image_files)*train_val_split)]
    val_files   = image_files[int(len(image_files)*train_val_split):]

    # Process training files
    for image_path in train_files:
        shutil.copy(image_path, output_path/'images'/'train'/image_path.name) 
        convert_voc_to_yolo(dataset_path/'Annotations'/f'{image_path.stem}.xml', output_path/'labels'/'train', classes_names)

    # Process validation files
    for image_path in val_files:
        shutil.copy(image_path, output_path/'images'/'val'/image_path.name) 
        convert_voc_to_yolo(dataset_path/'Annotations'/f'{image_path.stem}.xml', output_path/'labels'/'val', classes_names)

    # Generate data.yaml file
    data_dict ={
        'train': '../train/images',
        'val': '../valid/images',
        'nc': len(classes_names),
        'names': list(classes_names),
      }
    
    with open(output_path/'data.yaml', 'w') as f:
        yaml.dump(data_dict, f)

if __name__ == "__main__":
   parser=argparse.ArgumentParser()
   parser.add_argument('--dataset', required=True)
   parser.add_argument('--output', required=True)
   parser.add_argument('--split', type=float,default=0.8)
   opt=parser.parse_args()
   main(opt.dataset,opt.output,opt.split)
