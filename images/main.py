import os
from glob import glob
import pandas as pd
from xml.etree import ElementTree as et
from functools import reduce
from shutil import move
import warnings

warnings.filterwarnings('ignore')

# step-1: get path of each xml file
xmlfiles = glob('./images/*.xml')
xmlfiles = [x.replace('\\', '/') for x in xmlfiles]

# step-2: read xml files
def extract_text(filename):
    tree = et.parse(filename)
    root = tree.getroot()

    image_name = root.find('filename').text
    width = root.find('size').find('width').text
    height = root.find('size').find('height').text
    objs = root.findall('object')
    parser = []
    for obj in objs:
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        xmax = bndbox.find('xmax').text
        ymin = bndbox.find('ymin').text
        ymax = bndbox.find('ymax').text
        parser.append([image_name, width, height, name, xmin, xmax, ymin, ymax])
        
    return parser

parser_all = list(map(extract_text, xmlfiles))
data = reduce(lambda x, y: x + y, parser_all)

df = pd.DataFrame(data, columns=['filename', 'width', 'height', 'name', 'xmin', 'xmax', 'ymin', 'ymax'])

# type conversion
cols = ['width', 'height', 'xmin', 'xmax', 'ymin', 'ymax']
df[cols] = df[cols].astype(int)

# calculate normalized values (YOLO format)
df['center_x'] = ((df['xmax'] + df['xmin']) / 2) / df['width']
df['center_y'] = ((df['ymax'] + df['ymin']) / 2) / df['height']
df['w'] = (df['xmax'] - df['xmin']) / df['width']
df['h'] = (df['ymax'] - df['ymin']) / df['height']

# split into train and test
images = df['filename'].unique()
img_df = pd.DataFrame(images, columns=['filename'])
img_train = tuple(img_df.sample(frac=0.8)['filename'])
img_test = tuple(img_df.query(f'filename not in {img_train}')['filename'])

train_df = df.query(f'filename in {img_train}')
test_df = df.query(f'filename in {img_test}')

# label encoding (only one class: pothole)
def label_encoding(x):
    labels = {'pothole': 0}
    return labels[x]

train_df['id'] = train_df['name'].apply(label_encoding)
test_df['id'] = test_df['name'].apply(label_encoding)

# group by filename
cols = ['filename', 'id', 'center_x', 'center_y', 'w', 'h']
groupby_obj_train = train_df[cols].groupby('filename')
groupby_obj_test = test_df[cols].groupby('filename')

# create YOLO folder structure
train_img_folder = 'dataset/images/train'
train_label_folder = 'dataset/labels/train'
test_img_folder = 'dataset/images/test'
test_label_folder = 'dataset/labels/test'

os.makedirs(train_img_folder, exist_ok=True)
os.makedirs(train_label_folder, exist_ok=True)
os.makedirs(test_img_folder, exist_ok=True)
os.makedirs(test_label_folder, exist_ok=True)

# save function
def save_data(filename, img_folder, label_folder, group_obj):
    # move image
    src = os.path.join('images', filename)
    dst = os.path.join(img_folder, filename)
    if os.path.exists(src):  # avoid errors if already moved
        move(src, dst)
    
    # save label (YOLO format txt)
    text_filename = os.path.join(label_folder, os.path.splitext(filename)[0] + '.txt')
    group_obj.get_group(filename).set_index('filename').to_csv(
        text_filename, sep=' ', index=False, header=False
    )

# save train data
filename_series = pd.Series(groupby_obj_train.groups.keys())
filename_series.apply(save_data, args=(train_img_folder, train_label_folder, groupby_obj_train))

# save test data
filename_series_test = pd.Series(groupby_obj_test.groups.keys())
filename_series_test.apply(save_data, args=(test_img_folder, test_label_folder, groupby_obj_test))

print("✅ Dataset prepared successfully in YOLO format!")
