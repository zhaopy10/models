import os.path
from shutil import copyfile

split = 'train'   # 'train' or  'validation'
label_name = ['Person', 'Face']
label_map = {
    'Person': [1, '/m/01g317'],  # index and id
    'Face': [4, '/m/0dzct'],
}


img_dir = '../' + split + '/'
out_img = './' + split + '/'
anno_path = '../anno/' + split + '-annotations-bbox.csv'
out_anno = './' + split + '-annotations-bbox.csv'
out_list = './' + split + '.txt'

# go
label_ids = [label_map[name][1] for name in label_name]

lines = None
with open(anno_path) as f:
  lines = f.readlines()
lines = [x.strip('\n') for x in lines]

count = 0
filename = ''
valid = False
lines_to_write = []
with open(out_list, 'w') as list_file:
  with open(out_anno, 'w') as out:
    out.write('ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n')
    for line in lines:
      items = [x.strip() for x in line.split(',')]
      # if got new img and the previous one is valid, then save previous one
      if filename != items[0]:
        if valid:
          for line in lines_to_write:
            out.write(line + '\n')
          copyfile(img_dir + filename + '.jpg', 
              out_img + filename + '.jpg')
          list_file.write(filename + '\n')
          count = count + 1
          if count % 1000 == 0:
            print(str(count) + ' files done.')
        # reset variables
        valid = False
        lines_to_write = []
        filename = items[0]
      # put lines into lines_to_write, and check valid
      label_id = items[2]
      if label_id in label_ids:
        lines_to_write.append(line)
        xr = float(items[5]) - float(items[4])
        yr = float(items[7]) - float(items[6])
        if xr > 0.35 and yr > 0.5:
          valid = True


