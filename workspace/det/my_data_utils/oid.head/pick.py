import os.path
from shutil import copyfile

#label_name = ['Person', 'Face', 'Head', 'Human body']
#label_map = {
#    'Person': [1, '/m/01g317'],  # index and id
#    'Face': [4, '/m/0dzct'],
#    'Head': [39, '/m/04hgtk'],
#    'Human body': [32, '/m/02p0tk3'],
#}
label_name = ['Head']
label_map = {'Head': [39, '/m/04hgtk']}

input_dir = '/home/xuyithu/data/'
output_dir = '/home/xuyithu/workspace/data/'


for split in ['validation', 'train']:
  print('Start doing %s.', split)

  img_dir = input_dir + split + '/'
  out_img = output_dir + split + '/'
  anno_path = input_dir + 'annotations/' + split + '-annotations-bbox.csv'
  out_anno = output_dir + split + '-annotations-bbox.csv'
  out_list = output_dir + split + '.txt'
  
  # go
  label_ids = [label_map[name][1] for name in label_name]
  
  lines = None
  with open(anno_path) as f:
    lines = f.readlines()
  lines = [x.strip('\n') for x in lines]
  
  ori_count = 0
  count = 0
  filename = ''
  valid = False
  withoutgroup = True
  lines_to_write = []
  with open(out_list, 'w') as list_file:
    with open(out_anno, 'w') as out:
      out.write('ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n')
      for line in lines:
        items = [x.strip() for x in line.split(',')]
        # if got new img and the previous one is valid, then save previous one
        if filename != items[0]:
          ori_count += 1
          if valid and withoutgroup:
            for line in lines_to_write:
              out.write(line + '\n')
            copyfile(img_dir + filename + '.jpg', 
                out_img + filename + '.jpg')
            list_file.write(filename + '\n')
            count = count + 1
            if count % 1000 == 0:
              print(str(count) + ' files done out of ' + str(ori_count) + '.')
          # reset variables
          valid = False
          withoutgroup = True
          lines_to_write = []
          filename = items[0]
        # put lines into lines_to_write, and check valid
        label_id = items[2]
        if label_id in label_ids:
  #        lines_to_write.append(line)
          xr = float(items[5]) - float(items[4])
          yr = float(items[7]) - float(items[6])
          isGroupOf = int(items[10])
          isDepiction = int(items[11])
          if isGroupOf:
            withoutgroup = False
          if xr > 0.0 and yr > 0.0 and (not isGroupOf) and (not isDepiction):
            lines_to_write.append(line)
            valid = True


