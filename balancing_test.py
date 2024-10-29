# 어노테이션 파일 로드하는 함수

import json
import argparse
import copy
import random
from tqdm import tqdm

from hoi_label import hico_text_label, coco_class_dict, valid_obj_ids
hico_text_label_text = dict(zip(hico_text_label.values(), hico_text_label.keys()))
coco_class_dict_text = dict(zip(coco_class_dict.values(), coco_class_dict.keys()))
hico_text_label_invert = dict()

idx = 0
for k, v in hico_text_label.items():
      idx +=1
      hico_text_label_invert[v] = idx

hico_verb_dict = {}

for k,v in hico_text_label.items():
    verb_idx, obj_idx = k[0], k[1]
    if v.split(' ')[6] == 'a' or v.split(' ')[6] == 'an':
        verb = v.split(' ')[5]
    else:
        verb = ' '.join(v.split(' ')[5:7])
    hico_verb_dict[verb_idx+1] = verb

hico_verb_dict_text = dict(zip(hico_verb_dict.values(), hico_verb_dict.keys()))
hico_text_label_save = dict(zip(hico_text_label.values(), hico_text_label.keys()))
idx = 0
for k, v in hico_text_label.items():
      idx +=1
      hico_text_label_save[v] = idx
hico_text_label_invert = dict(zip(hico_text_label_save.values(), hico_text_label_save.keys()))

print('Done')

def argparser():
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='Argparse Tutorial')
    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument('--train_anno_path',          type=str,   default='/nas2/lait/1000_Members/pgb/hico/annotations/trainval_hico.json')
    parser.add_argument('--test_anno_path',     type=str,   default='/nas2/lait/1000_Members/pgb/hico/annotations/test_hico.json')
    parser.add_argument('--ref_anno_path',     type=str,   default='/workspace/softjin/dataset/hoi_dataset/external_hoi/351_external.json')
    
    parser.add_argument('--missing_anno',     action='store_true')
    parser.add_argument('--balance',     action='store_true')
    parser.add_argument('--limit_num',     type=int, default=10)
    parser.add_argument('--ordered_class_num',     type=int, default=351)
    args = parser.parse_args()
    return args


def get_num_instance_per_hoi(annotation):
    hoi_category_id_dict = dict()
    # train set 기준 hoi category id 별 hoi instance 개수 관리
    for anno in annotation:
        for files in anno['hoi_annotation']:
            hoi_category_id = get_hoi_category_id(files, anno['annotations'])

            if hoi_category_id not in hoi_category_id_dict:
                hoi_category_id_dict[hoi_category_id] = 0
            hoi_category_id_dict[hoi_category_id] += 1
    return hoi_category_id_dict

def get_annoatations(anno_path):
      response = anno_path
      with open(response, 'r') as f:
            annotation = json.load(f)
      return annotation

def get_hoi_file_list(annotation):
    hoi_file_list = dict()
    file_hoi_list = dict()

    for anno in annotation:
        for files in anno['hoi_annotation']:
            hoi_category_id = get_hoi_category_id(files, anno['annotations'])
            if hoi_category_id not in hoi_file_list:
                hoi_file_list[hoi_category_id] = []
            if anno['file_name'] not in file_hoi_list:
                file_hoi_list[anno['file_name']] = []
            hoi_file_list[hoi_category_id].append(anno['file_name'])
            file_hoi_list[anno['file_name']].append(hoi_category_id)
    
    return hoi_file_list, file_hoi_list

# 메모리 공유
def get_remove_file(ref_file_pool, file_name):
    for key, value_list in ref_file_pool.items():
        ref_file_pool[key] = [item for item in value_list if item != file_name]
    return ref_file_pool

   
        

def filtering_proces(ref_anno, limit_num, ordered_class_num, total_sorted_data, epoch=10):

    original_file_pool, file_hoi_dict = get_hoi_file_list(ref_anno)

    ref_file_pool = {hoi: [] for hoi in list(total_sorted_data.keys())[:ordered_class_num]}

    for ep in tqdm(range(epoch)):
        # adding
        for ref_hoi_category_id in reversed(list(total_sorted_data.keys())[:ordered_class_num]):
            while len(ref_file_pool[ref_hoi_category_id]) < limit_num:
                file_name = get_image(set(original_file_pool[ref_hoi_category_id]))
                if file_name is None:
                    break

                if file_name in ref_file_pool[ref_hoi_category_id]:
                    continue

                for hoi in file_hoi_dict[file_name]:
                    ref_file_pool[hoi].append(file_name)
                original_file_pool = get_remove_file(original_file_pool, file_name)
      
        # removing
        for ref_hoi_category_id in list(total_sorted_data.keys())[:ordered_class_num]:
            while len(ref_file_pool[ref_hoi_category_id]) > limit_num:
                file_name = get_image(set(ref_file_pool[ref_hoi_category_id]))
                if file_name is None:
                    break
                
                if file_name in original_file_pool[ref_hoi_category_id]:
                    continue

                for hoi in file_hoi_dict[file_name]:
                    original_file_pool[hoi].append(file_name)

                ref_file_pool = get_remove_file(ref_file_pool, file_name)

        # 마지막 에폭
        if ep == epoch -1:
            for ref_hoi_category_id in reversed(list(total_sorted_data.keys())[:ordered_class_num]):
                while len(ref_file_pool[ref_hoi_category_id]) < limit_num:

                    # count files and pick
                    file_lists = original_file_pool[ref_hoi_category_id]
                    file_name = min(file_lists, key=lambda x: abs(file_lists.count(x) - (limit_num - len(ref_file_pool[ref_hoi_category_id]))))

                    if file_name in ref_file_pool[ref_hoi_category_id]:
                        continue

                    for hoi in file_hoi_dict[file_name]:
                        ref_file_pool[hoi].append(file_name)
                    original_file_pool = get_remove_file(original_file_pool, file_name)
        
    return ref_file_pool


def get_image(ref_file_list):
    if len(ref_file_list) == 0:
        return None
    else:
        if isinstance(ref_file_list, set):
            file_name = random.choice(list(ref_file_list))
        else:  
            file_name = random.choice(ref_file_list)
        return file_name


def get_hoi_category_id(hoi_anno, annotations):
    obj = hoi_anno['object_id']
    verb_id = hoi_anno['category_id']
    object_id = annotations[obj]['category_id']            
    hoi_category_id = list(hico_text_label.keys()).index((verb_id-1, valid_obj_ids.index(object_id))) + 1
    return hoi_category_id


def preremove_annotation(ref_anno, involved_class_list):
    new_annotation = []
    for anno in ref_anno:

        new_anno = copy.deepcopy(anno)

        new_hoi = []
        for files in anno['hoi_annotation']:
            hoi_category_id = get_hoi_category_id(files, anno['annotations'])
            if hoi_category_id in involved_class_list:
                new_hoi.append(files)
        
        if len(new_hoi) != 0:
            new_anno['hoi_annotation'] = new_hoi
            new_annotation.append(new_anno)
    return new_annotation

def file_pool_to_annotation(file_pool, annotation):
    new_annotation = []
    set_file_pool = set(item for sublist in file_pool.values() for item in sublist)
    for anno in annotation:
        if anno['file_name'] in set_file_pool:
            new_annotation.append(anno)
    return new_annotation

def remove_annotations(annotation, exceed_hoi_file_list, limit_num, moe = 10):
    for hoi in exceed_hoi_file_list.keys():
        while len(exceed_hoi_file_list[hoi]) > limit_num + moe :
            print('exceed limit num ', len(exceed_hoi_file_list[hoi]), hico_text_label_invert[hoi])
            file_name = random.choice(exceed_hoi_file_list[hoi])

            for anno in annotation:
                if anno['file_name'] == file_name:
                    hoi_index = [idx for idx, hoi_anno in enumerate(anno['hoi_annotation']) if get_hoi_category_id(hoi_anno, anno['annotations']) == hoi]

                    del anno['hoi_annotation'][random.choice(hoi_index)]
                    del exceed_hoi_file_list[hoi][exceed_hoi_file_list[hoi].index(file_name)]
                    continue
            
            
    return annotation

        



if __name__ == '__main__':
    args = argparser()
    
    train_anno = get_annoatations(args.train_anno_path) # train annotation
    test_anno = get_annoatations(args.test_anno_path)   # test annotation

    total_anno = get_annoatations(args.train_anno_path) # train + test annotation
    total_anno.extend(get_annoatations(args.test_anno_path))
    total_hoi_category_id = get_num_instance_per_hoi(total_anno)
    total_sorted_data = dict(sorted(total_hoi_category_id.items(), key=lambda item: item[1] ,reverse=True)) # 딕셔너리를 값에 따라 정렬

    ref_anno = get_annoatations(args.ref_anno_path) # reference annotation

    # ordered_class에 없는 어노테이션들 우선 거르기
    ref_anno = preremove_annotation(ref_anno, list(total_sorted_data.keys())[:args.ordered_class_num])
    ref_hoi_category_id = get_num_instance_per_hoi(ref_anno)


    # filtering
    file_pool = filtering_proces(ref_anno, args.limit_num, args.ordered_class_num, total_sorted_data)
    new_anno = file_pool_to_annotation(file_pool, ref_anno)

    new_anno_hoi_file_list, new_anno_file_hoi_list = get_hoi_file_list(new_anno)
    new_anno = remove_annotations(new_anno, new_anno_hoi_file_list, args.limit_num)
    print()

