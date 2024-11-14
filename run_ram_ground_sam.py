import argparse
import os
import time, glob

import numpy as np
import json
import torch
import torchvision
from PIL import Image
import litellm
import check_category

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap, get_grouped_tokens

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS

# ChatGPT or nltk is required when using tags_chineses
# import openai
# import nltk

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def check_tags_chinese(tags_chinese, pred_phrases, max_tokens=100, model="gpt-3.5-turbo"):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    if openai_key:
        prompt = [
            {
                'role': 'system',
                'content': 'Revise the number in the tags_chinese if it is wrong. ' + \
                           f'tags_chinese: {tags_chinese}. ' + \
                           f'True object number: {object_num}. ' + \
                           'Only give the revised tags_chinese: '
            }
        ]
        response = litellm.completion(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
        reply = response['choices'][0]['message']['content']
        # sometimes return with "tags_chinese: xxx, xxx, xxx"
        tags_chinese = reply.split(':')[-1].strip()
    return tags_chinese


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    '''
    Output:
        boxes_filt: (num_filt, 4)
        scores: (num_filt, 256), tensor.float32
        pred_phrases: list of str, label(conf), unaligned
        pred_phrases_set: list of token token map. 
                            In each token map, there is a map of pair {token_name:'score}
    '''
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    grouped_token_ids, posmap_to_prompt_id = get_grouped_tokens(tokenized['input_ids'], tokenlizer)
    assert len(grouped_token_ids)+1 == len(caption.split('.'))
    
    # build pred
    bbox_matrix = []
    scores = [] # num_filt, list of max score in each bbox
    pred_phrases = []   # num_filt, list of unaligned phrases {label_name:score}
    pred_phrases_set = [] # num_filt, list of token map {label_name:score}


    for logit, box in zip(logits_filt, boxes_filt):
        token_map    = {}
        max_label = ''
        max_score = 0.0

        # get phrases in input texts
        posmap     = logit > text_threshold
        prompt_ids = posmap_to_prompt_id[posmap.nonzero(as_tuple=True)[0]]
        prompt_ids = torch.unique(prompt_ids)
        assert prompt_ids.min()>=0 and prompt_ids.max()<len(grouped_token_ids)

        for prompt_id in prompt_ids:
            prompt_posmap = grouped_token_ids[prompt_id]
            prompt_tokens = [tokenized['input_ids'][i] for i in prompt_posmap]
            pred_label = tokenlizer.decode(prompt_tokens)
            pred_score = logit[prompt_posmap].max().item()
            if logit[prompt_posmap].min().item()<text_threshold:
                print('[WARN] filter {} with scores {}'.format(pred_label, logit[prompt_posmap].tolist()))
                continue
            
            token_map[pred_label] = pred_score
            
            assert '#' not in pred_label, 'find # in pred_label'
            assert pred_label in caption, 'pred label not in caption'
            
            if pred_score>max_score:
                max_label = pred_label
                max_score = pred_score
            
        # record bbox
        scores.append(max_score)
        pred_phrases_set.append(token_map)
        # pred_phrases.append(max_label+ f"({str(max_score)[:4]})")
        bbox_matrix.append(box.unsqueeze(0))   
                 
        # continue
        # get phrases from all valid tokens
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append({pred_phrase: logit.max().item()})
        # pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        # scores.append(logit.max().item())
        # posmap = logit > text_threshold

    # Debug
    assert len(boxes_filt) == len(pred_phrases_set), 'pred_phrases_set length not match with boxes_filt'
    debug = True
    if debug:    
        msg = 'Detect objects\n'
        for bbox_labels in pred_phrases_set:
            # if len(bbox_labels)>1: msg += '[Ambiguous] '
            for label, conf in bbox_labels.items():
                msg += '{}:{:.3f} '.format(label, conf)
            msg += ';'
            # if len(bbox_labels)>1:
        print(msg)

    return boxes_filt, torch.Tensor(scores), pred_phrases, pred_phrases_set

def recover_bbox(boxes, image_size):
    ''' Read the box in unit of [0,1] and recover it to the original image size'''
    # size = image_pil.size
    H, W = image_size[1], image_size[0]
    for i in range(boxes.size(0)):
        boxes[i] = boxes[i] * torch.Tensor([W, H, W, H])
        boxes[i][:2] -= boxes[i][2:] / 2 # topleft
        boxes[i][2:] += boxes[i][:2] # bottomright
    return H, W

def filter_large_bbox(boxes, pred_phrases_set, image_size, threshold=0.9):
    ''' Filter boxes that are too large '''
    H, W = image_size[1], image_size[0]
    
    selection_mask = []
    for i in range(boxes.size(0)):
        box_width = (boxes[i][2] - boxes[i][0])/W
        box_height = (boxes[i][3] - boxes[i][1])/H
            
        if box_width>threshold and box_height>threshold:
            continue
        else: selection_mask.append(i)
    print('{}/{} bboxes are valid after size filtering'.format(len(selection_mask), boxes.size(0)))
    boxes = boxes[selection_mask]
    pred_phrases_set = [pred_phrases_set[idx] for idx in selection_mask]

def process_doors(boxes, pred_phrases_set, scores):
    ''' Reduce the score of door be covered by [cabinet, closet, fridge] '''

    
    for i, door_token_map in enumerate(pred_phrases_set):
        if 'door' in door_token_map:
            door_box = boxes[i]
            
            for j in range(boxes.size(0)):
                if i==j: continue
                if 'cabinet' in pred_phrases_set[j] or 'closet' in pred_phrases_set[j] or 'fridge' in pred_phrases_set[j]:
                    cabinet_box = boxes[j]
                    iou = torchvision.ops.box_iou(door_box.unsqueeze(0), cabinet_box.unsqueeze(0))
                    if iou>0.8:
                        scores[i] = scores[j]-0.1
                        break

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def show_box_tokens(box,ax,token_pairs):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    # ax.text(x0, y0, label)
    labels = ''
    for token, score in token_pairs.items():
        labels += '{}({:.3f}) '.format(token, score)
    ax.text(x0, y0, labels)

def extract_good_prompts(tags, labels):
    good_prompts = []
    label_list = []    
    # print('tags are {}'.format(tags))
    
    # print('find labels:')
    for token_map in labels:
        for label,score in token_map.items():    
            # label_name, logit = label.split('(')
            label_list.append(label)
            # print(label_name)
    # print('!!!!detections are: {}'.format(label_list))
    
    for tag in tags.split('.'):
        tag = tag.strip()
        if tag in label_list:
            good_prompts.append(tag)
    if len(good_prompts)>0:
        print('find good prompts:{}'.format(good_prompts))
    return good_prompts

def add_prev_prompts(prev_prompts, tags, invalid_augment_opensets):
    augmented_tags = tags
    if tags =='':return augmented_tags
    
    for prompt in prev_prompts:
        if tags.find(prompt) == -1 and prompt not in invalid_augment_opensets:
            augmented_tags += '. ' + prompt
    return augmented_tags    
    

def save_mask_data(output_dir,frame_name, raw_tags, tags, box_list, label_list,mask_list, save_np=False):
    value = 0  # 0 for background
    json_data = {
        'raw_tags':raw_tags,
        'tags': tags,
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for token_map, box in zip(label_list,box_list):
        value += 1
        json_data['mask'].append({
            'value': value,
            'labels': token_map,
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, '{}_label.json'.format(frame_name)), 'w') as f:
        json.dump(json_data, f)      
    
    if mask_list is None: return
    if save_np:
        mask_img = torch.zeros((len(mask_list),mask_list.shape[-2],mask_list.shape[-1])) # (K, H, W)
        for idx, mask in enumerate(mask_list):
            mask_img[idx,mask.cpu().numpy()[0] == True] = 1 #value + idx + 1
        np.save(os.path.join(output_dir, '{}_mask.npy'.format(frame_name)), mask_img.numpy())
    
    mask_instances_img = np.zeros((mask_list.shape[-2],mask_list.shape[-1]),np.uint8) # (H, W)
    for mask_id, mask in enumerate(mask_list):
        # if value>0:
        mask_instances_img[mask.cpu().numpy()[0] == True] = mask_id + 1
        
    cv2.imwrite(os.path.join(output_dir, '{}_mask.png'.format(frame_name)), mask_instances_img)
    

def read_tags_jsonfile(dir):
    tags = []
    with open(dir, 'r') as f:
        data = json.load(f)
        for item in data['mask']:
            if 'labels' in item:
                for name, score in item['labels'].items():
                    tags.append(name)
        return tags

def convert_tags(tags,valid_openset_names):
    valid_tags = ''
    valid_tag_list = []
    tag_list = []
    for tag in tags.split('.'):
        tag = tag.strip()
        if tag not in tag_list: tag_list.append(tag)

    # select valid tags
    for tag in tag_list:        
        if tag in valid_openset_names and tag not in valid_tag_list:
            valid_tags += tag + '. '
            valid_tag_list.append(tag)
            
    return valid_tags[:-2]

def convert2_baseline_tags(tags,mapper):
    '''
    Convert the open-set tags to the pre-defined close-set tags
    '''
    # valid_tags = ''
    # valid_tag_list = []
    input_tag_list = []
    
    close_set_tags = ''
    close_set_tag_list =[]
    
    for tag in tags.split(','):
        tag = tag.strip()
        if tag not in input_tag_list: input_tag_list.append(tag)

    # select valid tags
    for tag in input_tag_list:        
        if tag in mapper and mapper[tag] not in close_set_tag_list: 
            # valid_tags += tag + '. '
            # valid_tag_list.append(tag)
            
            close_set_tags += mapper[tag] + '. '
            close_set_tag_list.append(mapper[tag])
  
    return close_set_tags[:-1]

def read_scans(dir):
    with open(dir, 'r') as f:
        scans = []
        for line in f.readlines():
            scans.append(line.strip())
        f.close()
        return scans

def inference_single_image(image_path, ram_model, ground_dino_model, sam_model, args):
    print('--- processing {}---'.format(os.path.basename(image_path)))
    image_pil, image_cv = load_image(image_path)

    # run ram
    raw_image = image_pil.resize((384, 384))
    raw_image  = transform(raw_image).unsqueeze(0).to(device)
    
    t_start = time.time()
    res = inference_ram(raw_image,ram_model)
    duration_ram = time.time() - t_start
    
    # Process tags
    t0 = time.time()
    ram_tags=res[0].replace(' |', '.')
    if args.tag_mode=='proposed':
        valid_tags = convert_tags(ram_tags, mapper)
    elif args.tag_mode=='raw':
        valid_tags = ram_tags
    # elif args.tag_mode =='close_set':
    #     valid_tags = convert2_baseline_tags(ram_tags, mapper)
    else:
        raise NotImplementedError
    
    print('raw tags: ', ram_tags)
    print("valid Tags: ", valid_tags)
    # todo: abandon prompt augmentation
    # if args.prompt_augmentation:
    #     for prev_tag in reversed(prev_detected_prompts):
    #         if abs(frame_idx - prev_tag['frame'])>AUGMENT_WINDOW_SIZE: break
    #         valid_tags = add_prev_prompts(prev_tag['good_prompts'],valid_tags, invalid_augment_opensets)
    #     print('aug Tags:{}'.format(valid_tags))
    # else:
    #     print('propmts augmentation is off!')

    # run grounding dino model
    if len(valid_tags.split('.'))<1 or len(valid_tags)<2: 
        print('skip {} due to empty tags'.format(img_name))
        return None
    
    boxes_filt, scores, _, pred_phrases_set = get_grounding_output(
        ground_dino_model, image_cv, valid_tags, box_threshold, text_threshold, device=device
    )

    duration_groundDino = time.time() - t0
    if boxes_filt.size(0) == 0: return None
    H, W= recover_bbox(boxes_filt, image_pil.size)
    # process_doors(boxes_filt, pred_phrases_set, scores)

    # Filter overlapped bbox using NMS
    boxes_filt = boxes_filt.cpu()
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases_set = [pred_phrases_set[idx] for idx in nms_idx]
    if boxes_filt.size(0) < 1: return None

    filter_large_bbox(boxes_filt, pred_phrases_set, image_pil.size, threshold=0.9)

    # Update good prompts in adjacent frames
    good_prompts = [prompt for prompt in extract_good_prompts(valid_tags, pred_phrases_set) if prompt in ram_tags]
    prev_detected_prompts.append(
        {'frame':frame_idx,"good_prompts":good_prompts}
    )
    
    # Run SAM
    image_cv = cv2.imread(image_path)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)              
    if sam_model is not None:
        t0 = time.time()
        sam_model.set_image(image_cv)
        
        transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(device)
        if transformed_boxes.size(0) == 0:
            return None            
        masks, _, _ = sam_model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
        duration_sam = time.time() - t0
    else:
        masks = None
        duration_sam = 0

    t_total = time.time() - t_start
    print('RAM:{:.2f}s, Grounding:{:.2f}s, SAM:{:.2f}s, total:{:.2f}s'.format(
        duration_ram, duration_groundDino, duration_sam, t_total))
    ram_time.append(duration_ram)
    dino_time.append(duration_groundDino)
    total_time.append(t_total)
    sam_time.append(duration_sam)

    # draw output image
    if args.save_viz:
        viz_dir = os.path.join(scene_dir, 'viz')
        if os.path.exists(viz_dir)==False:
            os.makedirs(viz_dir)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image_cv)                
        for box, token_map in zip(boxes_filt, pred_phrases_set):
            show_box_tokens(box.numpy(), plt.gca(), token_map)
            
        if args.run_sam:
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                                
        plt.title('RAM-tags:' + ram_tags + '\n' + 'filter tags:'+valid_tags+'\n') # + 'RAM-tags_chineseing: ' + tags_chinese + '\n')
        plt.axis('off')
        plt.savefig(os.path.join(viz_dir, "{}_det.jpg".format(img_name)), 
                    bbox_inches="tight", dpi=300, pad_inches=0.0)

    save_mask_data(output_dir, img_name, ram_tags, valid_tags, boxes_filt, pred_phrases_set, masks)
    
    return boxes_filt, pred_phrases_set, masks

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, help="path to checkpoint file")
    parser.add_argument("--run_sam", action="store_true", help="run sam")
    parser.add_argument("--dataroot", type=str, required=True, help="path to data root")
    parser.add_argument("--split_file", type=str, help="name of the split file")
    parser.add_argument("--split", type=str, help="split for text prompt", default="scans")
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )

    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
    parser.add_argument("--frame_gap", type=int, default=1, help="skip frames")
    parser.add_argument("--tag_mode",type=str,default='proposed', help="raw, close_set, proposed")
    parser.add_argument("--prompt_augmentation",action='store_true', help="enable prompt augmentation")
    
    parser.add_argument("--box_threshold", type=float, default=0.35, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="iou threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--data_format",type=str, default='scannet', help="data format")
    parser.add_argument("--save_viz",action='store_true', help="save visualization")
    parser.add_argument("--categories",type=str, default='categories_new.json', help="categories file")
    
    args = parser.parse_args()
    args.dataset = args.data_format
    _, mapper, _ = check_category.read_category(args.categories)
    
    # cfg
    config_file = args.config  # change the path of the model config file
    ram_checkpoint = args.ram_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    dataroot = args.dataroot
    split_file = args.split_file
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    openai_key = args.openai_key
    split = args.split
    openai_proxy = args.openai_proxy
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device
    AUGMENT_WINDOW_SIZE = 5
    

    if args.dataset == 'scannet': # compatible with slabim
        RGB_FOLDER_NAME = 'color'
    elif args.dataset =='fusionportable':
        RGB_FOLDER_NAME = 'color'
    elif args.dataset =='rio':
        RGB_FOLDER_NAME = 'color'
    elif args.dataset == 'tum' or args.dataset=='realsense':
        RGB_FOLDER_NAME = 'rgb'
    elif args.dataset =='scenenn':
        RGB_FOLDER_NAME ='image'
    elif args.dataset =='matterport':
        RGB_FOLDER_NAME = 'color'
        split = 'v1/scans'
    else:
        raise NotImplementedError
    
    invalid_augment_opensets = ['floor','carpet','door','glass door','window','fridge','refrigerator']
    
    # load ram model
    ram_model = ram(pretrained=ram_checkpoint,
                                        image_size=384,
                                        vit='swin_l')
    ram_model.eval()
    ram_model = ram_model.to(device)
    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = TS.Compose([TS.Resize((384, 384)),
                            TS.ToTensor(), normalize])
    
    # load grounding-dino
    ground_dino_model = load_model(config_file, grounded_checkpoint, device=device)
    
    # initialize SAM
    if args.run_sam:
        assert sam_checkpoint is not None, 'SAM checkpoint directory is required to run SAM'
        sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    else: sam_model = None

    # Dataset
    if split_file==None:
        scans = os.listdir(dataroot)
    else:
        scans = read_scans(os.path.join(dataroot,'splits', split_file))
    print('find {} scans'.format(len(scans)))
    
    ram_time = []
    dino_time = []
    sam_time = []
    total_time = []
    frame_numer = 0
    
    for scan in scans:
        if args.dataset=='rio':
            scene_dir = os.path.join(dataroot, scan)
        else:    
           scene_dir = os.path.join(dataroot,split, scan)
        input_folder = os.path.join(scene_dir, RGB_FOLDER_NAME)
        imglist = glob.glob(os.path.join(input_folder, '*.png'))
        imglist += glob.glob(os.path.join(input_folder, '*.jpg'))
        imglist = sorted(imglist)
        
        if args.prompt_augmentation:
            output_dir = os.path.join(scene_dir, 'prediction_vaug5')
        else:
            output_dir = os.path.join(scene_dir, 'prediction_no_augment')

        if os.path.exists(output_dir)==False:
            os.makedirs(output_dir)
            
        print('********** Run {}, {} imgs **********'.format(scan, len(imglist)))
        prev_detected_prompts = []
        
        # load image
        for i, image_path in enumerate(imglist):
            img_name = image_path.split('/')[-1][:-4]
            if args.dataset =='scannet' or args.dataset=='realsense' or args.dataset=='rio' or args.dataset=='bim':
                frame_idx = int(img_name.split('-')[-1][:6])
            elif args.dataset =='fusionportable':
                frame_idx = int(img_name.split('.')[0])
            elif args.dataset == 'scenenn':
                frame_idx = int(img_name[-5:])
            elif args.dataset =='matterport' or args.dataset=="tum":
                frame_idx = i
            else:
                raise NotImplementedError
            
            if frame_idx % args.frame_gap !=0:
                continue
            if os.path.exists(os.path.join(output_dir, '{}_label.json'.format(img_name))):
                continue

            ret = inference_single_image(image_path, ram_model, ground_dino_model, sam_model, args)
            
            if isinstance(ret, tuple):
                boxes, pred_phrases_set, masks = ret
                frame_numer +=1
        
        print('********* finished {} **********'.format(scan))
        # break
       
    # Summarize time
    ram_time = np.array(ram_time)
    dino_time = np.array(dino_time)
    sam_time = np.array(sam_time)
    total_time = np.array(total_time)
    out_time_file = os.path.join(dataroot, 'output','autolabel_{}.log'.format(args.split_file.split('.')[0]))
    with open(out_time_file,'w') as f:
        f.write('{} scans, frame gap {}, {} frames \n'.format(len(scans),args.frame_gap, frame_numer))
        f.write('enable multiple token for each detection. identity boxes are merged. \n')
        f.write('filter out boxes that are too large \n')
        f.write('filter tag: {} \n'.format(args.tag_mode))
        f.write('average ram time:{:.3f} ms, dinotime:{:.3f} ms, samtime:{:.3f} ms, totaltime:{:.3f} ms\n'.format(
            1000*np.mean(ram_time), 1000*np.mean(dino_time), 1000*np.mean(sam_time), 1000*np.mean(total_time)))
        f.write('total time:{:.2f} s'.format(np.sum(total_time)))
        f.close()
