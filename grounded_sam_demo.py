import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    print('-------get grounding output----------')

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
    print('{}/{} predicted objects'.format(logits_filt.shape[0], logits.shape[0]))

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption) # {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
    print('input caption: {}'.format(caption))
    # print('token ids: {}'.format(tokenized['input_ids']))
    # print('{} text tokens'.format(len(tokenized['input_ids'])))
    # for tkid in tokenized['input_ids']:
    #     print(tokenlizer.decode(tkid))
    from GroundingDINO.groundingdino.util.utils import get_grouped_tokens
    
    grouped_token_ids, posmap_to_prompt_id = get_grouped_tokens(tokenized['input_ids'], tokenlizer)
    assert len(grouped_token_ids)+1 == len(caption.split('.'))
    
    # build pred
    pred_phrases = []
    scores = []
    
    for logit, box in zip(logits_filt, boxes_filt):
        # box: [x0, y0, x1, y1]
        # logit: 256
        posmap     = logit > text_threshold
        prompt_ids = posmap_to_prompt_id[posmap.nonzero(as_tuple=True)[0]]
        prompt_ids = torch.unique(prompt_ids)
        assert prompt_ids.min()>=0 and prompt_ids.max()<len(grouped_token_ids)
        
        pred_phrase = ''
        
        for prompt_id in prompt_ids:
            prompt_posmap = grouped_token_ids[prompt_id]
            prompt_tokens = [tokenized['input_ids'][i] for i in prompt_posmap]
            pred_label = tokenlizer.decode(prompt_tokens)
            pred_score = logit[prompt_posmap].max()
            if logit[prompt_posmap].min()>text_threshold:
                pred_phrase += pred_label + f"({str(pred_score.item())[:4]})"+ ' '
            # print('^_^ {}:{:.3f}'.format(pred_label,pred_score))
        
        
        # print(pred_phrase)
        scores.append(logit.max().item())
        pred_phrases.append(pred_phrase)
        
        continue
        pred_phrase = get_phrases_from_posmap(posmap, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        
        # print('{}:{:.3f}'.format(pred_phrase,logit.max().item()))
        continue
        all_posmap = logit>1e-6

        if posmap.sum()>1 and len(pred_phrase.split(' '))>1:
            print('{} has {}/{} valid text token, token ids: {}'.format(
                pred_phrase,posmap.sum(),all_posmap.sum(),posmap.nonzero(as_tuple=True)[0].tolist()))
            for i in posmap.nonzero(as_tuple=True)[0].tolist():
                print('{}:{:.3f}'.format(tokenlizer.decode(tokenized['input_ids'][i]),logit[i]))
                
        # print('all tokens')
        # for j in all_posmap.nonzero(as_tuple=True)[0].tolist():
        #     print('{}:{:.3f}'.format(tokenlizer.decode(tokenized['input_ids'][j]),logit[j]))

    print('receive pred output')
    return torch.Tensor(scores), boxes_filt, pred_phrases

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
    ax.text(x0, y0, label, bbox=dict(facecolor='green', alpha=0.5), fontsize=20, color='white')


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig('{}_mask.jpg'.format(output_dir), bbox_inches="tight", dpi=300, pad_inches=0.0)
    return None
    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit.strip()[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--frame_gap", type=int, default=1, help="skip frames")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--viz_mask", action="store_true", help="visualize mask")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    input_folder = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # make dir
    import glob, time
    os.makedirs(output_dir, exist_ok=True)
    if '.jpg' or '.png' in input_folder:
        imglist = [input_folder]
    else:
        imglist = glob.glob(os.path.join(input_folder, '*.jpg'))
       
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))


    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
            
    for id, image_path in enumerate(imglist):
        if id%args.frame_gap != 0:
            continue
        start_t = time.time()
        
        img_name = os.path.basename(image_path).split('.')[0]
        # load image
        image_pil, image = load_image(image_path)

        # run grounding dino model
        scores, boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )
        t_dino = time.time() - start_t
        # continue
    
        # SAM
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        # boxes_filt = boxes_filt.cpu()
        
        # Filter overlapped bbox using NMS
        import torchvision
        iou_threshold = 0.5
        boxes_filt = boxes_filt.cpu()
        tmp_count_bbox = boxes_filt.shape[0]
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        # pred_phrases_unaligned = [pred_phrases_unaligned[idx] for idx in nms_idx]
        # pred_phrases_set = [pred_phrases_set[idx] for idx in nms_idx]
        print('After NMS, {}/{} bbox are valid'.format(len(nms_idx), tmp_count_bbox))
        
        
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
        t_sam = time.time() - start_t - t_dino
        
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        if args.viz_mask:
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                
        for box, label in zip(boxes_filt, pred_phrases):
            concat_label = '{} {}'.format(label,pred_phrases[-1])
            show_box(box.numpy(), plt.gca(), label)
            print(label)
            # break

        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, "{}.jpg".format(img_name)), 
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )

        duration = time.time() - start_t
        print("dino: {:.2f}s, sam: {:.2f}s, total: {:.2f}s".format(t_dino,t_sam,duration))
        
        # save_mask_data(os.path.join(output_dir,img_name), masks, boxes_filt, pred_phrases)
        break

