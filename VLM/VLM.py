import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers.generation import GenerationConfig

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode

# cache_dir = "/root/autodl-tmp/.cache/huggingface/hub"


class QwenVLChat:
    def __init__(self) -> None:
        model_name = "Qwen/Qwen-VL-Chat"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        device = "cuda"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)

    def __call__(self, image):
        text='describe this image'
        query = self.tokenizer.from_list_format([
            {'image': image}, 
            {'text': text},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        # print(response)
        return response
        

class QwenVL:
    def __init__(self, size="2B"):
        assert size in {"2B", "7B"}, "wrong size!"
        model_name = "Qwen/Qwen2-VL-{}-Instruct".format(size)
        self.device = "cuda:1"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name, torch_dtype="auto", device_map=self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, image):
        # statement = "The picture is the scene that the car sees when driving on the highway, " \
        #                 "please act as an autonomous driving agent, so that the car can drive safely." \
        #                 "The throttle range of the car is -1~0.5, the throttle ≤0 indicates parking, and the throttle >0 indicates different degrees of acceleration." \
        #                 "The steering range is -1 to 1. Steering >0 indicates a turn to the right, steering <0 indicates a turn to the left, and steering =0 indicates a straight line." \
        #                 "Please reply in the following format according to the picture description: \nDescribe:\nthrottle:\nsteer:"
        # statement = "Please describe the image."
        statement = "The image depicts a scene seen when a ego-vehicle is traveling on a highway in an autonomous driving environment."\
                        " Describe the position of the ego-vehicle in the diagram and its position in relation to other vehicles based on the image."
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text", 
                        "text": statement
                    },
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]


class InternVL:
    def __init__(self):
        path = "OpenGVLab/InternVL2-2B"
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=1024, do_sample=True)

    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def load_image(self, image, input_size=448, max_num=12):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        else:
            image = image.convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def __call__(self, image):
        pixel_values = self.load_image(image, max_num=12).to(torch.bfloat16).cuda()
        # statement = "The picture is the scene that the car sees when driving on the highway, " \
        #                 "please act as an autonomous driving agent, so that the car can drive safely." \
        #                 "The throttle range of the car is -1~0.5, the throttle ≤0 indicates parking, and the throttle >0 indicates different degrees of acceleration." \
        #                 "The steering range is -1 to 1. Steering >0 indicates a turn to the right, steering <0 indicates a turn to the left, and steering =0 indicates a straight line." \
        #                 "Please reply in the following format according to the picture description: \nDescribe:\nthrottle:\nsteer:"
        # statement = "Please describe the image."
        statement = "The image depicts a scene seen when a ego-vehicle is traveling on a highway in an autonomous driving environment."\
                        " Describe the position of the ego-vehicle in the diagram and its position in relation to other vehicles based on the image."
        question = '<image>\n' + statement
        response = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config)
        return response