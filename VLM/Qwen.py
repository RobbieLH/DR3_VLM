import torch
import os
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# 设置环境变量，让系统只看到指定的 GPU 卡
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,3"


# from modelscope import snapshot_download
from transformers import Qwen2_5_VLForConditionalGeneration,  AutoTokenizer,AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

class QwenVL:
    def __init__(self, size="3B"):
        assert size in {"3B", "7B"}, "wrong size!"
        self.size = size
        self.model_dir = "Qwen/Qwen2.5-VL-{}-Instruct".format(self.size) #snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct")
        self.device = "auto"
        self.envVLM_count = 0
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_dir,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2.5-VL-7B-Instruct",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )

        # default processor
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.processor = AutoProcessor.from_pretrained(self.model_dir,max_pixels = 1280*28*28)

        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    def __call__(self, image):
       
        statement = "请描述图片内容,并预测图片内容"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        # "image": "E:/AI_project/Qwen2.5-VL/01-21-2025_09_58_PM.png",
                        #"image": "E:/my_data/temp_img/20250222200343.jpg"
                    },
                    {
                        "type": "text",
                        "text": statement
                    },
                ],
            }
        ]

        # Preparation for inference
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
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        self.envVLM_count += 1
        if self.envVLM_count == 1:
            print(output_text)
            data_type = type(output_text)

            # 打印结果:output_text 的数据类型是: <class 'list'>
            print(f"Qwen/Qwen2.5-VL-self.size是: {self.size}")
            print(f"output_text 的数据类型是: {data_type}")
            print(f"output_text 的数据类型是: {np.array(output_text).shape}")
        return output_text[0]
