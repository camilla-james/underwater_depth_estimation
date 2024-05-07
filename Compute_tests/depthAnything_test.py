import matplotlib.pyplot as plt
# external imports
import transformers
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import time 
import numpy as np
from PIL import Image
import requests
import datasets
from datasets import load_dataset
from tqdm import tqdm

def main():

    # import the dataset -> stream it so it does not take too long
    train_dataset = load_dataset("sayakpaul/nyu_depth_v2", split="train", 
                                 streaming = True, trust_remote_code=True)
    
    # load the dataset into a dataloader
    dataset = train_dataset.with_format("torch")
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size = 16)

    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    model.to(device)
    # prepare image for the model
    for i, batch in enumerate(tqdm(dataloader)):
    	t0 = time.time()
        image = batch.get('image')
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size= [image.size()[1], image.size()[2]],#image.size(),
            mode="bicubic",
            align_corners=False,
        )

        # visualize the prediction
        # output = prediction.squeeze().cpu().numpy()
        # print(output)
        # formatted = (output * 255 / np.max(output)).astype("uint8")
        # depth = Image.fromarray(formatted)
        # plt.imshow(depth)
        print(f"The time taken is:"{time.time()-t0})


if __name__ == "__main__":
    main()
