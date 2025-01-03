import requests
import os


with open(os.path.join("densepose", "model_final_162be9.pkl"), "wb") as file:
    file.write(requests.get("https://huggingface.co/zhengchong/CatVTON/resolve/main/DensePose/model_final_162be9.pkl").content)

with open(os.path.join("densepose", "densepose_rcnn_R_50_FPN_s1x.yaml"), "wb") as file:
    file.write(requests.get("https://huggingface.co/zhengchong/CatVTON/resolve/main/DensePose/densepose_rcnn_R_50_FPN_s1x.yaml").content)

with open(os.path.join("densepose", "Base-DensePose-RCNN-FPN.yaml"), "wb") as file:
    file.write(requests.get("https://huggingface.co/zhengchong/CatVTON/resolve/main/DensePose/Base-DensePose-RCNN-FPN.yaml").content)

with open(os.path.join("SCHP", "exp-schp-201908261155-lip.pth"), "wb") as file:
    file.write(requests.get("https://huggingface.co/zhengchong/CatVTON/resolve/main/SCHP/exp-schp-201908261155-lip.pth").content)

with open(os.path.join("SCHP", "exp-schp-201908301523-atr.pth"), "wb") as file:
    file.write(requests.get("https://huggingface.co/zhengchong/CatVTON/resolve/main/SCHP/exp-schp-201908301523-atr.pth").content)

print("Download completed!")