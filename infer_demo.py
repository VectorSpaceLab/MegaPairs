# from transformers import AutoModel
# from PIL import Image
# import torch

# MODEL_NAME = "JUNJIE99/MMRet-base"
# MMRet = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

# MMRet.set_processor(MODEL_NAME)
# with torch.no_grad():
#     queries_1 = MMRet.encode(
#         images=["./assets/cir_query.png", "./assets/cir_query.png"], 
#         text=["Make the background dark, as if the camera has taken the photo at night", "A horse is pulling this cart."]
#     )

#     queries_2 = MMRet.encode(
#         text=["A carriage, photographed at night.", "A horse is pulling a cart."]
#     )

#     candidates = MMRet.encode(
#         images=["./assets/cir_candi_1.png", "./assets/cir_candi_2.png"]
#     )
    
#     scores_1 = queries_1@candidates.T
#     scores_2 = queries_2@candidates.T

# print(scores_1)
# print(scores_2)








import torch
from transformers import AutoModel

MODEL_NAME = "JUNJIE99/MMRet-base" # or "JUNJIE99/MMRet-large"

model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True) # You must set trust_remote_code=True
model.set_processor(MODEL_NAME)
model.eval()

with torch.no_grad():
    query = model.encode(
        images = "./assets/cir_query.png", 
        text = "Make the background dark, as if the camera has taken the photo at night"
    )

    candidates = model.encode(
        images = ["./assets/cir_candi_1.png", "./assets/cir_candi_2.png"]
    )
    
    scores = query @ candidates.T
print(scores)