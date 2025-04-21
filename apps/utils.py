import yaml
import streamlit_authenticator as stauth
import random
import string
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import base64
import numpy as np
import cv2
import json
import requests
import io
from settings import LABELS_PATH, ACCOUNTS_PATH, ABOUTS_CONTENT_PATH, HOME_CONTENT_PATH, MODELS_PATH, COLABS_PATH, BLOGS_PATH, PREDICT_URL, MODEL_NAME_1, MODEL_NAME_2, PREDICT1_URL, PREDICT2_URL

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    labels_data = json.load(file)

with open(ACCOUNTS_PATH, "r", encoding="utf-8") as file:
    accounts_data = yaml.safe_load(file)

with open(MODELS_PATH, "r", encoding="utf-8") as file:
    models_data = json.load(file)

with open(COLABS_PATH, "r", encoding="utf-8") as file:
    colabs_data = json.load(file)

with open(BLOGS_PATH, "r", encoding="utf-8") as file:
    blogs_data = json.load(file)
    
with open(ABOUTS_CONTENT_PATH, "r", encoding="utf-8") as file:
    abouts_data = json.load(file)

with open(HOME_CONTENT_PATH, "r", encoding="utf-8") as file:
    home_data = json.load(file)

def add_account(username, name, role, org_password):
    new_user = {
        "name": name,
        "role": role,
        "org_passworh": org_password,
        "password": "",
    }

    if username not in accounts_data["credentials"]["usernames"]:
        accounts_data["credentials"]["usernames"][username] = new_user

        update_yaml()
        return True
    else:
        return False

def update_yaml():
    credentials = accounts_data['credentials']['usernames']
    usernames = list(credentials.keys())
    names = [info['name'] for info in credentials.values()]
    roles = [info['role'] for info in credentials.values()]
    org_passwords = [info['org_passworh'] for info in credentials.values()]

    cookie = accounts_data['cookie']
    accounts_data['cookie']['key'] = generate_random_string()

    passwords = stauth.Hasher(org_passwords).generate()
    for username, new_password in zip(usernames, passwords):
        credentials[username]['password'] = new_password

    with open(ACCOUNTS_PATH, 'w', encoding='utf-8') as file:
        yaml.dump(accounts_data, file, default_flow_style=False, allow_unicode=True)

def update_account(username, name, role, org_password):
    updated_user = {
        "name": name,
        "role": role,
        "org_passworh": org_password,
        "password": "",
    }

    if username in accounts_data["credentials"]["usernames"] or username.lower() == "none":
        accounts_data["credentials"]["usernames"][username] = updated_user

        update_yaml()
        return True
    else:
        return False

def load_accounts():
    credentials = accounts_data['credentials']['usernames']
    usernames = list(credentials.keys())
    names = [info['name'] for info in credentials.values()]
    roles = [info['role'] for info in credentials.values()]
    org_passwords = [info['org_passworh'] for info in credentials.values()]
    passwords = [info['password'] for info in credentials.values()]

    cookie = accounts_data['cookie']
    cookie_name = cookie['name']
    cookie_key = cookie['key']
    cookie_value = cookie['value']
    cookie_expiry_days = cookie['expiry_days']

    return usernames, names, roles, passwords, org_passwords, cookie_name, cookie_key, cookie_value, cookie_expiry_days

def generate_random_string(length=10): 
    all_characters = string.ascii_letters + string.digits + string.punctuation 
    random_string = ''.join(random.choice(all_characters) for _ in range(length)) 
    return random_string

def preprocess(image_link):
    base64_image = base64.b64decode(image_link)
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])
    buf = io.BytesIO(base64_image)
    image = Image.open(buf)
    image = image.convert("RGB")
    image = transform(image)
    return image

def get_model(model_name:str, num_class: int, device='cpu'):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_class)
        
    if model_name == 'resnet101':
        model = models.resnet101(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_class)

    if model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_class)
    
    if model_name == 'efficientnet_v2_m':
        model = models.efficientnet_v2_m(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_class)
        
    if model_name == 'efficientnet_v2_l':
        model = models.efficientnet_v2_l(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_class)
    
    if model_name == 'inception_v3':
        model = models.inception_v3(pretrained=False)
        model.aux_logits = False
        model.fc = nn.Linear(model.fc.in_features, num_class)
   
    if model_name == 'model_2':
        model = models.efficientnet_v2_s(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_class)
        
    # if model_name == 'model_2_efficientnet_v2_s':
    #     model = models.efficientnet_v2_s(pretrained=False)
    #     model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_class)
    # if model_name == 'model_2_efficientnet_v2_s1':
    #     model = models.efficientnet_v2_s(pretrained=False)
    #     model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_class) 
    # if model_name == 'model_2_best_efficientnet_v2_s1':
    #     model = models.efficientnet_v2_s(pretrained=False)
    #     model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_class)  
    
    model.load_state_dict(torch.load(f'{models_data[model_name]}', map_location=torch.device('cpu')), strict=False)
    model.eval()
    
    return model

def get_disease_info(disease_name, type): 
    for key, value in labels_data[str(type)].items(): 
        if value["Disease"] == disease_name: 
            return value
        
def get_description(result, type):
    if type:
        detected_disease_vie = None 
        for disease in labels_data[str(type)].values(): 
            if disease["Disease"] == result["Disease"]: 
                detected_disease_vie = disease["Vie"]
                break
        detail_strings = [] 
        for key, value in result["Detail"].items(): 
            for disease in labels_data[str(type)].values(): 
                if disease["Disease"] == key: 
                    detail_strings.append(f"{disease['Vie']}: {round(value*100, 2)}")

        final_string = f"Ảnh da của bạn có nguy cơ mắc bệnh {detected_disease_vie}."
        return final_string, detail_strings
    else:
        detected_disease_vie = None 
        for disease in labels_data[str(type)].values(): 
            if disease["Disease"] == result["Disease"]: 
                detected_disease_vie = disease["Vie"] 
                break # Generate the detail string 
        detail_strings = [] 
        for key, value in result["Detail"].items(): 
            for disease in labels_data[str(type)].values(): 
                if disease["Disease"] == key: 
                    detail_strings.append(f"{disease['Vie']}: {round(value*100, 2)}")

        final_string = ""
        if result["Disease"] == "Khong_benh": 
            final_string = "Ảnh da của bạn không có bệnh." 
        if result["Disease"] == "Ung_thu": 
            final_string = "Ảnh da của bạn có nguy cơ mắc bệnh ung thư da." 
        else: 
            final_string = "Ảnh da của bạn có nguy cơ mắc bệnh da khác."
        return final_string, detail_strings
    
def get_sub_dict(main_dict, sub_keys):
    return {key: main_dict[key] for key in sub_keys if key in main_dict}

def get_sub_list(main_list, indices):
    return [main_list[i] for i in indices if i < len(main_list)]

def m_infer(image_link: str,
          model_name: str,
          num_class: int = 3
         ):
    image = preprocess(image_link)
    image = torch.tensor(image, dtype=torch.float).unsqueeze(dim=0)
    model1 = get_model(MODEL_NAME_1, num_class)
    with torch.no_grad():
        output = model1(image)
        predicted = torch.nn.functional.softmax(output[0])
    if predicted.argmax().item() == 1:
        model2 = get_model(MODEL_NAME_2, num_class)
        with torch.no_grad():
            output = model2(image)
            predicted = torch.nn.functional.softmax(output[0])
        result = {
                "Disease": labels_data['1'][str(predicted.argmax().item())]['Disease'],
                "Score": predicted.max().item(),
                "Detail":{
                    "Day": predicted[0].item(),
                    "Vay": predicted[1].item(),
                    "Hac_to": predicted[2].item()
                }
            }
        # return JSONResponse(result)
        return result
    else:
        result = {
            "Disease": labels_data['0'][str(predicted.argmax().item())]['Disease'],
            "Score": predicted.max().item(),
            "Detail":{
                "Khong_benh": predicted[0].item(),
                "Ung_thu": predicted[1].item(),
                "Benh_khac": predicted[2].item()
            }
        }
        # return JSONResponse(result)
        return result
    
def infer_1(image_link: str,
          model_name: str,
          num_class: int = 3
         ):
    image = preprocess(image_link)
    image = torch.tensor(image, dtype=torch.float).unsqueeze(dim=0)
    print(image.max())
    model1 = get_model(MODEL_NAME_1, num_class)
    with torch.no_grad():
        output = model1(image)
        predicted = torch.nn.functional.softmax(output[0])
    result = {
        "Disease": labels_data['0'][str(predicted.argmax().item())]['Disease'],
        "Score": predicted.max().item(),
        "Detail":{
            "Khong_benh": predicted[0].item(),
            "Ung_thu": predicted[1].item(),
            "Benh_khac": predicted[2].item()
        }
    }
    return result

def infer_2(image_link: str,
          model_name: str,
          num_class: int = 3
         ):
    image = preprocess(image_link)
    image = torch.tensor(image, dtype=torch.float).unsqueeze(dim=0)
    # model2 = get_model(f"{MODEL_NAME_2}_{model_name}", num_class)
    model2 = get_model(MODEL_NAME_2, num_class)
    with torch.no_grad():
        output = model2(image)
        predicted = torch.nn.functional.softmax(output[0])
    result = {
            "Disease": labels_data['1'][str(predicted.argmax().item())]['Disease'],
            "Score": predicted.max().item(),
            "Detail":{
                "Day": predicted[0].item(),
                "Vay": predicted[1].item(),
                "Hac_to": predicted[2].item()
            }
        }
    return result
    
def base64_encode(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

def send_request(image, model_name):
    base64_image = base64_encode(image)
    
    # Payload
    payload = {
        "image": base64_image.decode("utf-8"),
        "model_name": model_name
    }
    # Send POST request
    response = requests.post(PREDICT_URL, json=payload)

    # Print the result
    return response.json()

def send_request_1(image, model_name):
    base64_image = base64_encode(image)
    
    # Payload
    payload = {
        "image": base64_image.decode("utf-8"),
        "model_name": model_name
    }
    # Send POST request
    response = requests.post(PREDICT1_URL, json=payload)

    # Print the result
    return response.json()

def send_request_2(image, model_name):
    base64_image = base64_encode(image)
    
    # Payload
    payload = {
        "image": base64_image.decode("utf-8"),
        "model_name": model_name
    }
    # Send POST request
    response = requests.post(PREDICT2_URL, json=payload)

    # Print the result
    return response.json()

if __name__ == "__main__":
    # add_account("user2", "Người dùng 2", "User", "123")
    # update_account("user2", "Người dùng 2", "User", "137")
    add_account("user3", "Người dùng 3", "User", "137")