import json


f_config = "Fooocus/config.txt"
f_api_config = "Fooocus-API/config.txt"

with open(f_config, "r") as file:
    data = json.load(file)

data["default_model"] = "juggernautXL_v9Rundiffusion.safetensors"

with open(f_config, "w") as file:
    json.dump(data, file, indent=4)

with open(f_api_config, "w") as file:
    json.dump(data, file, indent=4)
