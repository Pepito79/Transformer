from pathlib import Path
import os 

def get_config():
    return {
        "batch_size":5,
        "num_epochs":20,
        "lr":0.00001,
        "seq_len":350 ,  
        "d_model":512,
        "lang_src": "fr",
        "lang_tgt": "it",
        "model_folder": "/home/pepito/Documents/Python/ML/weights",
        "model_basename":"tmodel_",
        "preload": None,
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name":"runs/tmodel"
    }
    

def get_weights_file_path(config, epoch: str):
    model_folder = Path(config["model_folder"])
    model_folder.mkdir(parents=True, exist_ok=True)  # crée le dossier ./weights s'il n'existe pas
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return model_folder / model_filename
    