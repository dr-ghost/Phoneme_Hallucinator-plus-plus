from wavlm import *
import torch
from warnings import filterwarnings
filterwarnings("ignore")

if __name__ == "__main__":
    wav_lm = wavlm_model(
        model_name="wavlm_large",
        checkpoint_path="models/wavlm/checkpoints/wavlm_large",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )