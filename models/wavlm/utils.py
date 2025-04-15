import torch
# import fairseq
from packaging import version
import torch.nn.functional as F
# from fairseq import tasks
# from fairseq.checkpoint_utils import load_checkpoint_to_cpu
# from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from omegaconf import OmegaConf
from s3prl.upstream.interfaces import UpstreamBase
from torch.nn.utils.rnn import pad_sequence

import os
import gdown

from .ecapa_tdnn import ECAPA_TDNN_SMALL

# def load_model(filepath):
#     state = torch.load(filepath, map_location=lambda storage, loc: storage)
#     # state = load_checkpoint_to_cpu(filepath)
#     state["cfg"] = OmegaConf.create(state["cfg"])

#     if "args" in state and state["args"] is not None:
#         cfg = convert_namespace_to_omegaconf(state["args"])
#     elif "cfg" in state and state["cfg"] is not None:
#         cfg = state["cfg"]
#     else:
#         raise RuntimeError(
#             f"Neither args nor cfg exist in state keys = {state.keys()}"
#             )

#     task = tasks.setup_task(cfg.task)
#     if "task_state" in state:
#         task.load_state_dict(state["task_state"])

#     model = task.build_model(cfg.model)

#     return model, cfg, task


###################
# UPSTREAM EXPERT #
###################
# class UpstreamExpert(UpstreamBase):
#     def __init__(self, ckpt, **kwargs):
#         super().__init__(**kwargs)
#         assert version.parse(fairseq.__version__) > version.parse(
#             "0.10.2"
#         ), "Please install the fairseq master branch."

#         model, cfg, task = load_model(ckpt)
#         self.model = model
#         self.task = task

#         if len(self.hooks) == 0:
#             module_name = "self.model.encoder.layers"
#             for module_id in range(len(eval(module_name))):
#                 self.add_hook(
#                     f"{module_name}[{module_id}]",
#                     lambda input, output: input[0].transpose(0, 1),
#                 )
#             self.add_hook("self.model.encoder", lambda input, output: output[0])

#     def forward(self, wavs):
#         if self.task.cfg.normalize:
#             wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

#         device = wavs[0].device
#         wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
#         wav_padding_mask = ~torch.lt(
#             torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
#             wav_lengths.unsqueeze(1),
#         )
#         padded_wav = pad_sequence(wavs, batch_first=True)

#         features, feat_padding_mask = self.model.extract_features(
#             padded_wav,
#             padding_mask=wav_padding_mask,
#             mask=None,
#         )
#         return {
#             "default": features,
#         }

def wavlm_model(model_name: str, checkpoint_path: str, device: torch.DeviceObjType) -> torch.nn.Module:
    assert model_name in ["wavlm_base_plus", "wavlm_large"]
    
    feat_dim = 768 if model_name == "wavlm_base_plus" else 1024
    
    model = ECAPA_TDNN_SMALL(feat_dim=feat_dim, feat_type=model_name, config_path=None)
    
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
        
    model.to(device)
    
    return model


def download_cpkt(idx : int):
    """
    Download the wavlm_base_plus checkpoint from Google Drive.
    https://drive.usercontent.google.com/download?id=1OMdkp5Vv8A9WnHSTSoDwA8hxQWsEAu85&export=download&authuser=0&confirm=t&uuid=a64ed218-1916-4492-8fd8-06d8fa3dbcc5&at=AEz70l5_VmQCkAgiR4OoSRUPC7In:1743420047651    
    https://drive.usercontent.google.com/download?id=12-cB34qCTvByWT-QtOcZaqwwO21FLSqU&export=download&authuser=1&confirm=t&uuid=2b8f46e3-c94f-4070-9d56-4f1295bf6192&at=APcmpoxJsv-IqLmbuRAvsjnsrJZe%3A1744376524725"""
    file_ids = [
        "1OMdkp5Vv8A9WnHSTSoDwA8hxQWsEAu85",  # Corresponds to wavlm_base_plus
        "12-cB34qCTvByWT-QtOcZaqwwO21FLSqU"   # Corresponds to wavlm_large
    ]
    
    # Desired output file names
    file_names = ["wavlm_base._pluspt", "wavlm_large"]
    
    # Google Drive direct download URL base format
    base_url = "https://drive.google.com/uc?export=download&id="
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    download_dir = os.path.join(current_dir, "checkpoints")
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    for file_id, file_name in zip(file_ids, file_names):
        url = base_url + file_id
        output_path = os.path.join(download_dir, file_name)
        print(f"Downloading {file_name} from {url}...")
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded {file_name} to {output_path}")

if __name__ == "__main__":
    download_cpkt(0)