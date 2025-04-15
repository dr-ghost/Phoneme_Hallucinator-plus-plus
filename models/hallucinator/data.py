import os
import requests
import zipfile
import soundfile as sf
from torch.utils.data import Dataset

def download_vctk(url, download_dir, extract=True):
    """
    Download and (optionally) extract the VCTK corpus ZIP file.
    
    Args:
        url (str): URL pointing to the VCTK ZIP file.
        download_dir (str): Directory where the dataset will be stored.
        extract (bool): Whether to extract the zip file after downloading.
    
    Returns:
        str: Path to the extracted dataset directory.
    """
    os.makedirs(download_dir, exist_ok=True)
    zip_path = os.path.join(download_dir, 'VCTK-Corpus-0.92.zip')
    
    if not os.path.exists(zip_path):
        print("Downloading VCTK corpus...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print("Download finished.")
    else:
        print("ZIP file already exists, skipping download.")
    
    extract_path = os.path.join(download_dir, 'VCTK-Corpus-0.92')
    if extract and not os.path.exists(extract_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        print("Extraction finished.")
    else:
        print("Dataset already extracted or extraction skipped.")
    
    return extract_path


class VCTKDataset(Dataset):
    """
    PyTorch Dataset class for the VCTK Corpus.
    
    Assumes the dataset follows the default VCTK structure:
        root_dir/
            ├── wav48/      <-- Audio data (48kHz), organized by speaker (e.g., p225, p226, ...)
            └── txt/        <-- (Optional) Transcripts; one text file per speaker, if available
    """
    def __init__(self, root_dir, subset="wav48", transform=None):
        """
        Args:
            root_dir (str): Directory containing the extracted VCTK corpus.
            subset (str): Subfolder name under root_dir that contains the WAV files.
                          Typically 'wav48' or 'wav16' depending on version.
            transform (callable, optional): Optional transform to be applied
                on an audio sample.
        """
        self.root_dir = root_dir
        self.wav_dir = os.path.join(root_dir, subset)
        self.transform = transform
        
        # Gather all data: we assume each speaker is a folder under wav_dir
        self.data = []  # list of dictionaries with keys: 'speaker', 'wav_path'
        if not os.path.isdir(self.wav_dir):
            raise ValueError(f"Directory {self.wav_dir} does not exist. Check the dataset structure.")
        
        for speaker in sorted(os.listdir(self.wav_dir)):
            spk_path = os.path.join(self.wav_dir, speaker)
            if os.path.isdir(spk_path):
                for fname in sorted(os.listdir(spk_path)):
                    if fname.lower().endswith('.wav'):
                        file_path = os.path.join(spk_path, fname)
                        self.data.append({
                            'speaker': speaker,
                            'wav_path': file_path
                        })
        
        print(f"Loaded {len(self.data)} audio files from {len(os.listdir(self.wav_dir))} speakers.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Read audio using soundfile
        wav, sr = sf.read(sample['wav_path'])
        if self.transform:
            wav = self.transform(wav)
        # Return a tuple of (speaker_id, waveform, sample_rate)
        return sample['speaker'], wav, sr


# Example usage:
if __name__ == "__main__":
    # Replace this URL with the official direct download link for the ZIP file.
    vctk_url = "https://doi.org/10.7488/ds/2645"  # Note: You may have to resolve this DOI to the actual ZIP URL.
    download_directory = "./data/vctk"

    # Download and extract the dataset
    dataset_path = download_vctk(vctk_url, download_directory, extract=True)

    # Instantiate the dataset (assuming audio files are in 'wav48' folder)
    vctk_dataset = VCTKDataset(root_dir=dataset_path, subset="wav48", transform=None)
    
    # Test one sample
    speaker_id, waveform, sample_rate = vctk_dataset[0]
    print(f"Speaker: {speaker_id}, Sample Rate: {sample_rate}, Audio Shape: {waveform.shape}")
