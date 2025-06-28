"""
sample:
{"transcription": "CHAPTER SIXTEEN I MIGHT HAVE TOLD YOU OF THE BEGINNING OF THIS LIAISON IN A FEW LINES BUT I WANTED YOU TO SEE EVERY STEP BY WHICH WE CAME I TO AGREE TO WHATEVER MARGUERITE WISHED", "audio_path": "/root/epfs/data/LibriSpeech/LibriSpeech/train-clean-100/374/180298/374-180298-0000.flac"}
{"transcription": "MARGUERITE TO BE UNABLE TO LIVE APART FROM ME IT WAS THE DAY AFTER THE EVENING WHEN SHE CAME TO SEE ME THAT I SENT HER MANON LESCAUT FROM THAT TIME SEEING THAT I COULD NOT CHANGE MY MISTRESS'S LIFE I CHANGED MY OWN", "audio_path": "/root/epfs/data/LibriSpeech/LibriSpeech/train-clean-100/374/180298/374-180298-0001.flac"}
{"transcription": "I WISHED ABOVE ALL NOT TO LEAVE MYSELF TIME TO THINK OVER THE POSITION I HAD ACCEPTED FOR IN SPITE OF MYSELF IT WAS A GREAT DISTRESS TO ME THUS MY LIFE GENERALLY SO CALM", "audio_path": "/root/epfs/data/LibriSpeech/LibriSpeech/train-clean-100/374/180298/374-180298-0002.flac"}
{"transcription": "ASSUMED ALL AT ONCE AN APPEARANCE OF NOISE AND DISORDER NEVER BELIEVE HOWEVER DISINTERESTED THE LOVE OF A KEPT WOMAN MAY BE THAT IT WILL COST ONE NOTHING", "audio_path": "/root/epfs/data/LibriSpeech/LibriSpeech/train-clean-100/374/180298/374-180298-0003.flac"}
{"transcription": "NOTHING IS SO EXPENSIVE AS THEIR CAPRICES FLOWERS BOXES AT THE THEATRE SUPPERS DAYS IN THE COUNTRY WHICH ONE CAN NEVER REFUSE TO ONE'S MISTRESS AS I HAVE TOLD YOU I HAD LITTLE MONEY", "audio_path": "/root/epfs/data/LibriSpeech/LibriSpeech/train-clean-100/374/180298/374-180298-0004.flac"}
{"transcription": "MY FATHER WAS AND STILL IS RECEVEUR GENERAL AT C HE HAS A GREAT REPUTATION THERE FOR LOYALTY THANKS TO WHICH HE WAS ABLE TO FIND THE SECURITY WHICH HE NEEDED IN ORDER TO ATTAIN THIS POSITION", "audio_path": "/root/epfs/data/LibriSpeech/LibriSpeech/train-clean-100/374/180298/374-180298-0005.flac"}
{"transcription": "I CAME TO PARIS STUDIED LAW WAS CALLED TO THE BAR AND LIKE MANY OTHER YOUNG MEN PUT MY DIPLOMA IN MY POCKET AND LET MYSELF DRIFT AS ONE SO EASILY DOES IN PARIS", "audio_path": "/root/epfs/data/LibriSpeech/LibriSpeech/train-clean-100/374/180298/374-180298-0006.flac"}
{"transcription": "MY EXPENSES WERE VERY MODERATE ONLY I USED UP MY YEAR'S INCOME IN EIGHT MONTHS AND SPENT THE FOUR SUMMER MONTHS WITH MY FATHER WHICH PRACTICALLY GAVE ME TWELVE THOUSAND FRANCS A YEAR AND IN ADDITION THE REPUTATION OF A GOOD SON", "audio_path": "/root/epfs/data/LibriSpeech/LibriSpeech/train-clean-100/374/180298/374-180298-0007.flac"}
{"transcription": "FOR THE REST NOT A PENNY OF DEBT THIS THEN WAS MY POSITION WHEN I MADE THE ACQUAINTANCE OF MARGUERITE YOU CAN WELL UNDERSTAND THAT IN SPITE OF MYSELF MY EXPENSES SOON INCREASED", "audio_path": "/root/epfs/data/LibriSpeech/LibriSpeech/train-clean-100/374/180298/374-180298-0008.flac"}
{"transcription": "MARGUERITE'S NATURE WAS VERY CAPRICIOUS AND LIKE SO MANY WOMEN SHE NEVER REGARDED AS A SERIOUS EXPENSE THOSE THOUSAND AND ONE DISTRACTIONS WHICH MADE UP HER LIFE SO WISHING TO SPEND AS MUCH TIME WITH ME AS POSSIBLE", "audio_path": "/root/epfs/data/LibriSpeech/LibriSpeech/train-clean-100/374/180298/374-180298-0009.flac"}    
"""

from pathlib import Path
import argparse
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Generate JSONL file from LibriSpeech dataset")
    parser.add_argument("--librispeech_dir", type=str, default="/root/epfs/data/LibriSpeech/LibriSpeech", 
                        help="Path to LibriSpeech dataset directory")
    parser.add_argument("--subset", type=str, default="train-clean-100", 
                        help="Subset of LibriSpeech dataset")
    parser.add_argument("--output_file", type=str, default="data/train.jsonl", 
                        help="Output JSONL file path")
    return parser.parse_args()

def main(args):
    librispeech_dir = Path(args.librispeech_dir)
    subset = args.subset
    output_file = args.output_file
    
    assert subset in ["train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other", "test-clean", "test-other","train-960"], f"Invalid subset: {subset}" 
    assert librispeech_dir.is_dir(), f"Data directory {librispeech_dir} does not exist"
    
    # Collect all transcription files
    if subset == "train-960":
        train_files = []
        for subset in ["train-clean-100", "train-clean-360", "train-other-500"]:
            train_files.extend(list(librispeech_dir.glob(f"{subset}/**/*.trans.txt")))
        trans_files = train_files
    else:
        trans_files = list(librispeech_dir.glob(f"{subset}/**/*.trans.txt"))
    data_entries = []
    
    # Process each transcription file
    for trans_file in tqdm(trans_files, desc="Processing transcription files"):
        with open(trans_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Process each line in transcription file
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Split into file_id and transcription
            parts = line.split(' ', 1)
            if len(parts) < 2:
                logger.warning(f"Invalid line format in {trans_file}: {line}")
                continue
                
            file_id = parts[0]
            transcription = parts[1].strip()
            
            # Construct audio file path
            audio_file = trans_file.parent / f"{file_id}.flac"
            
            # Check if audio file exists
            if not audio_file.exists():
                logger.warning(f"Audio file not found: {audio_file}")
                continue
            
            # Create data entry
            entry = {
                "transcription": transcription.upper(),  # Convert to uppercase as shown in examples
                "audio_path": str(audio_file.absolute())
            }
            
            data_entries.append(entry)
      
    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in tqdm(data_entries, desc="Writing JSONL file"):
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    args = get_args()
    main(args)