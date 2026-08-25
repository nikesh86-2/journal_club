# In Act Mode, replace the contents of training_trigger.py with the following:

import subprocess
import os

def trigger_vlab2_training():
    train_script = os.path.join(os.path.dirname(__file__), 'train_lora.py')
    if os.path.exists(train_script):
        subprocess.run(['python', train_script])
        return True
    else:
        print("Error: train_lora.py not found")
        return False

def merge_lora_weights():
    merge_script = os.path.join(os.path.dirname(__file__), 'merge_lora.py')
    if os.path.exists(merge_script):
        subprocess.run(['python', merge_script, '--lora_adapter', 'training/journal_club_output', '--output', 'training/journal_club_merged_model'])
        return True
    else:
        print("Error: merge_lora.py not found")
        return False

def check_and_trigger_training():
    """
    Modified from original - directly calls local scripts instead of VLAB
    """
    if trigger_vlab2_training():
        return merge_lora_weights()
    return False