import json
import os

def get_rev_vocab(vocab):
    if  not vocab:
        return None
    return {idx:key for key, idx in vocab.items()}


def send_message_to_slack(config_name):
    project_name = os.path.basename(os.path.abspath("."))
    print(f"The learning is finished with {project_name} project")
