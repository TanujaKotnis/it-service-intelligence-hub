"""
IT Helpdesk Ticket Dataset Builder.

Primary source: Console-AI/IT-helpdesk-synthetic-tickets (HuggingFace Hub)
  - 500 real-style IT helpdesk tickets across 11 categories
  - Categories mapped to our 7-class schema
  - Text: subject + description combined

Supplementary source: hand-crafted synthetic templates
  - Used only for printer_problems (absent in the Hub dataset)
  - Keeps class balance reasonable
"""

import random
from datasets import Dataset, DatasetDict, load_dataset

LABEL2ID = {
    "network_connectivity": 0,
    "software_installation": 1,
    "hardware_malfunction": 2,
    "security_incident": 3,
    "account_access": 4,
    "email_issues": 5,
    "printer_problems": 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Mapping from Hub dataset categories to our schema
CATEGORY_MAP = {
    "Network": "network_connectivity",
    "RemoteWork": "network_connectivity",
    "Infrastructure": "network_connectivity",
    "Software": "software_installation",
    "Licensing": "software_installation",
    "Hardware": "hardware_malfunction",
    "Performance": "hardware_malfunction",
    "Security": "security_incident",
    "Account": "account_access",
    "Communication": "email_issues",
    # Training has no good mapping — excluded
}

# Synthetic templates only for printer_problems (not in Hub dataset)
PRINTER_TEMPLATES = [
    "Printer on floor 3 is offline and not printing any jobs. Queue is building up.",
    "Paper is jamming inside the printer every 2-3 pages. Already cleared once.",
    "My print jobs are stuck in the queue and not printing even after restarting.",
    "Printer is printing blank pages. Toner was just replaced.",
    "Cannot find the printer in the network. It was working fine until last week.",
    "Printed document quality is poor with streaks and smudges all over.",
    "Network printer driver installation failed on my new laptop.",
    "Color printer is printing in black and white only even when color is selected.",
    "Print job is being sent to a different printer than the one I selected.",
    "Printer makes loud noise when printing and sometimes does not finish the job.",
    "Duplex printing is not working. It prints single-sided despite my settings.",
    "Scanner on the MFD is not working. Documents sent to email are not arriving.",
    "Printer on second floor needs toner replacement. It is out of paper too.",
    "I cannot install a printer manually. I do not have admin rights to add printers.",
    "Print server is not reachable. All printers on this floor are unavailable.",
    "My default printer keeps changing after every restart.",
    "Printer is online but takes 10+ minutes to print a simple 2-page document.",
    "HP LaserJet on my desk is showing 'Replace Imaging Drum' message.",
    "I need to set up a new printer for a conference room. Who should I contact?",
    "Printed pages have a gray tint across the whole page. Looks like a drum issue.",
]

# Backward-compatibility alias expected by app.py (dict form)
TEMPLATES = {
    "network_connectivity": [
        "VPN connection fails with authentication error.",
        "Cannot access corporate network after working remotely.",
        "Wi-Fi disconnects every few minutes in the office.",
        "Network drive shows as disconnected after login.",
        "Internet is very slow on the corporate network today.",
    ],
    "software_installation": [
        "Need to install Python for data analysis work.",
        "SAP GUI keeps crashing after the latest update.",
        "Microsoft Office license has expired on my laptop.",
        "Software Center shows no applications available.",
        "AutoCAD says no license available when I try to open it.",
    ],
    "hardware_malfunction": [
        "Laptop screen flickers and sometimes goes black.",
        "Keyboard keys are stuck and not responding.",
        "Computer is running very slow and overheating.",
        "USB ports on my laptop are not recognizing devices.",
        "Monitor shows no signal when connected to docking station.",
    ],
    "security_incident": [
        "I received a suspicious email asking for my credentials.",
        "My account may have been accessed without my knowledge.",
        "I accidentally clicked a phishing link in an email.",
        "Unusual login activity detected on my account.",
        "Ransomware warning appeared on my screen.",
    ],
    "account_access": [
        "Cannot log in to my corporate account after password reset.",
        "Account locked out after too many failed attempts.",
        "MFA app is not generating codes on my new phone.",
        "SSO redirect loop when accessing internal portals.",
        "Need access to a shared mailbox for my new role.",
    ],
    "email_issues": [
        "Outlook is not receiving new emails since this morning.",
        "Cannot send emails — getting bounce-back errors.",
        "Teams meeting audio is not working.",
        "Shared calendar is not visible in Outlook.",
        "Emails are going to recipients' spam folders.",
    ],
    "printer_problems": PRINTER_TEMPLATES,
}


def _load_hub_dataset():
    """Load and map the HuggingFace Hub dataset to our 7-class schema."""
    try:
        raw = load_dataset("Console-AI/IT-helpdesk-synthetic-tickets", split="train")
    except Exception as e:
        print(f"[dataset_generator] Could not load Hub dataset: {e}")
        return [], []

    texts, labels = [], []
    for row in raw:
        mapped = CATEGORY_MAP.get(row["category"])
        if mapped is None:
            continue  # skip unmapped categories (e.g. Training)
        # Combine subject + description for richer input
        text = f"{row['subject']}. {row['description']}".strip()
        texts.append(text)
        labels.append(LABEL2ID[mapped])
    return texts, labels


def _load_printer_synthetic(n: int = 20, seed: int = 42):
    """Return n synthetic printer_problems samples."""
    random.seed(seed)
    templates = PRINTER_TEMPLATES * (n // len(PRINTER_TEMPLATES) + 1)
    selected = templates[:n]
    return selected, [LABEL2ID["printer_problems"]] * n


def build_dataset(printer_samples: int = 20, seed: int = 42) -> DatasetDict:
    """
    Build the training dataset.

    Uses real Hub data for 6 categories and synthetic data for printer_problems.
    Returns a DatasetDict with train / validation splits (80/20).
    """
    all_texts, all_labels = _load_hub_dataset()

    # Supplement with printer_problems synthetic data
    p_texts, p_labels = _load_printer_synthetic(printer_samples, seed)
    all_texts.extend(p_texts)
    all_labels.extend(p_labels)

    if not all_texts:
        raise RuntimeError("No data loaded. Check HuggingFace login / network access.")

    # Shuffle
    random.seed(seed)
    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)

    # 80/20 split
    split = int(0.8 * len(all_texts))
    dataset = DatasetDict({
        "train": Dataset.from_dict({
            "text": list(all_texts[:split]),
            "label": list(all_labels[:split]),
        }),
        "validation": Dataset.from_dict({
            "text": list(all_texts[split:]),
            "label": list(all_labels[split:]),
        }),
    })
    return dataset


def get_sample_tickets():
    """Return one sample ticket per category for display/demo purposes."""
    hub_texts, hub_labels = _load_hub_dataset()
    seen = set()
    samples = []
    for text, label in zip(hub_texts, hub_labels):
        cat = ID2LABEL[label]
        if cat not in seen:
            samples.append({"text": text, "label": cat})
            seen.add(cat)
    # Add printer sample
    if "printer_problems" not in seen:
        samples.append({"text": PRINTER_TEMPLATES[0], "label": "printer_problems"})
    return samples


if __name__ == "__main__":
    ds = build_dataset()
    print(f"Train samples:      {len(ds['train'])}")
    print(f"Validation samples: {len(ds['validation'])}")
    print("\nLabel distribution (train):")
    from collections import Counter
    counts = Counter(ds["train"]["label"])
    for label_id, count in sorted(counts.items()):
        print(f"  {ID2LABEL[label_id]}: {count}")
