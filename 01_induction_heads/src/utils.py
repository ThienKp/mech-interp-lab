import os
from datetime import datetime

VOCAB_LIST: list = ["T00", "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T10", "T11", "T12", "T13", "T14", "T15", "T16", "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30", "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44", "T45", "T46", "T47", "T48", "T49", "T50", "T51", "T52", "T53", "T54", "T55", "T56", "T57", "T58", "T59", "T60", "T61", "T62", "T63", "T64", "T65", "T66", "T67", "T68", "T69", "T70", "T71", "T72", "T73", "T74", "T75", "T76", "T77", "T78", "T79", "T80", "T81", "T82", "T83", "T84", "T85", "T86", "T87", "T88", "T89", "T90", "T91", "T92", "T93", "T94", "T95", "T96", "T97", "T98", "T99"]

VOCAB_SIZE: int = len(VOCAB_LIST)

SEED: int = 515
D_MODEL: int = 64
N_HEADS: int = 4
D_HEAD: int = D_MODEL // N_HEADS
N_LAYERS: int = 2
CONTEXT_LENGTH: int = 64
MODEL_DATABASE: str = "01_induction_heads/model_db"
VISUAL_DATABASE: str = "01_induction_heads/visualizations"
REPEATED: bool = False

def ensure_model_database() -> None:
    """
    Ensure that the model database exists. If it doesn't, create it.
    """
    os.makedirs(MODEL_DATABASE, exist_ok=True)

def ensure_visualizations() -> None:
    """
    Ensure that the visualizations directory exists. If it doesn't, create it.
    """
    os.makedirs(VISUAL_DATABASE, exist_ok=True)

def generate_new_save() -> str:
    """
    Generate a new save directory for the model with a timestamp.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_dir = os.path.join(MODEL_DATABASE, f"model_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    return f"model_{timestamp}"

def get_model_filename(pathname: str) -> str:
    """
    Generate a model filename based on the provided pathname.
    """
    return os.path.join(MODEL_DATABASE, pathname, "model.pt")

def get_config_filename(pathname: str) -> str:
    """
    Generate a configuration filename based on the provided pathname.
    """
    return os.path.join(MODEL_DATABASE, pathname, "config.json")

def get_metrics_filename(pathname: str) -> str:
    """
    Generate a metrics filename based on the provided pathname.
    """
    return os.path.join(MODEL_DATABASE, pathname, "metrics.json")

def get_visualization_filename(filename: str) -> str:
    """
    Generate a visual filename in the visualization directory.
    """
    return os.path.join(VISUAL_DATABASE, filename)