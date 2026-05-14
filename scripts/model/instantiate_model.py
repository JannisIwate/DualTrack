from omegaconf import OmegaConf 
from src.models import get_model 

cfg_path = 'dualtrack_ft_tus_rec_2025.yaml'
cfg = OmegaConf.load(cfg_path)
cfg.checkpoint = 'dualtrack_ft_tus_rec_2025_v3_best.pt'

model = get_model(**cfg)