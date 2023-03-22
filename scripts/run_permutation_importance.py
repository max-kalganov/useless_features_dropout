import gin

from src.utils import dump_features_importance

if __name__ == '__main__':
    gin.parse_config_file("configs/features_importance_config.gin")
    dump_features_importance()
