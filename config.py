# config.py - Backward compatibility wrapper
# 向后兼容包装器，实际配置在 configs/whale.py
# This file re-exports all configurations from configs/whale.py
# for backward compatibility with existing code that imports config

from configs.whale import *
