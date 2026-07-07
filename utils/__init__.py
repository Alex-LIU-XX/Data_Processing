from .signal_analysis import (
    ThresholdConfig,
    compute_adaptive_threshold,
    compute_deviation_signals,
    detect_sudden_changes,
    extract_trend,
)
from .trend_alignment import (
    AlignmentConfig,
    align_signals,
    analyze_trend_alignment,
    compute_optimal_lag,
    directional_agreement,
)
from .robot_visualizer import (
    RobotVisualizer,
    joint_map_dataset_to_urdf,
    parse_urdf,
)
