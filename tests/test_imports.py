"""
Test that all modules in the src/ directory can be imported.
This helps catch missing dependencies or circular import errors early.
"""


def test_gaussian_pipeline_imports():
    from gaussian_pipeline import gaussian_optimizer

    # from gaussian_pipeline import gaussian_initializer # Add as created
    # from gaussian_pipeline import gaussian_renderer # Add as created
    # from gaussian_pipeline import gaussian_postprocess # Add as created
    # from gaussian_pipeline import gaussian_output # Add as created
    # from gaussian_pipeline import texture_baker # Add as created
    assert gaussian_optimizer is not None


def test_preprocessing_imports():
    from preprocessing import camera_estimator, human_segmenter, keypoint_detector

    assert camera_estimator is not None
    assert human_segmenter is not None
    assert keypoint_detector is not None


def test_smplx_fitting_imports():
    from smplx_fitting import pixie_initializer, smplx_refiner

    assert pixie_initializer is not None
    assert smplx_refiner is not None


def test_utils_imports():
    # from utils import data_utils # Add as created
    # from utils import torch_utils # Add as created
    # from utils import colmap_wrappers # Add as created
    # from utils import geometry # Add as created
    assert True  # Placeholder until utils modules are added


def test_evaluation_imports():
    # from evaluation import quality_assessor # Add as created
    assert True  # Placeholder until evaluation modules are added


def test_main_pipeline_importable_components():
    import main_pipeline

    assert main_pipeline is not None
