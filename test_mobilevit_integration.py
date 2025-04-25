import torch
from ultralytics.nn.tasks import DetectionModel

def test_mobilevit_integration():
    """
    Test the integration of MobileViTv2Backbone into the YOLO model.
    This function:
    1. Loads the YOLO model with MobileViTv2Backbone.
    2. Runs a forward pass with dummy input.
    3. Verifies the output shapes of the feature maps.
    """
    # Load the YOLO model with the updated yolo11.yaml configuration
    model_path = "content/Transformer_Backbone_for_YOLOv11/ultralytics/cfg/models/11/yolo11.yaml"
    model = DetectionModel(cfg=model_path, ch=3, nc=80, verbose=True)

    # Create dummy input (batch size 1, 3 channels, 256x256 image)
    dummy_input = torch.randn(1, 3, 256, 256)

    # Run a forward pass
    print("Running forward pass...")
    output = model(dummy_input)

    # Check the output shapes
    print("\nOutput shapes:")
    if isinstance(output, list):
        for i, feature_map in enumerate(output):
            print(f"Feature map {i}: {feature_map.shape}")
    else:
        print(output.shape)

    # Verify the expected output shapes
    expected_shapes = [
        (1, 256, 32, 32),  # Feature map 0 (P3)
        (1, 384, 16, 16),  # Feature map 1 (P4)
        (1, 512, 8, 8),    # Feature map 2 (P5)
    ]
    for i, feature_map in enumerate(output):
        assert feature_map.shape == expected_shapes[i], f"Feature map {i} shape mismatch: {feature_map.shape}"

    print("\nMobileViTv2Backbone integration test passed!")

if __name__ == "__main__":
    test_mobilevit_integration()