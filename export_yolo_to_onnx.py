from pathlib import Path
import onnxruntime_extensions
from YOLO.add_pre_post_processing_to_model import yolo_detection


def yolov8_to_onnx(yolo_path, onnx_path):
    from pip._internal import main as pipmain
    try:
        import ultralytics
    except ImportError:
        pipmain(['install', 'ultralytics'])
        import ultralytics
    pt_model = Path(yolo_path)
    model = ultralytics.YOLO(str(pt_model))
    success = model.export(format='onnx')
    assert success, "Failed to export to onnx"
    import shutil
    shutil.move(pt_model.with_suffix('.onnx'), onnx_path)


def add_pre_post_processing(yolo_path, onnx_path, onnx_after):
    onnx_path = Path(onnx_path)
    onnx_after = Path(onnx_after)
    yolov8_to_onnx(yolo_path, onnx_path)
    yolo_detection(onnx_path, onnx_after, 'jpg', num_classes=3, onnx_opset=18)


def test_inference(img_in, img_out, onnx_model_file):
    import onnxruntime as ort
    import numpy as np
    onnx_model_file = Path(onnx_model_file)
    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())

    image = np.frombuffer(open(img_in, 'rb').read(),
                          dtype=np.uint8)
    session = ort.InferenceSession(str(onnx_model_file), providers=providers, sess_options=session_options)

    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: image}
    # debug = session.run(['transposed_debug'], inp)[0]
    outputs = session.run(['image_out'], inp)[0]

    open(img_out, 'wb').write(outputs)


#
#
if __name__ == '__main__':
    yolo_path = '/opt/github/pytorch-to-onnx/best.pt'
    onnx_path = '/opt/github/pytorch-to-onnx/models/quality_onnx.onnx'
    onnx_after = '/opt/github/pytorch-to-onnx/models/quality_onnx_pre_post.onnx'
    add_pre_post_processing(yolo_path, onnx_path, onnx_after)

    img_test = '/opt/data/pti_ocr/quality/quality_yolo2/images/train/73_0.jpg'
    img_out = '/opt/github/pytorch-to-onnx/output.jpg'
    test_inference(img_test, img_out, onnx_after)