import enum

from onnxruntime_extensions.tools.pre_post_processing import *


class ModelSource(enum.Enum):
    PYTORCH = 0
    TENSORFLOW = 1
    OTHER = 2


def yolo_detection(model_file: Path, output_file: Path, output_format: str = 'jpg',
                   onnx_opset: int = 16, num_classes: int = 80, input_shape: List[int] = None):
    """
    SSD-like model and Faster-RCNN-like model are including NMS inside already, You can find it from onnx model zoo.

    A pure detection model accept fix-sized(say 1,3,640,640) image as input, and output a list of bounding boxes, which
    the numbers are determinate by anchors.

    This function target for Yolo detection model. It support YOLOv3-yolov8 models theoretically.
    You should assure this model has only one input, and the input shape is [1, 3, h, w].
    The model has either one or more outputs.
        If the model has one output, the output shape is [1,num_boxes, coor+(obj)+cls]
            or [1, coor+(obj)+cls, num_boxes].
        If the model has more than one outputs, you should assure the first output shape is
            [1, num_boxes, coor+(obj)+cls] or [1, coor+(obj)+cls, num_boxes].
    Note: (obj) means it's optional.

    :param model_file: The input model file path.
    :param output_file: The output file path, where the finalized model saved to.
    :param output_format: The output image format, jpg or png.
    :param onnx_opset: The opset version of onnx model, default(16).
    :param num_classes: The number of classes, default(80).
    :param input_shape: The shape of input image (height,width), default will be asked from model input.
    """
    model = onnx.load(str(model_file.resolve(strict=True)))
    inputs = [create_named_value("image", onnx.TensorProto.UINT8, ["num_bytes"])]

    model_input_shape = model.graph.input[0].type.tensor_type.shape
    model_output_shape = model.graph.output[0].type.tensor_type.shape

    # We will use the input_shape to create the model if provided by user.
    if input_shape is not None:
        assert len(input_shape) == 2, "The input_shape should be [h, w]."
        w_in = input_shape[1]
        h_in = input_shape[0]
    else:
        assert (model_input_shape.dim[-1].HasField("dim_value") and
                model_input_shape.dim[-2].HasField("dim_value")), "please provide input_shape in the command args."

        w_in = model_input_shape.dim[-1].dim_value
        h_in = model_input_shape.dim[-2].dim_value

    # Yolov5(v3,v7) has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    # Yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    # https://github.com/ultralytics/ultralytics/blob/e5cb35edfc3bbc9d7d7db8a6042778a751f0e39e/examples/YOLOv8-CPP-Inference/inference.cpp#L31-L33
    # We always want the box info to be the last dim for each of iteration.
    # For new variants like YoloV8, we need to add an transpose op to permute output back.
    need_transpose = False

    output_shape = [model_output_shape.dim[i].dim_value if model_output_shape.dim[i].HasField("dim_value") else -1
                    for i in [-2, -1]]
    if output_shape[0] != -1 and output_shape[1] != -1:
        need_transpose = output_shape[0] < output_shape[1]
    else:
        assert len(model.graph.input) == 1, "Doesn't support adding pre and post-processing for multi-inputs model."
        try:
            import numpy as np
            import onnxruntime
        except ImportError:
            raise ImportError(
                """Please install onnxruntime and numpy to run this script. eg 'pip install onnxruntime numpy'.
Because we need to execute the model to determine the output shape in order to add the correct post-processing""")

        # Generate a random input to run the model and infer the output shape.
        session = onnxruntime.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        input_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[model.graph.input[0].type.tensor_type.elem_type]
        inp = {input_name: np.random.rand(1, 3, h_in, w_in).astype(dtype=input_type)}
        outputs = session.run(None, inp)[0]
        assert len(outputs.shape) == 3 and outputs.shape[0] == 1, "shape of the first model output is not (1, n, m)"
        if outputs.shape[1] < outputs.shape[2]:
            need_transpose = True
        assert num_classes + 4 == outputs.shape[2] or num_classes + 5 == outputs.shape[2], \
            "The output shape is neither (1, num_boxes, num_classes+4(reg)) nor (1, num_boxes, num_classes+5(reg+obj))"

    pipeline = PrePostProcessor(inputs, onnx_opset)
    # precess steps are responsible for converting any jpg/png image to CHW BGR float32 tensor
    # jpg-->BGR(Image Tensor)-->Resize (scaled Image)-->LetterBox (Fix sized Image)-->(from HWC to)CHW-->float32-->1CHW
    pipeline.add_pre_processing(
        [
            ConvertImageToBGR(),  # jpg/png image to BGR in HWC layout
            # Resize an arbitrary sized image to a fixed size in not_larger policy
            Resize((h_in, w_in), policy='not_larger'),
            LetterBox(target_shape=(h_in, w_in)),  # padding or cropping the image to (h_in, w_in)
            ChannelsLastToChannelsFirst(),  # HWC to CHW
            ImageBytesToFloat(),  # Convert to float in range 0..1
            Unsqueeze([0]),  # add batch, CHW --> 1CHW
        ]
    )
    # NMS and drawing boxes
    post_processing_steps = [
        # Debug(),
        Squeeze([0]),  # - Squeeze to remove batch dimension
        SplitOutBoxAndScore(num_classes=num_classes),  # Separate bounding box and confidence outputs
        SelectBestBoundingBoxesByNMS(),  # Apply NMS to suppress bounding boxes
        (ScaleBoundingBoxes(),  # Scale bounding box coords back to original image
         [
             # A connection from original image to ScaleBoundingBoxes
             # A connection from the resized image to ScaleBoundingBoxes
             # A connection from the LetterBoxed image to ScaleBoundingBoxes
             # We can use the three image to calculate the scale factor and offset.
             # With scale and offset, we can scale the bounding box back to the original image.
             utils.IoMapEntry("ConvertImageToBGR", producer_idx=0, consumer_idx=1),
             utils.IoMapEntry("Resize", producer_idx=0, consumer_idx=2),
             utils.IoMapEntry("LetterBox", producer_idx=0, consumer_idx=3),
         ]),

        # Debug take Bounding Boxes NMS layer and return it as output
        # You can modify or create your own output layer by modify _create_graph_for_step function
        Debug(),

        # DrawBoundingBoxes on the original image
        # Model imported from pytorch has CENTER_XYWH format
        # two mode for how to color box,
        #   1. colour_by_classes=True, (colour_by_classes), 2. colour_by_classes=False,(colour_by_confidence)
        (DrawBoundingBoxes(mode='CENTER_XYWH', num_classes=num_classes, colour_by_classes=True),
         [
             utils.IoMapEntry("ConvertImageToBGR", producer_idx=0, consumer_idx=0),
             utils.IoMapEntry("ScaleBoundingBoxes", producer_idx=0, consumer_idx=1),
         ]),
        # Encode to jpg/png
        ConvertBGRToImage(image_format=output_format),
    ]
    # transpose to (num_boxes, coor+conf) if needed
    if need_transpose:
        post_processing_steps.insert(1, Transpose([1, 0]))

    pipeline.add_post_processing(post_processing_steps)

    new_model = pipeline.run(model)
    new_mode = onnx.shape_inference.infer_shapes(new_model)
    onnx.save_model(new_model, str(output_file.resolve()))
