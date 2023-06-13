## Convert from pytorch to ONNX

### CRAFT

Convert [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) text detection model to onnx

Run craft_to_onnx to convert CRAFT .pth model to .onnx </br>
Input shape need to be fixed, set the shape according to your project

### CTPN

Convert [CTPN](https://github.com/eragonruan/text-detection-ctpn) text detection model to onnx

Run craft_to_onnx to convert CTPN .pth model to .onnx </br>
Input shape need to be fixed, set the shape according to your project


### YOLO

Run export_yolo_to_onnx to convert YOLO .pt model to .onnx with NMS function </br>
Input shape is dynamic after convert </br>
Output contain of 3 layer, image_out, scaled_box_out_next, scaled_box_out_debug </br>
image_out is image with drew bounding boxes </br>
scaled_box_out_next and scaled_box_out_debug is the bounding boxes output with [x, y, w, h, conf, class]