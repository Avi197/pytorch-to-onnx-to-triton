import tritonclient.http as httpclient

triton_url = 'localhost:8000'
triton_client = httpclient.InferenceServerClient(url=triton_url, verbose=False)


def infer_craft(model_name, img):
    input0 = httpclient.InferInput("input", img.shape, "FP32")
    input0.set_data_from_numpy(img, binary_data=True)
    inputs = [input0]

    output0 = httpclient.InferRequestedOutput('output', binary_data=True)
    outputs = [output0]

    results = triton_client.infer(
        model_name,
        inputs=inputs,
        outputs=outputs,
    )

    return results

