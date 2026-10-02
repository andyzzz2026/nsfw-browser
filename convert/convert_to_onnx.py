
import os

os.environ["OPENNSFW2_KERAS"] = "tf-keras"

from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort
import opennsfw2 as n2


BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    BASE_DIR / "web" / "models" / "model.onnx"
)


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading pretrained OpenNSFW2...")

    model = n2.make_open_nsfw_model()

    print("Parameters:", model.count_params())

    print("Converting TensorFlow to ONNX...")

    input_signature = [
        tf.TensorSpec(
            (None, 224, 224, 3),
            tf.float32,
            name="input"
        )
    ]

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=17,
        output_path=str(OUTPUT_PATH)
    )

    print("Validating ONNX...")

    onnx_model = onnx.load(str(OUTPUT_PATH))
    onnx.checker.check_model(onnx_model)

    print("Comparing TensorFlow and ONNX...")

    rng = np.random.default_rng(42)

    sample = rng.uniform(
        -100,
        150,
        (1, 224, 224, 3)
    ).astype(np.float32)

    tf_result = model.predict(
        sample,
        verbose=0
    )

    session = ort.InferenceSession(
        str(OUTPUT_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    onnx_result = session.run(
        [output_name],
        {input_name: sample}
    )[0]

    print("TensorFlow:", tf_result[0])
    print("ONNX:", onnx_result[0])

    np.testing.assert_allclose(
        tf_result,
        onnx_result,
        rtol=1e-3,
        atol=1e-4
    )

    size = OUTPUT_PATH.stat().st_size / 1024**2

    print("\nSUCCESS!")
    print("Output:", OUTPUT_PATH)
    print(f"Model size: {size:.2f} MB")
    print("Input:", input_name)
    print("Output:", output_name)


if __name__ == "__main__":
    main()
