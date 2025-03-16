import time
from google.cloud import vision
from google.cloud import storage
import json


def async_ocr_pdf(input_gcs_uri: str, output_gcs_uri: str, project_id: str, batch_size: int = 100):
    client = vision.ImageAnnotatorClient()
    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    gcs_source = vision.GcsSource(uri=input_gcs_uri)
    input_config = vision.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")
    gcs_destination = vision.GcsDestination(uri=output_gcs_uri)
    output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=batch_size)
    async_request = vision.AsyncAnnotateFileRequest(features=[feature], input_config=input_config, output_config=output_config)
    operation = client.async_batch_annotate_files(requests=[async_request])
    print("Processing PDF OCR...")
    operation.result(timeout=300)
    print("OCR processing complete.")

def list_gcs_objects(bucket_name: str, prefix: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    print("Objects in bucket (with prefix={}):".format(prefix))
    for blob in bucket.list_blobs(prefix=prefix):
        print(blob.name)


def download_and_parse_ocr_output(bucket_name: str, prefix: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    all_text = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith(".json"):
            json_data = blob.download_as_text()
            data = json.loads(json_data)
            responses = data.get("responses", [])
            for r in responses:
                annotation = r.get("fullTextAnnotation", {})
                text = annotation.get("text", "")
                all_text.append(text)
    combined_text = "\n".join(all_text)
    return combined_text


if __name__ == "__main__":
    PROJECT_ID = "x"
    INPUT_URI = "xx"
    OUTPUT_URI = "xx"  # Ensure the output ends with '/'

    async_ocr_pdf(
        input_gcs_uri=INPUT_URI,
        output_gcs_uri=OUTPUT_URI,
        project_id=PROJECT_ID,
        batch_size=100
    )

    print("Listing OCR output files:")
    list_gcs_objects(bucket_name="xx", prefix="output/")

    print("\nParsing OCR JSON outputs to extract text...")
    text_result = download_and_parse_ocr_output(
        bucket_name="xx",
        prefix="output/"
    )

    print("\nExtracted Text (First 500 characters):")
    print(text_result[:500])

    with open("final_output.txt", "w", encoding="utf-8") as out_file:
        out_file.write(text_result)
    print("\nText saved to final_output.txt")
