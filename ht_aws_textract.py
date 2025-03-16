import time
import boto3


def start_text_detection(s3_bucket, s3_file):
    textract = boto3.client('textract', region_name='us-east-1')

    response = textract.start_document_text_detection(
        DocumentLocation={
            'S3Object': {
                'Bucket': s3_bucket,
                'Name': s3_file
            }
        }
    )
    return response["JobId"]


def is_job_complete(job_id):
    textract = boto3.client('textract', region_name='us-east-1')
    status = "IN_PROGRESS"

    while status == "IN_PROGRESS":
        response = textract.get_document_text_detection(JobId=job_id)
        status = response["JobStatus"]
        print(f"Current Textract job status: {status}")
        if status == "IN_PROGRESS":
            time.sleep(5)
    return status


def get_job_results(job_id):
    textract = boto3.client('textract', region_name='us-east-1')
    pages = []
    response = textract.get_document_text_detection(JobId=job_id)
    pages.append(response)
    while "NextToken" in response:
        next_token = response["NextToken"]
        response = textract.get_document_text_detection(
            JobId=job_id,
            NextToken=next_token
        )
        pages.append(response)

    return pages


def extract_text_from_pages(pages):
    all_text = []
    for page in pages:
        blocks = page.get("Blocks", [])
        for block in blocks:
            if block["BlockType"] == "LINE":
                all_text.append(block["Text"])
    return "\n".join(all_text)


if __name__ == "__main__":
    S3_BUCKET = "xxx"
    S3_FILE = "xxx"
    job_id = start_text_detection(S3_BUCKET, S3_FILE)
    print(f"Started Textract job with JobId: {job_id}")
    final_status = is_job_complete(job_id)

    if final_status == "SUCCEEDED":
        print("Textract job succeeded. Retrieving results...")
        result_pages = get_job_results(job_id)
        final_text = extract_text_from_pages(result_pages)
        print("\n--- EXTRACTED TEXT (first 500 chars) ---")
        print(final_text[:500])
        with open("textract_output.txt", "w", encoding="utf-8") as f:
            f.write(final_text)
        print("\nFull text saved to 'textract_output.txt'.")

    else:
        print(f"Textract job ended with status: {final_status}")
