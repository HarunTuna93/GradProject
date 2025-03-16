import requests
import time
from azure.storage.blob import BlobServiceClient

endpoint = "xx"
key = "xx"
account_name = "xx"
container_name = "xx"
sas_token = (
    "xx"
    "xx"
)

def get_image_urls_from_blob(account_name, container_name, sas_token):
    print("Retrieving...")
    blob_service_client = BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=sas_token
    )
    container_client = blob_service_client.get_container_client(container_name)
    image_urls = []
    for blob in container_client.list_blobs():
        blob_url = (
            f"https://{account_name}.blob.core.windows.net/"
            f"{container_name}/{blob.name}?{sas_token}"
        )
        print(f"Found blob: {blob.name}")
        image_urls.append(blob_url)
    print(f"Total images: {len(image_urls)}")
    return image_urls


def analyze_image(image_url):
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/json"
    }
    data = {"url": image_url}
    #request
    print(f"\nSending analyze request for image: {image_url}")
    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 202:  # 202 Accepted
        operation_location = response.headers["Operation-Location"]
        print(f"Operation accepted. Polling for results at: {operation_location}")
    else:
        print(f"Error: HTTP {response.status_code}")
        print(f"Response Text: {response.text}")
        return None
    print("Polling for results...")
    while True:
        poll_response = requests.get(operation_location, headers=headers)
        poll_result = poll_response.json()
        if poll_response.status_code != 200:
            print(f"Polling failed: HTTP {poll_response.status_code}")
            print(f"Response: {poll_result}")
            return None
        status = poll_result.get("status")
        print(f"Status: {status}")
        if status == "succeeded":
            return poll_result
        elif status == "failed":
            print("Analysis failed.")
            print(poll_result)
            return None
        time.sleep(2)

def extract_text_from_result(result):
    extracted_text = []
    read_results = result.get("analyzeResult", {}).get("readResults", [])
    for page in read_results:
        for line in page.get("lines", []):
            extracted_text.append(line["text"])
    return "\n".join(extracted_text)


def main():
    image_urls = get_image_urls_from_blob(account_name, container_name, sas_token)
    if not image_urls:
        print("No images found in the container.")
        return

    all_text = []
    for idx, image_url in enumerate(image_urls, start=1):
        print(f"\n--- Processing image {idx}/{len(image_urls)} ---")
        result = analyze_image(image_url)
        if result:
            text = extract_text_from_result(result)
            all_text.append(text)

    combined_text = "\n\n".join(all_text)
    with open("ocr_results.txt", "w", encoding="utf-8") as f:
        f.write(combined_text)
    print("\nOCR results saved to 'ocr_results1.txt'.")


if __name__ == "__main__":
    main()
