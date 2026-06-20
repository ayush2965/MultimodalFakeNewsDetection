import easyocr
import cv2
from text_predict import text_predict

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if CUDA available

def extract_text(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return ""

    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform OCR
    results = reader.readtext(gray)

    if not results:
        return ""

    # Sort by top-left y, then x (reading order)
    results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))

    # Join all text fragments into one sentence
    full_text = " ".join([res[1] for res in results])

    return full_text


# Example usage
if __name__ == "__main__":
    image_path = r"D:\lion.jpeg"

    sentence = extract_text(image_path)

    if sentence:
        print("Detected text:")
        print(sentence)
    else:
        print("No text detected.")
    