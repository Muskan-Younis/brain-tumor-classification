import os
import cv2
import hashlib
from PIL import Image

IMG_SIZE = 224

# 🔁 INPUT (original dataset)
train_path = "brain_tumor_dataset/Training"
test_path = "brain_tumor_dataset/Testing"

# 📁 OUTPUT (cleaned dataset)
output_base = "cleaned_dataset"


# -----------------------------
# Helper: create same structure
# -----------------------------
def create_output_structure(input_path, output_path):
    for class_name in os.listdir(input_path):
        class_input = os.path.join(input_path, class_name)
        class_output = os.path.join(output_path, class_name)

        if os.path.isdir(class_input):
            os.makedirs(class_output, exist_ok=True)


# -----------------------------
# Cleaning pipeline
# -----------------------------
def clean_folder(input_folder, output_folder):
    hashes = set()
    total = 0
    saved = 0
    skipped = 0

    for class_name in os.listdir(input_folder):
        class_input = os.path.join(input_folder, class_name)
        class_output = os.path.join(output_folder, class_name)

        if not os.path.isdir(class_input):
            continue

        for file in os.listdir(class_input):
            input_path = os.path.join(class_input, file)
            total += 1

            try:
                # ✅ Check corrupted
                img_pil = Image.open(input_path)
                img_pil.verify()

                # ✅ Read with OpenCV
                img = cv2.imread(input_path)
                if img is None:
                    skipped += 1
                    continue

                # ✅ Resize
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # ✅ Hash (after resize for consistency)
                img_bytes = img.tobytes()
                file_hash = hashlib.md5(img_bytes).hexdigest()

                if file_hash in hashes:
                    skipped += 1
                    continue

                hashes.add(file_hash)

                # ✅ Save cleaned image
                output_file = os.path.join(class_output, f"{saved}.jpg")
                cv2.imwrite(output_file, img)
                saved += 1

            except:
                skipped += 1

    print(f"[DONE] {input_folder}")
    print(f"Total: {total} | Saved: {saved} | Skipped: {skipped}\n")


# -----------------------------
# Count dataset
# -----------------------------
def count_images(folder):
    print(f"\n📊 Distribution for {folder}:")
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)

        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            print(f"{class_name}: {count}")


# -----------------------------
# RUN EVERYTHING
# -----------------------------
print("🚀 Creating cleaned dataset...\n")

# Create folder structure
create_output_structure(train_path, os.path.join(output_base, "Training"))
create_output_structure(test_path, os.path.join(output_base, "Testing"))

# Clean Training
clean_folder(train_path, os.path.join(output_base, "Training"))

# Clean Testing
clean_folder(test_path, os.path.join(output_base, "Testing"))

# Show distribution
count_images(os.path.join(output_base, "Training"))
count_images(os.path.join(output_base, "Testing"))

print("\n✅ CLEANED DATASET READY → 'cleaned_dataset'")