import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageTk
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import seaborn as sns
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileWriter, PdfFileReader
import sys

def generate_page_list(num_pages):
    print(f'{num_pages} pages')
    page_list = []

  
    is_odd = num_pages % 2

    if is_odd
        num_pages += 1

    a, b = 0, num_pages - 1

   +2)
    page_type = '4M' if num_pages % 4 == 0 else '4M+2'

    print(f'page type: {page_type}')

    for i in range(num_pages):
        flag = i % 4

        if flag == 0:
            page_list.append(b)
            b -= 1
        elif flag == 1 or flag == 2:
            page_list.append(a)
            a += 1
        elif flag == 3:
            page_list.append(b)
            b -= 1

    
        page_list[page_list.index(num_pages - 1)] = 'b'

    return page_list

def create_booklet(input_file):
    output = PdfFileWriter()

    try:
        pdf = PdfFileReader(open(input_file, 'rb'))
    except FileNotFoundError:
        print('Input file not found')
        sys.exit(1)

    print('Processing...')

    last_page = pdf.getPage(0)
    width = last_page.mediaBox.getWidth()
    height = last_page.mediaBox.getHeight()

    for page_index in generate_page_list(pdf.getNumPages()):
        if page_index == 'b':
            output.addBlankPage(width, height)
        else:
            output.addPage(pdf.getPage(page_index))

    output_file_path = f"{input_file}_booklet.pdf"
    print('Writing file...')

    with open(output_file_path, 'wb') as output_stream:
        output.write(output_stream)

    print(f"{output.getNumPages()} pages created in {output_file_path}")


def classify_image_torch(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    
    with torch.no_grad():
        output = model_torch(img)
    
    _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item()


def classify_image_keras(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = transforms.ToTensor()(img)
    img_array = torch.unsqueeze(img_array, 0)
    img_array = preprocess_input(img_array)

  
    predictions = model_keras.predict(img_array)

    
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions[0]


def display_result(image_path, result_text):
    img = Image.open(image_path)
    img.thumbnail((300, 300))
    img = ImageTk.PhotoImage(img)

    panel = tk.Label(root, image=img)
    panel.image = img
    panel.grid(row=0, column=1, padx=10, pady=10)

    result_label.config(text=result_text)

def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        result_torch = classify_image_torch(file_path)
        result_keras = classify_image_keras(file_path)

        display_result(file_path, f"PyTorch: {result_torch}\nKeras: {result_keras}")



root = tk.Tk()
root.title("Image Classification App")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python script.py [input file]')
        sys.exit(1)

    create_booklet(sys.argv[1])
    root.mainloop()
