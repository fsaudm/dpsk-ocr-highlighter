from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch
import os

cache_path = "/home/jovyan/shared-dsrs/ai-models/hub"

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'


qc = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float
)

tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          trust_remote_code=True,
                                          cache_dir=cache_path,
                                         )
model = AutoModel.from_pretrained(model_name,
                                  trust_remote_code=True, 
                                  use_safetensors=True,
                                  device_map="auto",
                                  cache_dir=cache_path,
                                  torch_dtype=torch.float,
                                  # _attn_implementation='flash_attention_2',
                                  # quantization_config=qc, # 14,504 MB without quantization
                                 )

model = model.eval()


# --------------------------------------------------
# model = model.eval().cuda().to(torch.bfloat16)


# # PDF
# import fitz # PyMuPDF

# pdf_path = "sample.pdf"
# doc = fitz.open(pdf_path)

# output_folder = "pdf_pages"
# os.makedirs(output_folder, exist_ok=True)

# for page_num in range(len(doc)):
#     page = doc.load_page(page_num)
#     pix = page.get_pixmap()
#     output_image_path = os.path.join(output_folder, f"page_{page_num}.jpg")
#     pix.save(output_image_path)

# doc.close()

# --------------------------------------------------


# prompt = "<image>\nFree OCR."
# prompt = "<image>\Locate in what quadrant <|ref|>the book that is red is on the pencil<|/ref|> is in the image." #"<image>\n<|grounding|>Convert the document to markdown. "
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# prompt = "<image>\nDoes this page contain a table of content? Give me just text"# Respond only yes or no, depending on the answer."
# prompt = "<image>\nFree OCR."
prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = 'pdf_pages/page_2.jpg'
# image_file = 'trog 2.png'
output_path = 'ocr-results/'

# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

# res = model.infer(tokenizer,
#                   prompt=prompt,
#                   image_file=image_file,
#                   output_path=output_path,
                  
#                   # The Global Canvas (Low Res Context)
#                   # 1024 is usually fine, but 1280 adds slightly more global context
#                   base_size=1024, 
                  
#                   # The Detail Tiles (High Res Crops)
#                   # Standard is 640 or 1024.
#                   # Boosting this to 1536 or 2048 means each "tile" sees more detail.
#                   image_size=1536, 
                  
#                   # ENABLE DYNAMIC TILING
#                   crop_mode=True, 
#                   save_results=True,
#                  )



res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    
    # Global view: Keep at 1024 (Architecture Max)
    base_size=1024,
    
    # Crop view: Set to 1024 (Max Safe Limit)
    # Default Gundam is 640. This is 2.5x more pixels per crop.
    image_size=800,
    
    # Enable Dynamic Tiling
    crop_mode=True,
    
    save_results=True,
)

# Qwen3-VL-30B-A3B-Thinking