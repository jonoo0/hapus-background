import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import io
import os
import sys

# Import allowed image extensions
IMAGE_EXTENSIONS = [
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".pfm",
]

# Setup constants
OUTPUT_FOLDER = 'output_images'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@st.cache_resource
def load_model():
    """Load the BiRefNet model with caching"""
    try:
        torch.set_float32_matmul_precision("high")
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet_lite", 
            trust_remote_code=True
        )
        return model.to(DEVICE)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise

def process_image(image, model):
    """Process a single image and remove its background"""
    # Define image transformation pipeline
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Prepare image
    image = image.convert("RGB")
    original_size = image.size
    input_tensor = transform_image(image).unsqueeze(0).to(DEVICE)
    
    # Process image
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        mask = transforms.ToPILImage()(pred).resize(original_size)
        
        # Apply mask to original image
        result = image.copy()
        result.putalpha(mask)
        
    return result

def sidebar_data():
    # Title and introduction
    st.sidebar.title("Kenapa?")
    st.sidebar.write("""
    Saya nggak akan upload ini kalau web hapus background yang di post di IMPHNEN dijadiin berbayar.
    """)
    st.sidebar.markdown("Menggunakan model dari [Birefnet](https://github.com/ZhengPeng7/BiRefNet) versi lite yang ukurannya 170mb hihihi.")

    # System information
    st.sidebar.markdown("### SysInfo()")
    st.sidebar.code(f"""
Python: {sys.version.split()[0]}
Torch: {torch.__version__}
Torchvision: {torch.__version__}
""")


def main():
    st.title("Web untuk hapus background")
    st.write("Upload gambar untuk diproses")

    sidebar_data()
    

    # File uploader with supported extensions
    uploaded_file = st.file_uploader(
        "Pilih gambar...",
        type=[ext.replace(".", "") for ext in IMAGE_EXTENSIONS]
    )
    
    if uploaded_file is not None:
        try:
            # Load the model (will use cached version if already loaded)
            model = load_model()
            
            # Load and display original image
            image = Image.open(io.BytesIO(uploaded_file.read()))
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Gambar asli")
                st.image(image, use_container_width=True)  # Updated parameter
            
            # Process image and display result
            with st.spinner("Removing background..."):
                result_image = process_image(image, model)
            
            with col2:
                st.subheader("Hasilnya")
                st.image(result_image, use_container_width=True)  # Updated parameter
            
            # Add download button for processed image
            buf = io.BytesIO()
            result_image.save(buf, format='PNG')
            st.download_button(
                label="Download hasil",
                data=buf.getvalue(),
                file_name="processed_image.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
