import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid

st.set_page_config(
    page_title="Monet Style Transfer",
    page_icon="🎨",
    layout="wide"
)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # padding to preserve spatial size
            nn.Conv2d(dim, dim, 3), # first conv layer
            nn.InstanceNorm2d(dim), # normalization 
            nn.ReLU(inplace=True), # activation with ReLU
            nn.ReflectionPad2d(1), # padding
            nn.Conv2d(dim, dim, 3), # second conv layer
            nn.InstanceNorm2d(dim), # normalization
        )

    def forward(self, x):
        return x + self.block(x)

# full ResNet generator --part 1
class ResnetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_blocks=9):
        super().__init__()

        # encoder (initial conv block)
        model = [
            nn.ReflectionPad2d(3), # padding
            nn.Conv2d(in_channels, 64, 7), # large kernel in order to capture more features
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        # encoder --part 2: downsampling layers
        in_features = 64
        for _ in range(2):
            out_features = in_features * 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # transformation layers (ResNet blocks)
        for _ in range(n_blocks): # n_block = 9
            model += [ResnetBlock(in_features)]

        # decoder: upsampling layers
        for _ in range(2):
            out_features = in_features // 2
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # output layer
        model += [
            nn.ReflectionPad2d(3), # padding
            nn.Conv2d(64, out_channels, 7), # out_channels = 3
            nn.Tanh() # for normalization the output to [-1, 1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# PatchGAN Discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # helper function to define a conv layer
        def block(in_feat, out_feat, norm=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_feat)) # apply normalization if needed
            layers.append(nn.LeakyReLU(0.2, inplace=True)) # ReLU activation
            return layers

        # the discriminator model as a series of conv layers
        self.model = nn.Sequential(
            *block(in_channels, 64, norm=False), # first layer without normalization (as mentionned on the original paper)
            *block(64, 128), 
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1) # final output layer with single channel output (real or fake)
        )

    def forward(self, x):
        return self.model(x)

# Image transformation for model input
def transform_image(image, size=256):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # converts to [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # converts to [-1, 1]
    ])
    
    image = transform(image)
    return image

# Function to denormalize images for display
def denormalize(tensor):
    tensor = tensor.clone().detach().cpu()
    tensor = tensor * 0.5 + 0.5  # Denormalize
    tensor = tensor.clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()

# Calculate loss for display
def calculate_cycle_loss(real_img, reconstructed_img):
    criterion = torch.nn.L1Loss()
    return criterion(real_img, reconstructed_img).item()

def calculate_adversarial_loss(discriminator, img, target_real):
    prediction = discriminator(img)
    target = torch.ones_like(prediction) if target_real else torch.zeros_like(prediction)
    criterion = torch.nn.MSELoss()
    return criterion(prediction, target).item()

# Load models
@st.cache_resource
def load_models(model_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    G_photo2monet = ResnetGenerator()
    G_monet2photo = ResnetGenerator()
    D_monet = PatchDiscriminator()
    D_photo = PatchDiscriminator()
    
    try:
        # Check if model files exist
        for name, path in model_paths.items():
            if not os.path.exists(path):
                st.error(f"Model file not found: {path}")
                return None, None, None, None, device
        
        # Load model weights
        G_photo2monet.load_state_dict(torch.load(model_paths["G_photo2monet"], map_location=device))
        G_monet2photo.load_state_dict(torch.load(model_paths["G_monet2photo"], map_location=device))
        D_monet.load_state_dict(torch.load(model_paths["D_monet"], map_location=device))
        D_photo.load_state_dict(torch.load(model_paths["D_photo"], map_location=device))
        
        # Set models to evaluation mode
        G_photo2monet.eval()
        G_monet2photo.eval()
        D_monet.eval()
        D_photo.eval()
        
        # Move models to device
        G_photo2monet.to(device)
        G_monet2photo.to(device)
        D_monet.to(device)
        D_photo.to(device)
        
        st.success("Models loaded successfully!")
        return G_photo2monet, G_monet2photo, D_monet, D_photo, device
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error(f"Details: {str(e)}")
        return None, None, None, None, device

# Main Streamlit app
def main():
    st.title("🎨 Monet Style Transfer with CycleGAN")
    st.write("Upload a photo and transform it into Monet's painting style using CycleGAN")
    
    # Sidebar for model paths and settings
    st.sidebar.header("Model Settings")
    
    # File uploader for models
    st.sidebar.subheader("Upload Your Model Files")
    uploaded_g_photo2monet = st.sidebar.file_uploader("Upload Generator Photo→Monet", type=["pth"])
    uploaded_g_monet2photo = st.sidebar.file_uploader("Upload Generator Monet→Photo", type=["pth"])
    uploaded_d_monet = st.sidebar.file_uploader("Upload Discriminator Monet", type=["pth"])
    uploaded_d_photo = st.sidebar.file_uploader("Upload Discriminator Photo", type=["pth"])
    
    # Option to use local files instead
    st.sidebar.subheader("Or Use Local Model Files")
    use_local = st.sidebar.checkbox("Use local model files instead")
    
    if use_local:
        model_paths = {
            "G_photo2monet": st.sidebar.text_input("Generator Photo→Monet path", "./output/kaggle/working/G_B2A.pth"),
            "G_monet2photo": st.sidebar.text_input("Generator Monet→Photo path", "./output/kaggle/working/G_A2B.pth"),
            "D_monet": st.sidebar.text_input("Discriminator Monet path", "./output/kaggle/working/D_A.pth"),
            "D_photo": st.sidebar.text_input("Discriminator Photo path", "./output/kaggle/working/D_B.pth")
        }
    else:
        # Save uploaded models to temporary files if provided
        model_paths = {}
        if uploaded_g_photo2monet:
            with open("temp_g_photo2monet.pth", "wb") as f:
                f.write(uploaded_g_photo2monet.getvalue())
            model_paths["G_photo2monet"] = "temp_g_photo2monet.pth"
        if uploaded_g_monet2photo:
            with open("temp_g_monet2photo.pth", "wb") as f:
                f.write(uploaded_g_monet2photo.getvalue())
            model_paths["G_monet2photo"] = "temp_g_monet2photo.pth"
        if uploaded_d_monet:
            with open("temp_d_monet.pth", "wb") as f:
                f.write(uploaded_d_monet.getvalue())
            model_paths["D_monet"] = "temp_d_monet.pth"
        if uploaded_d_photo:
            with open("temp_d_photo.pth", "wb") as f:
                f.write(uploaded_d_photo.getvalue())
            model_paths["D_photo"] = "temp_d_photo.pth"
    
    # Image processing options
    st.sidebar.subheader("Image Processing Options")
    image_size = st.sidebar.slider("Processing Image Size", 128, 512, 256, step=32)
    show_losses = st.sidebar.checkbox("Show Loss Metrics", value=True)
    
    # Load models
    load_models_button = st.sidebar.button("Load Models")
    models_loaded = False
    
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
    
    # Check if all required models are available
    all_models_available = False
    if use_local or not use_local and all(key in model_paths for key in ["G_photo2monet", "G_monet2photo", "D_monet", "D_photo"]):
        all_models_available = True
    
    if all_models_available and (load_models_button or st.session_state.models_loaded):
        G_photo2monet, G_monet2photo, D_monet, D_photo, device = load_models(model_paths)
        if G_photo2monet is not None:  # Check if models loaded successfully
            st.session_state.models_loaded = True
            models_loaded = True
            st.session_state.G_photo2monet = G_photo2monet
            st.session_state.G_monet2photo = G_monet2photo
            st.session_state.D_monet = D_monet
            st.session_state.D_photo = D_photo
            st.session_state.device = device
    elif not all_models_available:
        st.sidebar.warning("Please provide all model files before loading")
        st.session_state.models_loaded = False
    
    # Main content area
    st.subheader("Transform Your Photos into Monet-style Paintings")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image = None
    
    # Process button
    process_button = st.button("Transform to Monet Style")
    
    if image is not None:
        # Display original image
        st.subheader("Original Photo")
        st.image(image, use_container_width=False, caption="Original Image")
        
        # Only proceed if models are loaded and process button is clicked
        if models_loaded and process_button:
            with st.spinner("Applying Monet style..."):
                # Process image
                input_tensor = transform_image(image, size=image_size)
                input_tensor = input_tensor.to(st.session_state.device)
                
                with torch.no_grad():
                    # Generate Monet-style image
                    monet_style = st.session_state.G_photo2monet(input_tensor)
                    
                    # Reconstruct original image
                    reconstructed = st.session_state.G_monet2photo(monet_style)
                
                # Convert tensors to images for display
                monet_img = denormalize(monet_style)
                reconstructed_img = denormalize(reconstructed)
                
                # Display results
                st.subheader("Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(monet_img, use_container_width=False, caption="Monet Style")
                    
                    # Convert numpy array to PIL Image for download
                    monet_pil = Image.fromarray((monet_img * 255).astype(np.uint8))
                    monet_bytes = io.BytesIO()
                    monet_pil.save(monet_bytes, format='PNG')
                    
                    st.download_button(
                        label="Download Monet-Style Image",
                        data=monet_bytes.getvalue(),
                        file_name="monet_style.png",
                        mime="image/png"
                    )
                
                with col2:
                    st.image(reconstructed_img, use_container_width=False, caption="Reconstructed Photo")
                    
                    # Convert numpy array to PIL Image for download
                    reconstructed_pil = Image.fromarray((reconstructed_img * 255).astype(np.uint8))
                    reconstructed_bytes = io.BytesIO()
                    reconstructed_pil.save(reconstructed_bytes, format='PNG')
                    
                    st.download_button(
                        label="Download Reconstructed Image",
                        data=reconstructed_bytes.getvalue(),
                        file_name="reconstructed.png",
                        mime="image/png"
                    )
                
                # Calculate and display losses if enabled
                if show_losses:
                    with st.expander("Loss Metrics"):
                        cycle_loss = calculate_cycle_loss(input_tensor, reconstructed)
                        adv_loss_monet = calculate_adversarial_loss(st.session_state.D_monet, monet_style, True)
                        
                        st.write(f"Cycle Consistency Loss: {cycle_loss:.4f}")
                        st.write(f"Adversarial Loss (G: Photo→Monet): {adv_loss_monet:.4f}")
                        
                        # Plot losses
                        fig, ax = plt.subplots(figsize=(8, 4))
                        losses = [cycle_loss, adv_loss_monet]
                        labels = ['Cycle Loss', 'Adversarial Loss']
                        ax.bar(labels, losses)
                        ax.set_ylabel('Loss Value')
                        ax.set_title('CycleGAN Losses')
                        st.pyplot(fig)
                
                
        elif not models_loaded and process_button:
            st.warning("Please load the models first using the sidebar options.")
        elif not process_button and models_loaded:
            st.info("Click 'Transform to Monet Style' to process the image.")
    else:
        if process_button:
            st.warning("Please upload or select an image first.")

if __name__ == "__main__":
    main()
