import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st

# Define the Autoencoder model
class ColorAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.down1 = nn.Conv2d(1, 64, 3, stride=2, padding=1)   # Adjusted padding to maintain size
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(128, 3, 3, stride=2, padding=1, output_padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoding layers
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        d3 = self.relu(self.down3(d2))
        d4 = self.relu(self.down4(d3))
        
        # Decoding layers
        u1 = self.relu(self.up1(d4))
        u2 = self.relu(self.up2(torch.cat((u1, d3), dim=1)))  # Concatenate u1 and d3
        u3 = self.relu(self.up3(torch.cat((u2, d2), dim=1)))  # Concatenate u2 and d2
        u4 = self.sigmoid(self.up4(torch.cat((u3, d1), dim=1)))  # Concatenate u3 and d1
        
        return u4

# Image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((256, 256)),                # Resize images
    transforms.ToTensor()                          # Convert to tensor
])

# Load the pre-trained model
model = ColorAutoEncoder()
model.load_state_dict(torch.load('color_autoencoder_model.pth', map_location=torch.device('cpu')))
model.eval()

# Streamlit app layout
st.title("Image Colorizer Using Autoencoder")
st.write("Upload a grayscale image and colorize it using the autoencoder model.")

# Image upload feature
uploaded_file = st.file_uploader("Choose a grayscale image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption='Uploaded Grayscale Image', use_column_width=True)

    # Preprocess the image
    input_image = transform(input_image).unsqueeze(0)  # Add batch dimension

    # Button for colorizing
    if st.button('Submit', key='submit_button'):
        # Colorize the image using the model
        with torch.no_grad():
            colorized_image = model(input_image).squeeze(0)

        # Convert the output back to a PIL image and display
        colorized_image_pil = transforms.ToPILImage()(colorized_image)
        st.image(colorized_image_pil, caption='Colorized Image', use_column_width=True)
