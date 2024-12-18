from django.shortcuts import render
from .forms import ImageUploadForm
from io import BytesIO
import base64
from .model import process_image
from django.core.exceptions import ValidationError
import os



# Allowed image file extensions for upload
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

# Allowed image file extensions for upload
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

def upload_image(request):
    # Check if the HTTP request method is POST (form submission)
    if request.method == 'POST':
        # Create an instance of ImageUploadForm with the submitted data and files
        form = ImageUploadForm(request.POST, request.FILES)
        
        # Check if the submitted form data is valid
        if form.is_valid():
            # Extract the uploaded image file from the cleaned form data
            input_image = form.cleaned_data['image']
            
            # Get the file extension of the uploaded image and convert it to lowercase
            file_extension = os.path.splitext(input_image.name)[1].lower()
            
            # Check if the file extension is in the allowed formats
            if file_extension not in ALLOWED_EXTENSIONS:
                # Return an error message if the file format is not allowed
                error_message = "Please upload an image in one of the allowed formats: .png, .jpg, .jpeg."
                return render(request, 'enhancer/upload.html', {'form': form, 'error_message': error_message})
            
            # Process the image (e.g., enhance, edit, or transform the uploaded image)
            output_image = process_image(input_image)
            
            # Convert the processed image to a base64-encoded string for embedding in HTML
            buffer = BytesIO()  # Create an in-memory buffer for processed image
            output_image.save(buffer, format="PNG")  # Save the image in PNG format to the buffer
            buffer.seek(0)  # Move the pointer to the start of the buffer
            
            # Encode the processed image data to base64
            img_str1 = base64.b64encode(buffer.getvalue()).decode()
            
            # Convert the input image to base64 directly
            input_image.seek(0)  # Ensure pointer is at the start of the uploaded file
            img_str2 = base64.b64encode(input_image.read()).decode()

            # Render the result page and pass both images (input and processed) to the template
            return render(request, 'enhancer/result.html', {'output_image': img_str1, 'input_image': img_str2})
    
    else:
        # If the request method is GET, display an empty form for image upload
        form = ImageUploadForm()
    
    # Render the upload form template with an empty or invalid form
    return render(request, 'enhancer/upload.html', {'form': form})


