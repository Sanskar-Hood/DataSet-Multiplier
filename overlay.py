from PIL import Image

# Load the images
background = Image.open("inpainted_image-Photoroom.png")
overlay = Image.open("img.png")

# Resize overlay to match the background if needed
overlay = overlay.resize(background.size)

# Set transparency level (0 is fully transparent, 255 is fully opaque)
transparency = 110  # Adjust as needed
overlay.putalpha(transparency)

# Overlay images
combined = Image.alpha_composite(background.convert("RGBA"), overlay)

# Save the result
# combined.save("overlay_result.png")
combined.show()
