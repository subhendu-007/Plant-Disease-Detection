import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element
st.sidebar.title("Plant Disease Recognition")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

if app_mode == "Home":
    st.title("Plant Disease Recognition System")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown(
        """
        Welcome to the Plant Disease Recognition System! 🌿🔍

        Our mission is to help identify plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

        **How It Works:**
        1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.

        **Why Choose Us?**
        - **Accuracy:** Utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for a seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

        **Get Started:**
        Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

        **About Us:**
        Learn more about the project, our team, and our goals on the **About** page.
        """
    )

elif app_mode == "About":
    st.title("About")
    st.markdown(
        """
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on Kaggle.
        This dataset consists of about 87K RGB images of healthy and diseased crop leaves, which are categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure.
        A new directory containing 33 test images is created later for prediction purposes.
        
        **Content:**
        - **Train:** 70,295 images
        - **Test:** 33 images
        - **Validation:** 17,572 images
        """
    )

elif app_mode == "Disease Recognition":
    st.title("Disease Recognition")
    st.write("Upload an image of a plant with suspected diseases:")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        if test_image is not None:
            st.write("Our Prediction:")
            result_index = model_prediction(test_image)
            class_name = [
                'Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy Apple',
                'Healthy Blueberry', 'Powdery Mildew on Cherry', 'Healthy Cherry', 
                'Cercospora Leaf Spot on Corn', 'Common Rust on Corn', 'Northern Leaf Blight on Corn', 
                'Healthy Corn', 'Black Rot on Grape', 'Esca (Black Measles) on Grape', 
                'Leaf Blight (Isariopsis Leaf Spot) on Grape', 'Healthy Grape', 
                'Huanglongbing (Citrus Greening) on Orange', 'Bacterial Spot on Peach',
                'Healthy Peach', 'Bacterial Spot on Bell Pepper', 'Healthy Bell Pepper', 
                'Early Blight on Potato', 'Late Blight on Potato', 'Healthy Potato', 
                'Healthy Raspberry', 'Healthy Soybean', 'Powdery Mildew on Squash', 
                'Leaf Scorch on Strawberry', 'Healthy Strawberry', 'Bacterial Spot on Tomato', 
                'Early Blight on Tomato', 'Late Blight on Tomato', 'Leaf Mold on Tomato', 
                'Septoria Leaf Spot on Tomato', 'Two-spotted Spider Mite on Tomato', 
                'Target Spot on Tomato', 'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus',
                'Healthy Tomato'
            ]
            st.success("The predicted disease is: {}".format(class_name[result_index]))
        else:
            st.error("Please upload an image first.")