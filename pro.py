
import streamlit as st
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import cv2
import base64
import time

# Load your trained model
def load_model():
    model = tf.keras.models.load_model('C:\\Users\\God\\PycharmProjects\\pythonProject2\\CNN_Project\\CNN_AI_REAL.h5')
    return model

model = load_model()



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    add_bg_from_local('Maroon and Red Modern Sales Report Presentation.png')


# st.markdown("""
#     <style>
#     div[class*="stTabs"] button {
#         font-size: 20px;  /* Increase text size */
#         font-weight: bold;
#         color: white; /* Text color */
#
#     }
#     </style>
#     """, unsafe_allow_html=True)


st.write(" ")
# st.write(" ")
# st.write(" ")

tab_1,tab_2,tab_3,tab_4 = st.tabs([":gray-background[ 🏠 **HOME**]",":gray-background[ 💬  **INSIGHTS**]",":gray-background[ 💻  **PREDICTION**]",":gray-background[  💾  **TAKEAWAYS**]"])

with tab_1:

    # Captivating Title

    st.header(":violet[**Welcome to the AI vs. Reality Image Classifier!**]", divider='gray')




    # Introduction with Markdown
    st.write("""
    This interactive app helps you distinguish between real portraits and AI-generated images. Upload an image or use your webcam for real-time classification.
    
    
    Use the navigation menu on the top to explore the different sections of the app:
    - **HOME**: You're here! This is the home page.
    - **INSIGHTS**: Information about the model.
    - **PREDICTION**: Make predictions using the model.
    - **TAKEAWAYS**: Summary and conclusion of the project.
    """)

    st.write("""
               Test the app with different images and witness the AI vs. Reality Image Classifier's accuracy.
               """)
    # st.header(' ',divider='gray')



    st.title("Introducing Sora — OpenAI’s text-to-video model")

    st.write("Sora can create videos of up to 60 seconds featuring highly detailed scenes, complex camera motion, and multiple characters with vibrant emotions.")

    # Embed a YouTube video
    youtube_url = "https://youtu.be/HK6y8DAPN_0?si=Ne_8ZTMLRdYid1nx"
    st.video(youtube_url)

    # if __name__ == "__main__":
    #     st.run()

    # Disclaimer with Markdown
    st.write("""
            ### Note:
            This application is for demonstration purposes only and may not guarantee perfect image classification.
            """)
    # st.header(' ',divider='gray')

    linkedin_logo = "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"
    # github_logo = "https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg"

    linkedin_url = "www.linkedin.com/in/sreehari8274"
    # github_url = "https://github.com/Sreehari82"

    st.markdown(
        f"""
        <style>
        .icon {{
            width: 24px;
            height: 24px;
            margin: 0 5px;
            vertical-align: middle;
        }}
        </style>
        <div>
            <span style="color: red;">CREATED BY:</span> <span>SREEHARI S</span>
            <a href="{linkedin_url}" target="_blank"><img src="{linkedin_logo}" class="icon"/></a>

        </div>
        """,
        unsafe_allow_html=True)

with tab_2:
    tab_a,tab_b = st.tabs(["**OVERVIEW**","**TECHNOLOGIES**"])




    with tab_a:
        # Project Overview with Bullet Point
        st.header('PROJECT IN DETAIL')
        st.write("""
        - This project leverages a Convolutional Neural Network (CNN) trained to classify images as real portraits or AI-generated. The CNN model is designed to analyze intricate patterns and features in images, distinguishing between authentic human portraits and those created by AI algorithms.

        - Deep learning techniques ensure accurate classification results by enabling the model to learn and adapt from a large dataset of both real and AI-generated images. The neural network architecture includes multiple layers that process image data, extracting meaningful features that differentiate between the two categories.

        - The trained model has undergone rigorous testing and validation to achieve high accuracy in image classification tasks. It forms the core component of this application, providing users with a reliable tool to explore the boundaries between human creativity and artificial intelligence in visual art.

        - Explore the capabilities of this AI vs REALITY Image Classifier by uploading your own images or using the webcam feature for real-time analysis.
        """)

        # User-Friendly Instructions
        st.header('INSTRUCTIONS')
        st.write("""
        1. **Upload Image:** Click "Upload Image" and select a JPEG or PNG image file.
        2. **Use Webcam:** Click "Use Webcam" to capture an image for real-time classification using your camera.
        3. **See Results:** The model will analyze the image and display whether it's a real portrait or AI-generated.
        """)

    with tab_b:
        # Technologies Breakdown
        st.header('TECHNOLOGIES BREAKDOWN')
        st.write("""
        - **Python:** The programming language powering this application's development. Python's versatility and extensive libraries make it ideal for implementing complex machine learning models and web applications.
        - **TensorFlow & Keras:** Deep learning libraries used to build and train the CNN model. TensorFlow provides a powerful framework for creating and training neural networks, while Keras offers a user-friendly interface to build and customize deep learning models.
        - **Streamlit:** The framework enabling the creation of this interactive web app.Streamlit simplifies the process of building and deploying data-driven applications, allowing seamless integration of machine learning models with user-friendly interfaces.
        - **OpenCV:** A library facilitating computer vision tasks, including webcam image capture.OpenCV provides robust tools for image and video processing, enabling functionalities such as real-time image analysis and manipulation within the application.
        """)

    # # Team Recognition (if applicable)
    # st.header('The Creators')
    # st.write(f"""
    # This project was developed by {', '.join([your_name, *collaborators])} as part of {project_name or course_name}.
    # """)

    # with tab_c:
        # Encouraging Conclusion


        # st.write("CREATED BY : ", ":blue[SREEHARI S]")



#<a href="{github_url}" target="_blank"><img src="{github_logo}" class="icon"/></a>

with tab_3:

    st.title("UPLOAD PHOTO HERE!")
    # st.write("Upload an image or use your webcam to capture an image")

    # tab1, tab2 = st.tabs([":gray-background[ 📤  **UPLOAD IMAGE** ]",":gray-background[ 📷  **WEBCAM**]" ])
    tab1, tab2 = st.tabs([" 📤  **UPLOAD IMAGE** "," 📷  **WEBCAM**" ])


    # Define the classify_image function before using it
    def classify_image(image):
        st.image(image, caption='Captured Image.', use_column_width=True)
        with st.spinner("Classifying..."):

            # Preprocess the image
            img_array = np.array(image)
            if img_array.shape[-1] == 4:  # Ensure the image has 3 channels (RGB)
                img_array = rgb2gray(img_array)  # Convert to grayscale
            img_resized = resize(img_array, (100, 100, 1))
            img_resized = np.expand_dims(img_resized, axis=0)

            # Make prediction
            prediction = model.predict(img_resized)
            prediction_class = np.argmax(prediction, axis=1)

            # Display prediction result
            if prediction_class == 0:
                st.write("## IMAGE IS **AI-GENERATED**.")
            else:
                st.write("## IMAGE IS **REAL PORTRAIT** OF A PERSON.")




    with tab1:
        # File uploader for image upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        pred = st.button("PREDICT")

        if pred:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                classify_image(image)




    with tab2:
        # Webcam capture
        if st.button('📸 CAPTURE IMAGE AND PREDICT', key='capture_image'):

            with st.spinner("Initializing webcam..."):
                with st.spinner("Please wait for 5 seconds to capture the image..."):
                    time.sleep(5)  # Wait for 5 seconds
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("Error: Could not open webcam.")
                    else:
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(frame)
                            classify_image(image)
                        else:
                            st.error("Failed to capture image from webcam.")
                        cap.release()

        st.write("Make sure to allow webcam access when prompted.")

with tab_4:
    def show_conclusion():
        # st.header(":violet:[ Conclusion]")

        st.header("Summary of the Project")
        st.write("""
        This project focuses on developing an AI vs REALITY Image Classifier using a Convolutional Neural Network (CNN) trained with deep learning techniques. The goal is to accurately distinguish between real human portraits and AI-generated images. Leveraging advanced machine learning algorithms and computer vision tools, the classifier provides a practical tool for exploring the boundaries between human creativity and artificial intelligence in visual art.
        """)

        st.header("Key Findings")
        st.write("""
        - **Accurate Classification**: The CNN model effectively distinguishes between real portraits and AI-generated images, showcasing the capabilities of modern AI in visual content analysis.
        - **User Interaction**: The Streamlit-based interface allows users to upload images or use their webcam for real-time classification, enhancing user engagement and accessibility.
        - **Technological Integration**: Python, TensorFlow, Keras, Streamlit, and OpenCV are integrated to deliver robust image analysis capabilities, including webcam image capture and processing.
        - **Educational Value**: The project serves as a demonstration of applying machine learning and computer vision technologies to differentiate between human-created and AI-generated visual content, educating users on AI capabilities.
        - **Real-World Applications**: Potential applications include media authentication, artistic creation, and understanding AI's evolving capabilities in generating realistic human-like images.
        - **Continuous Improvement**: Ongoing updates and enhancements ensure the model's accuracy and usability, reflecting advancements in machine learning and user experience design.
        """)

        st.header("Next Steps")
        st.write("""
        - **Enhanced Features**: Explore adding more advanced features such as style transfer or image enhancement.
        - **Performance Optimization**: Continuously optimize the model for faster processing and improved accuracy.
        - **Community Engagement**: Foster a community around the project for feedback, collaboration, and further development.
        """)

        st.markdown("- [Link to my colab notebook](https://colab.research.google.com/drive/15v8xjF2T5NmSpsISvZ17bPG3Ue9X_fcF)")
        # st.markdown("- [Demo Video](https://youtu.be/yourvideolink)")


    show_conclusion()



