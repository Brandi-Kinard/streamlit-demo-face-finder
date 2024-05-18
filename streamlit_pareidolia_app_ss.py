import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# Function for detecting faces in an image.
def detectFaceOpenCVDnn(net, frame):
    # Create a blob from the image and apply some pre-processing.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set the blob as input to the model.
    net.setInput(blob)
    # Get Detections.
    detections = net.forward()
    return detections

# Function for annotating the image with bounding boxes for each detected face.
def process_detections(frame, detections, conf_threshold=0.5):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # Loop over all detections and draw bounding boxes around each face.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            bb_line_thickness = max(1, int(round(frame_h / 200)))
            # Draw bounding boxes around detected faces.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_8)
    return frame, bboxes

# Function to load the DNN model.
# @st.cache(allow_output_mutation=True)
def load_model():
    # modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    # configFile = "deploy.prototxt"
    modelFile = os.path.join("model", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    configFile = os.path.join("model", "deploy.prototxt")
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

net = load_model()

# ------------------------------------------------
# ------------------------------------------------

# Create application title and introduction.
st.title("Face Finder")
#st.subheader("Uncovering Hidden Faces in Everyday Objects")
st.markdown("""
<div style="font-size: 20px;">
    Welcome to Face Finder, an interactive demo that reveals the hidden faces our minds often perceive in ordinary 
    scenes. Using advanced AI, we delve into the fascinating world of facial illusions, raising awareness about the 
    quirks of human perception and the biases present in facial recognition technologies. Explore, learn, and reflect 
    on the implications of AI in our daily lives.
</div><br />
""", unsafe_allow_html=True)

st.image("Faces.png", caption=f"Can you spot a face in these images?")

#------------------------------------------------
#------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Upload and Process Your Image", "What is Pareidolia?", "Research Insights"])

with tab1:
    # Interactive Image Upload Section
    st.header("Upload Your Image")
    # st.write("Drag and drop an image or click to browse. Try to find faces in clouds, rocks, trees, or even your morning toast!")
    st.markdown("""
    <div style="font-size: 20px;">
        <b>Drag and drop an image or click to browse.</b> Try to find faces in clouds, rocks, trees, or even your morning toast!
    </div>
    """, unsafe_allow_html=True)

    img_file_buffer = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

    # Display sample images for users to click and upload
    st.subheader("Or try one of these fun examples:")
    example_images = ["aw_man_is_it_monday_again.png", "pepper_panic.png", "milk_jug_man.png", "sad_face.png",
                      "one_happy_pickle.png"]  # Example file names
    example_captions = ["Is it Monday again? 😒", "Pepper Panic 🫑", "Milk jug man? 🥛", "Sad Socket Face 🔌",
                        "Lone happy pickle 😀"]
    for img, caption in zip(example_images, example_captions):
        if st.button(caption):
            img_file_buffer = open(img, "rb")

    if img_file_buffer is not None:
        # Read the file and convert it to OpenCV Image.
        raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        # Loads image in a BGR channel order.
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        # Create placeholders to display input and output images.
        placeholders = st.columns(2)
        # Display Input image in the first placeholder.
        placeholders[0].image(image, channels='BGR', caption="Input Image")
        # Add a blank space
        st.text("")
        # Add explanatory text about the confidence threshold
        st.markdown("""
            ### Confidence Threshold
            Adjust the slider to set the confidence level for face detection. 
            A higher value means stricter detection, while a lower value allows for more detections.
            """)
        # Create a Slider and get the threshold from the slider.
        conf_threshold = st.slider("Set Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)
        # Call the face detection model to detect faces in the image.
        detections = detectFaceOpenCVDnn(net, image)
        # Process the detections based on the current confidence threshold.
        out_image, _ = process_detections(image, detections, conf_threshold=conf_threshold)
        # Display Detected faces.
        placeholders[1].image(out_image, channels='BGR', caption="Output Image with Detected Faces")
        # Convert OpenCV image to PIL.
        out_image = Image.fromarray(out_image[:, :, ::-1])
        # Create a link for downloading the output file.
        #st.markdown(get_image_download_link(out_image, "face_output.jpg", 'Download Output Image'), unsafe_allow_html=True)
        # Add optional text to nudge users about the download option
        st.markdown("### Want to save your processed image?")

        # Convert PIL image to bytes
        img_bytes = BytesIO()
        out_image.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()

        # Create a download button
        st.download_button(
            label="Download Output Image",
            data=img_bytes,
            file_name="face_output.jpg",
            mime="image/jpeg"
        )


    # Add some space with a blank line
    # st.text("")
    # st.divider()



with tab2:
    # Fun Facts and Trivia Section
    st.header("What is Pareidolia?")
    # st.write("Did you know? Pareidolia is the tendency to perceive a specific, often meaningful image in a
    # random or ambiguous visual pattern. It’s how we sometimes see faces in the moon or animals in the clouds!")
    st.markdown("""
    <div style="font-size: 20px;">
       <b>Pareidolia (pair-ee-DOH-lee-uh)</b> is the tendency to perceive a specific, often meaningful image in a random or ambiguous 
       visual pattern. Common examples include seeing faces in everyday objects like clouds or rocks.
    </div>
   """, unsafe_allow_html=True)

    # Add some space with a blank line
    st.text("")

    # Fun Facts and Trivia Section
    st.markdown("""
        <div style="font-size: 20px;">
           <div style="font-size: 24px;"><b>Psychological Basis 🧠</b></div>
           This phenomenon is believed to arise from the human brain's predisposition to recognize faces, 
           which is crucial for social interactions.
           <br /><br />
           <div style="font-size: 24px;"><b>AI and Pareidolia 🤖</b></div>
           In AI, this can be explored to understand how machines perceive and interpret visual data, highlighting the 
           challenges and limitations of current technologies.
        </div>
    """, unsafe_allow_html=True)

    # Add some space with a blank line
    st.text("")


with tab3:
    # Exploring Machine Pareidolia Research Section
    st.header("Bridging Human and Machine Perception: Exploring AI Detection of Pareidolia and Bias in Facial Recognition Technologies")
    #st.image("face-pareidolia.png", caption=f"Image source: researchgate.net")

    # Abstract
    st.subheader("Abstract 📄")
    st.markdown("""
    <div style="font-size: 20px;">
        This project explores the phenomenon of pareidolia using a web-based computer vision app. 
        By leveraging OpenCV and Streamlit, we investigate whether AI can detect faces in images where humans perceive 
        familiar shapes, bridging the gap between human and machine vision.
    </div><br />
    """, unsafe_allow_html=True)

    # Introduction
    st.subheader("Introduction 📖")
    st.markdown("""
    <div style="font-size: 20px;">
        Pareidolia is a psychological phenomenon where people see patterns, such as faces, in random stimuli. 
        This tendency reveals a lot about human perception and cognition. 
        As AI technology evolves, it is crucial to understand if machines can develop similar perceptual capabilities. 
        This project aims to bridge the gap between human and machine vision by exploring how AI detects faces 
        in various images.
        <br /><br />
        <div style="font-size: 22px;"><b>Connecting Pareidolia to AI Bias</b></div>
        AI systems, especially facial recognition technologies, can exhibit biases based on the data they are trained on. 
        Just as humans can mistakenly see faces in random patterns, AI can also make errors, often influenced by biases in the training data. 
        Understanding these biases is essential for developing fair and equitable AI systems.
    </div><br />
    """, unsafe_allow_html=True)

    # Literature Review
    st.subheader("Literature Review 📚")
    st.markdown("""
    <div style="font-size: 20px;">
        Several studies have explored pareidolia in human perception. 
        For example, <a href="https://www.sciencedirect.com/science/article/abs/pii/S0010945214000288">Liu et al. (2014)</a> 
        investigated how the human brain processes faces seen in everyday objects, revealing neural and behavioral 
        correlates of face pareidolia. More recent work by 
        <a href="https://royalsocietypublishing.org/doi/10.1098/rspb.2021.0966">Alais et al. (2021)</a>  
        examined the shared mechanisms for facial expression in human faces and face pareidolia. 
        In the realm of AI, researchers like <a href="https://www.nature.com/articles/s41597-019-0052-3">Chang et al. (2019)
        </a> have developed models that mimic human visual perception, but few have specifically addressed pareidolia. 
        The scarcity of specific studies on pareidolia in AI models underscores the novelty and importance of this 
        research direction.
    </div><br />
    """, unsafe_allow_html=True)

    # AI Biases in Facial Recognition
    st.subheader("AI Biases in Facial Recognition ⚖️")
    st.markdown("""
    <div style="font-size: 20px;">
        AI systems, particularly facial recognition technologies, can exhibit biases based on the data they are trained on. 
        This has significant implications, especially concerning fairness and discrimination. 
        <a href="https://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf">Buolamwini and Gebru (2018)</a> demonstrated that many facial recognition systems have higher error rates for darker-skinned and female faces. Understanding these biases is crucial for developing fair and equitable AI systems that serve all individuals without discrimination.
    </div><br />
    """, unsafe_allow_html=True)

    # Joy Buolamwini’s Research
    st.subheader("Joy Buolamwini’s Research 🧠")
    st.markdown("""
    <div style="font-size: 20px;">
        MIT researcher Joy Buolamwini has extensively studied these biases, revealing that many facial recognition 
        systems have higher error rates for darker-skinned and female faces. Her work emphasizes the importance of 
        diversity in training data and the ethical deployment of AI technologies. 
        You can read more about her research <a href="https://www.media.mit.edu/projects/gender-shades/overview/">here</a>.
        <br /><br />
        <div style="font-size: 22px;"><b>Real-World Implications of AI Bias</b></div> 
        AI biases in facial recognition technologies can lead to significant real-world consequences:
        <br />
        - <b>Wrongful Arrests:</b> There have been several cases where individuals were wrongfully arrested due to 
        misidentifications by facial recognition systems. For example, Robert Williams, a black man, was wrongfully 
        arrested in Detroit after being misidentified by a facial recognition system (<a href="https://shorturl.at/Ae0s5">source</a>).
        <br />
        - <b>Discrimination:</b> Biases in AI can perpetuate and amplify existing social inequalities, leading to 
        unfair treatment of marginalized communities.
        <br />
        Understanding and addressing these biases is crucial for developing fair and equitable AI systems.
    </div><br />
    """, unsafe_allow_html=True)

    # Methodology
    st.subheader("Methodology 🔬")
    st.markdown("""
    <div style="font-size: 20px;">
        We use the OpenCV library for face detection and Streamlit for the web interface. 
        The DNN model, "res10_300x300_ssd_iter_140000_fp16.caffemodel," is employed for detecting faces. 
        Users upload images where they perceive faces, and the app processes these images to identify and highlight 
        faces using green bounding boxes.
    </div><br />
    """, unsafe_allow_html=True)

    # Experimental Setup
    st.subheader("Experimental Setup 🧪")
    st.markdown("""
    <div style="font-size: 20px;">
        Different types of images were tested, including natural scenes (clouds, rocks) and everyday objects (toasts, cars). 
        The confidence threshold for detection is adjustable, allowing users to study the sensitivity of the AI 
        in recognizing patterns. Sample images are provided to encourage user interaction. 
        As part of our scientific experiment, we’re investigating whether machines can experience pareidolia just like humans. 
        Users upload images where they see faces and let’s see if our AI detects them too.
    </div><br />
    """, unsafe_allow_html=True)

    # Results
    st.subheader("Results 📊")
    st.markdown("""
    <div style="font-size: 20px;">
        The app successfully detects faces in various images, demonstrating the widespread occurrence of pareidolia. 
        The following are examples of AI detection results:
    </div><br />
    """, unsafe_allow_html=True)

    # Showcase different results (in RGB)
    test_images = ["cone_2.png", "jeep_anger.png", "gate_green.png"]  # Replace with paths to your test images
    for img in test_images:
        # Read the file and convert it to PIL Image
        pil_image = Image.open(img)

        # Convert PIL Image to numpy array (OpenCV format, but still RGB)
        test_image = np.array(pil_image)

        # Convert RGB to BGR for OpenCV processing
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

        detections = detectFaceOpenCVDnn(net, test_image)
        out_test_image, _ = process_detections(test_image, detections, conf_threshold=0.23)

        # Convert BGR back to RGB for displaying in Streamlit
        out_test_image = cv2.cvtColor(out_test_image, cv2.COLOR_BGR2RGB)

        st.image(out_test_image, caption=f"AI detection on {img.split('_')[0]}")

    # Discussion
    st.subheader("Discussion 💬")
    st.markdown("""
    <div style="font-size: 20px;">
        The findings suggest that AI can be trained to recognize patterns similarly to humans, 
        though there are limitations in accuracy. The adjustable confidence threshold allows for studying how 
        sensitive the AI is in detecting faces, which can be influenced by various factors like image quality and complexity. 
        Through this experiment, we're exploring the fascinating intersection of human perception and artificial intelligence. 
        Future work can explore more complex models and diverse datasets to enhance AI's ability to mimic human perception. 
        Can machines develop pareidolia, or is it a uniquely human trait? Let's find out together!
    </div><br />
    """, unsafe_allow_html=True)

    # Conclusion
    st.subheader("Conclusion 🏁")
    st.markdown("""
    <div style="font-size: 20px;">
        Pareidolia highlights the human tendency to find patterns in randomness, revealing interesting aspects of 
        human perception and cognition. This project bridges human and machine vision, offering insights into 
        AI's development. The results indicate that while AI can mimic some aspects of human perception, it still has 
        a long way to go in achieving the nuanced understanding that humans possess.
    </div><br />
    """, unsafe_allow_html=True)

    # References
    st.subheader("References 📚")
    st.markdown("""
    <div style="font-size: 20px;">
        <li style="font-size: 20px;">
            <b>Alais, D., et al. (2021)</b>. A shared mechanism for facial expression in human faces and face pareidolia. 
            <b>Proceedings of the Royal Society B: Biological Sciences</b>, 288(1954), 20210966. 
            <a href="https://royalsocietypublishing.org/doi/10.1098/rspb.2021.0966">royalsocietypublishing.org</a>
        </li>
        <li style="font-size: 20px;">
            <b>Buolamwini, J., & Gebru, T. (2018)</b>. Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification. 
            <b>Proceedings of Machine Learning Research</b>, 81, 1-15. 
            <a href="https://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf">proceedings.mlr.press</a>
        </li>
        <li style="font-size: 20px;">
            <b>Chang, N., et al. (2019)</b>. BOLD5000, a public fMRI dataset while viewing 5000 visual images. 
            <b>Scientific Data</b>, 6(1), 49.
            <a href="https://www.nature.com/articles/s41597-019-0052-3">nature.com</a>
        </li>
        <li style="font-size: 20px;">
            <b>Liu, J., Li, J., Feng, L., Li, L., Tian, J., & Lee, K. (2014).</b> Seeing Jesus in toast: Neural and behavioral 
            correlates of face pareidolia. <b>Cortex</b>, 53, 60-77. 
            <a href="https://www.sciencedirect.com/science/article/abs/pii/S0010945214000288">sciencedirect.com</a>
        </li>
    </div><br />
    """, unsafe_allow_html=True)

#------------------------------------------------
#------------------------------------------------

st.markdown("---")
st.markdown("""
### 👋 Get in Touch

Hi there! I'm Brandi Kinard, a passionate Full Stack Designer and AI Engineer. I hope you enjoyed exploring the hidden faces in everyday objects with me. If you'd like to connect, feel free to reach out through my LinkedIn or check out more of my projects on GitHub.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/brandi-kinard/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Brandi-Kinard)
""")
