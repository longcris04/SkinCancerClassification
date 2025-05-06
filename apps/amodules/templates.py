import streamlit as st
from PIL import Image
import os
from streamlit_option_menu import option_menu
from PIL import Image
import json
from streamlit_js_eval import streamlit_js_eval
from utils import *
from settings import MODEL_NAME_1, SIDEBAR_IMG_PATH

def run_home():
    st.markdown('<p style="font-size:30px; font-weight:bold;">AI SYSTEM FOR SUPPORTING SCREENING AND DIAGNOSIS OF SKIN CANCER IN VIETNAM</p>', unsafe_allow_html=True)

    for item in home_data:
        if item['title'] != "":
            st.markdown(f"<h3>{item['title']}</h3>", unsafe_allow_html=True)
        # Check if 'image' field is not empty and not an empty string
        if item['image'] != "":
            # Load and display the image
            image = Image.open(item['image'])
            st.image(image, use_container_width=True)
        
        # Check if 'text' field is not empty and not an empty string
        if item['text'] != "":
            # Display the text
            st.write(item['text'])

def run_overview():
    st.markdown('<p style="font-size:30px; font-weight:bold;">DATA AND MODELS</p>', unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=['Data', 'Models'],
        icons=['database', 'robot'],
        styles={
            "nav-link": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "400", "font-style": "normal"},
            "nav-link-selected": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "700", "font-style": "bold"}
        },
        orientation='horizontal'
    )
    if selected == "Data":
        labels = {
            'DLTW': 'Central Dermatology',
            'PAD-20': 'PAD-20',
            'No_disease': 'Healthy Skin',
            'Skin_cancer': 'Skin Cancer',
            'Other_disease': 'Other Skin Disease',
            'Rash': 'Basal Cell Carcinoma',
            'Psoriasis': 'Squamous Cell Carcinoma',
            'Melanoma': 'Melanoma',
        }
        
        def load_and_display_images(path, name,cols=3):
            """
            Load and display images from the specified path in a grid layout
            """
            if not os.path.exists(path):
                st.error(f"The path {path} does not exist")
                return

            # Get all image files
            image_files = [f for f in os.listdir(path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Display images in grid
            for i in range(0, len(image_files), cols):
                columns = st.columns(cols)
                for col, image_file in zip(columns, image_files[i:i+cols]):
                    with col:
                        try:
                            img_path = os.path.join(path, image_file)
                            image = Image.open(img_path)
                            image = image.resize((300, 300))
                            st.image(image, caption=name, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading image '{image_file}': {e}")
        
        
        # Dataset selection
        dataset = st.selectbox(
            "Select Dataset",
            options=["DLTW", "PAD-20"],
            format_func=lambda x: labels[x]
        )

        # First level categories based on dataset
        if dataset == "DLTW":
            first_level = st.selectbox(
                "Select Type",
                options=["No_disease", "Skin_cancer"],
                format_func=lambda x: labels[x]
            )
            
            # Second level for Skin_cancer
            if first_level == "Skin_cancer":
                second_level = st.selectbox(
                    "Select Cancer Type",
                    options=["Rash", "Psoriasis", "Melanoma"],
                    format_func=lambda x: labels[x]
                )
                # Display images for the selected cancer type
                image_path = os.path.join("dataset", dataset, "train/Skin_cancer", second_level)
                load_and_display_images(image_path, labels[second_level])
            else:
                # Display images for No_disease
                image_path = os.path.join("dataset", dataset, "train", first_level)
                load_and_display_images(image_path, labels[first_level])
                
        else:  # PAD-20
            category = st.selectbox(
                "Select Type",
                options=["Other_disease", "Skin_cancer"],
                format_func=lambda x: labels[x]
            )
            
            if category == "Skin_cancer":
                cancer_type = st.selectbox(
                    "Select Cancer Type",
                    options=["Rash", "Psoriasis", "Melanoma"],
                    format_func=lambda x: labels[x]
                )
                image_path = os.path.join("dataset", dataset, "train/Skin_cancer", cancer_type)
                load_and_display_images(image_path, labels[cancer_type])
            else:
                image_path = os.path.join("dataset", dataset, "train", category)
                load_and_display_images(image_path, labels[category])


    elif selected == "Models":
        st.markdown('<p style="font-size:25px; font-weight:bold;">Skin Cancer Screening Models</p>', unsafe_allow_html=True)
        items = get_sub_list(colabs_data, [0,1,2,3,4,5])
        # Display each item as a form with text and a hyperlink
        for idx, item in enumerate(items):
            with st.form(key=f"blog_form_{idx}_1"):
                st.markdown(f"### {item['name']}")
                st.markdown(f'<p style="font-size:15px; font-style:italic; color: #6a6a6a"; padding-top:5px; padding-bottom:5px>{item["note"]}</p>', unsafe_allow_html=True)
                submitted = st.form_submit_button("Training Code", )
                if submitted:
                    streamlit_js_eval(js_expressions=f'window.open("{item["link"]}", "_blank");', key=f"blog_js_eval_{idx}_1")

        st.markdown('<p style="font-size:25px; font-weight:bold;">Models for Differentiating Skin Cancer Types</p>', unsafe_allow_html=True)
        items = get_sub_list(colabs_data, [6,7])
        # Display each item as a form with text and a hyperlink
        for idx, item in enumerate(items):
            with st.form(key=f"blog_form_{idx}_2"):
                st.markdown(f"### {item['name']}")
                st.markdown(f'<p style="font-size:15px; font-style:italic; color: #6a6a6a"; padding-top:5px; padding-bottom:5px>{item["note"]}</p>', unsafe_allow_html=True)
                submitted = st.form_submit_button("Training Code", )
                if submitted:
                    streamlit_js_eval(js_expressions=f'window.open("{item["link"]}", "_blank");', key=f"blog_js_eval_{idx}_2")

def run_blogs():
    st.markdown('<p style="font-size:30px; font-weight:bold;">RESOURCES ON SKIN CANCER AND OTHER DISEASES</p>', unsafe_allow_html=True)
    # Display each item as a form with text and a hyperlink
    for idx, item in enumerate(blogs_data):
        with st.form(key=f"blog_form_{idx}_1"):
            st.markdown(f"### {item['name']}")
            st.markdown(f'<p style="font-size:16px; font-style:italic; color: #6a6a6a">Source: {item["source"]}</p>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Learn More", )
            if submitted:
                streamlit_js_eval(js_expressions=f'window.open("{item["link"]}", "_blank");', key=f"blog_js_eval_{idx}_1")

def run_prediction():
    st.markdown('<p style="font-size:30px; font-weight:bold;">SCREENING SUPPORT</p>', unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=['Use', 'User Guide'],
        icons=['card-image', 'blockquote-left'],
        styles={
            "nav-link": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "400", "font-style": "normal"},
            "nav-link-selected": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "700", "font-style": "bold"}
        },
        orientation='horizontal'
    )

    if selected == "Use":
        uploaded_file = st.file_uploader("Select an Image for Diagnosis", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            image = image.resize((300,300))
            st.image(image, caption='Uploaded Image', use_container_width=True)

        if 'cancer' not in st.session_state:
            st.session_state['cancer'] = False

        if uploaded_file is None:
            st.session_state['cancer'] = False

        # Create both buttons at the same level
        diagnose_button = st.button('Screen', disabled=uploaded_file is None)

        if diagnose_button:
            if uploaded_file is not None:                
                response = send_request_1(image, MODEL_NAME_1)
                try:
                    final_str, detail_str = get_description(response, 0)
                    st.write(final_str)
                    for i in detail_str:
                        st.write(f"- {i}%")
                    if response['Disease'] not in ['No_disease', 'Other_disease']:
                        st.session_state['cancer'] = True
                except:
                    st.write(f"An error occurred.")

        # Separate if condition for cancer_button
        if st.session_state['cancer']:
            cancer_button = st.button('Diagnose', disabled=uploaded_file is None)
            if cancer_button:
                if uploaded_file is not None:
                    response = send_request_2(image, MODEL_NAME_2) 
                    try:
                        final_str, detail_str = get_description(response, 1)
                        st.write(final_str)
                        for i in detail_str:
                            st.write(f"- {i}%.")
                    except:
                        st.write(f"An error occurred.")
    elif selected == "User Guide":
        # Guide content
        st.write("""
        The **Screening Support** feature helps you analyze skin images to predict the risk of skin-related diseases, particularly skin cancer.
        Please follow the steps below:
        """)
        
        # Usage steps
        st.markdown("""
        ### **1. Upload an Image**
        - Click on **"Select an Image for Diagnosis"** to upload a skin image from your computer.
        - Supported formats: `png`, `jpg`, `jpeg`.
        - After uploading, the interface will display the selected image for confirmation.
        """)
        
        st.markdown("""
        ### **2. Click the Screen Button**
        - Once the image is uploaded, click the **"Screen"** button to start the analysis.
        - The image will be sent to the server for processing and initial prediction.
        """)
        
        st.markdown("""
        ### **3. Initial Prediction Output**
        The prediction will be displayed in the following format:
        ```plaintext
        Your skin image indicates a risk of skin cancer.
        - Healthy Skin: 0.0%
        - Skin Cancer: 94.2%
        - Other Skin Disease: 5.8%
        ```
        - The example above indicates a **skin cancer** risk with a confidence level of **94.2%**.
        """)

        st.markdown("""
        ### **4. Display the Diagnose Button (if applicable)**
        - If the prediction indicates a risk of **skin cancer**, the interface will display a **"Diagnose"** button.
        - Click this button to send the image to the server again for a more detailed analysis.
        """)

        st.markdown("""
        ### **5. Detailed Prediction Output**
        After clicking the **Diagnose** button, the detailed result will be displayed as follows:
        ```plaintext
        Your skin image indicates a risk of Basal Cell Carcinoma.
        - Basal Cell Carcinoma: 80.2%.
        - Squamous Cell Carcinoma: 19.72%.
        - Melanoma: 0.09%.
        ```
        - The prediction above indicates a risk of **Basal Cell Carcinoma** with a confidence level of **80.2%**.
        """)

        st.markdown("""
        ### **Important Notes**
        - **Image Quality**: Ensure the image is clear and fully captures the skin area to be analyzed.
        - **Prediction Results**: This is only a support tool and does not replace a medical diagnosis. 
        Please consult a specialist doctor for specific advice.
        """)

def run_abouts():
    st.markdown('<p style="font-size:30px; font-weight:bold;">ABOUT US</p>', unsafe_allow_html=True)

    for item in abouts_data:        
        col1, col2 = st.columns([1, 3])
        with col1:
            try:
                # Load and display the image
                image = Image.open(item['avatar'])
                image = image.resize((300,300))
                st.image(image, use_container_width =True)
            except Exception as e:
                st.error(f"Error loading image '{item['avatar']}': {e}")
        
        with col2:
            # Show the name as a prefilled, non-editable text
            st.title(item['name'])
            st.text(item['detail'])

def run_accounts():
    st.markdown('<p style="font-size:30px; font-weight:bold;">ACCESS PERMISSIONS</p>', unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=['Add', 'Modify'],
        icons=['patch-plus-fill', 'diagram-2-fill'],
        styles={
            "nav-link": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "400", "font-style": "normal"},
            "nav-link-selected": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "700", "font-style": "bold"}
        },
        orientation='horizontal'
    )
    if selected == "Add":
        with st.form("signup_form",clear_on_submit=True):
            name = st.text_input("Name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            role = st.selectbox("Permission",("Admin", "User"))
            signup_clicked = st.form_submit_button("Add")
        
            if signup_clicked:
                if add_account(username, name, role, password):
                    st.success("Added successfully")
                else:
                    st.error("An error occurred")
    if selected == "Modify":
        list_username, list_names, list_roles, _, list_password, _, _, _, _ = load_accounts()
        list_username.append("none")
        selected_username = st.selectbox("Permission", list_username, index=len(list_username)-1)
        if selected_username.lower() != "none":
            index = list_username.index(selected_username)
            roles = ["Admin", "User"]
            role_index = roles.index(list_roles[index])
            with st.form("modify_form",clear_on_submit=True):
                st.text(selected_username)
                name = st.text_input("Name", placeholder=f"{list_names[index]}")
                password = st.text_input("Password", type="password", placeholder=f"{list_password[index]}")
                role = st.selectbox("Permission",roles, index=role_index)
                modify_clicked = st.form_submit_button("Modify")
            
                if modify_clicked:
                    if update_account(selected_username, name, role, password):
                        st.success("Modified successfully")
                    else:
                        st.error("An error occurred")

def run_page(role="admin"):
    if role.lower() == "user":
        with st.sidebar:
            image = Image.open(os.path.join(SIDEBAR_IMG_PATH))
            st.image(image, use_container_width=True)
            selected = option_menu(
                menu_title=None,
                options=['Home', 'Data and Models', 'Blogs', 'Screening and Diagnosis Support', 'About Us'],
                icons=['house-fill', 'bar-chart-line-fill', 'file-earmark-richtext-fill', 'aspect-ratio-fill', 'award-fill'],
                styles={
                    "nav-link": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "400", "font-style": "normal"},
                    "nav-link-selected": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "700", "font-style": "bold"}
                },
                )
    if role.lower() == "admin":
        with st.sidebar:
            image = Image.open(os.path.join(SIDEBAR_IMG_PATH))
            st.image(image, use_container_width=True)
            selected = option_menu(
                menu_title=None,
                options=['Home', 'Data and Models', 'Blogs', 'Screening and Diagnosis Support', 'About Us', "Accounts"],
                icons=['house-fill', 'bar-chart-line-fill', 'file-earmark-richtext-fill', 'aspect-ratio-fill', 'award-fill', "people-fill"],
                styles={
                    "nav-link": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "400", "font-style": "normal"},
                    "nav-link-selected": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "700", "font-style": "bold"}
                },
                )
    

    if selected == "Home":
        run_home()
    if selected == "Data and Models":
        run_overview()
    if selected == "Blogs":
        run_blogs()
    if selected == "Screening and Diagnosis Support":
        run_prediction()
    if selected == "About Us":
        run_abouts()
    if selected == "Accounts":
        run_accounts()

