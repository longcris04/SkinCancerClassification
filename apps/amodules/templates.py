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
    st.markdown('<p style="font-size:30px; font-weight:bold;">HỆ THỐNG ỨNG DỤNG TRÍ TUỆ NHÂN TẠO TRONG HỖ TRỢ SÀNG LỌC, CHẨN ĐOÁN MỘT SỐ BỆNH UNG THƯ DA TẠI VIỆT NAM</p>', unsafe_allow_html=True)

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
    st.markdown('<p style="font-size:30px; font-weight:bold;">DỮ LIỆU VÀ MÔ HÌNH</p>', unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=['Dữ liệu', 'Mô hình'],
        icons=['database', 'robot'],
        styles={
            "nav-link": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "400", "font-style": "normal"},
            "nav-link-selected": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "700", "font-style": "bold"}
        },
        orientation='horizontal'
    )
    if selected == "Dữ liệu":
        labels = {
            'DLTW': 'Da liễu trung ương',
            'PAD-20': 'PAD-20',
            'Khong_benh': 'Da không bệnh',
            'Ung_thu': 'Ung thư da',
            'Benh_khac': 'Bệnh da khác',
            'Day': 'Ung thư tế bào đáy',
            'Vay': 'Ung thư tế bào vảy',
            'Hac_to': 'Ung thư hắc tố',
        }
        
        def load_and_display_images(path, name,cols=3):
            """
            Load and display images from the specified path in a grid layout
            """
            if not os.path.exists(path):
                st.error(f"Đường dẫn {path} không tồn tại")
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
                            st.error(f"Lỗi khi tải ảnh '{image_file}': {e}")
        
        
        # Dataset selection
        dataset = st.selectbox(
            "Chọn Dataset",
            options=["DLTW", "PAD-20"],
            format_func=lambda x: labels[x]
        )

        # First level categories based on dataset
        if dataset == "DLTW":
            first_level = st.selectbox(
                "Chọn loại",
                options=["Khong_benh", "Ung_thu"],
                format_func=lambda x: labels[x]
            )
            
            # Second level for Ung_thu
            if first_level == "Ung_thu":
                second_level = st.selectbox(
                    "Chọn loại ung thư",
                    options=["Day", "Vay", "Hac_to"],
                    format_func=lambda x: labels[x]
                )
                # Display images for the selected cancer type
                image_path = os.path.join("dataset", dataset, "train/Ung_thu", second_level)
                load_and_display_images(image_path, labels[second_level])
            else:
                # Display images for Khong_benh
                image_path = os.path.join("dataset", dataset, "train", first_level)
                load_and_display_images(image_path, labels[first_level])
                
        else:  # PAD-20
            category = st.selectbox(
                "Chọn loại",
                options=["Benh_khac", "Ung_thu"],
                format_func=lambda x: labels[x]
            )
            
            if category == "Ung_thu":
                cancer_type = st.selectbox(
                    "Chọn loại ung thư",
                    options=["Day", "Vay", "Hac_to"],
                    format_func=lambda x: labels[x]
                )
                image_path = os.path.join("dataset", dataset, "train/Ung_thu", cancer_type)
                load_and_display_images(image_path, labels[cancer_type])
            else:
                image_path = os.path.join("dataset", dataset, "train", category)
                load_and_display_images(image_path, labels[category])


    elif selected == "Mô hình":
        st.markdown('<p style="font-size:25px; font-weight:bold;">Mô hình hỗ trợ sàng lọc bệnh ung thư</p>', unsafe_allow_html=True)
        items = get_sub_list(colabs_data, [0,1,2,3,4,5])
        # Display each item as a form with text and a hyperlink
        for idx, item in enumerate(items):
            with st.form(key=f"blog_form_{idx}_1"):
                st.markdown(f"### {item['name']}")
                st.markdown(f'<p style="font-size:15px; font-style:italic; color: #6a6a6a"; padding-top:5px; padding-bottom:5px>{item["note"]}</p>', unsafe_allow_html=True)
                submitted = st.form_submit_button("Training Code", )
                if submitted:
                    streamlit_js_eval(js_expressions=f'window.open("{item["link"]}", "_blank");', key=f"blog_js_eval_{idx}_1")

        st.markdown('<p style="font-size:25px; font-weight:bold;">Mô hình phân biệt các bệnh ung thư</p>', unsafe_allow_html=True)
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
    st.markdown('<p style="font-size:30px; font-weight:bold;">TÀI LIỆU TÌM HIỂU VỀ UNG THƯ DA VÀ CÁC BỆNH KHÁC</p>', unsafe_allow_html=True)
    # Display each item as a form with text and a hyperlink
    for idx, item in enumerate(blogs_data):
        with st.form(key=f"blog_form_{idx}_1"):
            st.markdown(f"### {item['name']}")
            st.markdown(f'<p style="font-size:16px; font-style:italic; color: #6a6a6a">Nguồn: {item["source"]}</p>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Tìm hiểu", )
            if submitted:
                streamlit_js_eval(js_expressions=f'window.open("{item["link"]}", "_blank");', key=f"blog_js_eval_{idx}_1")

def run_prediction():
    st.markdown('<p style="font-size:30px; font-weight:bold;">HỖ TRỢ SÀNG LỌC</p>', unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=['Sử dụng', 'Hướng dẫn sử dụng'],
        icons=['card-image', 'blockquote-left'],
        styles={
            "nav-link": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "400", "font-style": "normal"},
            "nav-link-selected": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "700", "font-style": "bold"}
        },
        orientation='horizontal'
    )

    if selected == "Sử dụng":
        uploaded_file = st.file_uploader("Chọn ảnh để chẩn đoán", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            image = image.resize((300,300))
            st.image(image, caption='Ảnh đã tải lên', use_container_width=True)

        if 'cancer' not in st.session_state:
            st.session_state['cancer'] = False

        if uploaded_file is None:
            st.session_state['cancer'] = False

        # Create both buttons at the same level
        diagnose_button = st.button('Sàng lọc', disabled=uploaded_file is None)

        if diagnose_button:
            if uploaded_file is not None:                
                response = send_request_1(image, MODEL_NAME_1)
                try:
                    final_str, detail_str = get_description(response, 0)
                    st.write(final_str)
                    for i in detail_str:
                        st.write(f"- {i}%")
                    if response['Disease'] not in ['Khong_benh','Benh_khac']:
                        st.session_state['cancer'] = True
                except:
                    st.write(f"Xảy ra lỗi.")

        # Separate if condition for cancer_button
        if st.session_state['cancer']:
            cancer_button = st.button('Chẩn đoán', disabled=uploaded_file is None)
            if cancer_button:
                if uploaded_file is not None:
                    response = send_request_2(image, MODEL_NAME_2) 
                    try:
                        final_str, detail_str = get_description(response, 1)
                        st.write(final_str)
                        for i in detail_str:
                            st.write(f"- {i}%.")
                    except:
                        st.write(f"Xảy ra lỗi.")
    elif selected == "Hướng dẫn sử dụng":
        # Nội dung hướng dẫn
        st.write("""
        Tính năng **Hỗ trợ sàng lọc** giúp bạn phân tích ảnh da để đưa ra dự đoán về nguy cơ mắc các bệnh liên quan đến da, đặc biệt là ung thư da.
        Vui lòng làm theo các bước sau:
        """)
        
        # Các bước sử dụng
        st.markdown("""
        ### **1. Tải ảnh lên**
        - Nhấn vào **"Chọn ảnh để chẩn đoán"** để tải ảnh da cần phân tích từ máy tính.
        - Hỗ trợ các định dạng: `png`, `jpg`, `jpeg`.
        - Sau khi tải ảnh, giao diện sẽ hiển thị hình ảnh bạn vừa chọn để xác nhận.
        """)
        
        st.markdown("""
        ### **2. Nhấn nút Sàng lọc**
        - Sau khi ảnh được tải lên, nhấn vào nút **"Sàng lọc"** để bắt đầu phân tích.
        - Hình ảnh sẽ được gửi lên máy chủ để xử lý và đưa ra dự đoán ban đầu.
        """)
        
        st.markdown("""
        ### **3. Đầu ra dự đoán ban đầu**
        Dự đoán sẽ có định dạng như sau:
        ```plaintext
        Ảnh da của bạn có nguy cơ mắc bệnh ung thư da.
        - Da không bệnh: 0.0%
        - Ung thư da: 94.2%
        - Bệnh da khác: 5.8%
        ```
        - Ví dụ trên cho thấy hình ảnh có khả năng mắc **bệnh ung thư da** với độ tự tin là **94.2%**.
        """)

        st.markdown("""
        ### **4. Hiển thị nút Chẩn đoán (nếu cần)**
        - Nếu dự đoán xác định nguy cơ **ung thư da**, giao diện sẽ hiển thị nút **"Chẩn đoán"**.
        - Nhấn vào nút này để gửi hình ảnh lên máy chủ một lần nữa nhằm thực hiện phân tích chuyên sâu.
        """)

        st.markdown("""
        ### **5. Đầu ra dự đoán chi tiết**
        Sau khi nhấn nút **Chẩn đoán**, kết quả chi tiết sẽ hiển thị như sau:
        ```plaintext
        Ảnh da của bạn có nguy cơ mắc bệnh Ung thư tế bào đáy.
        - Ung thư tế bào đáy: 80.2%.
        - Ung thư tế bào vảy: 19.72%.
        - Ung thư hắc tố: 0.09%.
        ```
        - Dự đoán trên cho biết hình ảnh có khả năng mắc bệnh **ung thư tế bào đáy** với độ tự tin là **80.2%**.
        """)

        st.markdown("""
        ### **Lưu ý quan trọng**
        - **Chất lượng ảnh**: Đảm bảo hình ảnh rõ nét và thể hiện đầy đủ vùng da cần phân tích.
        - **Kết quả dự đoán**: Đây chỉ là công cụ hỗ trợ và không thay thế cho chẩn đoán y khoa. 
        Vui lòng tham khảo ý kiến bác sĩ chuyên môn để được tư vấn cụ thể.
        """)

def run_abouts():
    st.markdown('<p style="font-size:30px; font-weight:bold;">VỀ CHÚNG TÔI</p>', unsafe_allow_html=True)

    for item in abouts_data:        
        col1, col2 = st.columns([1, 3])
        with col1:
            try:
                # Load and display the image
                image = Image.open(item['avatar'])
                image = image.resize((300,300))
                st.image(image, use_container_width =True)
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh '{item['avatar']}': {e}")
        
        with col2:
            # Show the name as a prefilled, non-editable text
            st.title(item['name'])
            st.text(item['detail'])

def run_accounts():
    st.markdown('<p style="font-size:30px; font-weight:bold;">QUYỀN TRUY CẬP</p>', unsafe_allow_html=True)
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
                    st.success("Thêm thành công")
                else:
                    st.error("Đã xảy ra lỗi")
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
                        st.success("Thay đổi thành công")
                    else:
                        st.error("Đã xảy ra lỗi")

def run_page(role="admin"):
    if role.lower() == "user":
        with st.sidebar:
            image = Image.open(os.path.join(SIDEBAR_IMG_PATH))
            st.image(image, use_container_width=True)
            selected = option_menu(
                menu_title=None,
                options=['Trang chủ', 'Dữ liệu và Mô hình', 'Blogs', 'Hỗ trợ sàng lọc, chẩn đoán', 'Về chúng tôi'],
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
                options=['Trang chủ', 'Dữ liệu và Mô hình', 'Blogs', 'Hỗ trợ sàng lọc, chẩn đoán', 'Về chúng tôi', "Tài khoản"],
                icons=['house-fill', 'bar-chart-line-fill', 'file-earmark-richtext-fill', 'aspect-ratio-fill', 'award-fill', "people-fill"],
                styles={
                    "nav-link": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "400", "font-style": "normal"},
                    "nav-link-selected": {"font-size": "18px", "font-family": "'Source Sans Pro', sans-serif", "font-weight": "700", "font-style": "bold"}
                },
                )
    

    if selected == "Trang chủ":
        run_home()
    if selected == "Dữ liệu và Mô hình":
        run_overview()
    if selected == "Blogs":
        run_blogs()
    if selected == "Hỗ trợ sàng lọc, chẩn đoán":
        run_prediction()
    if selected == "Về chúng tôi":
        run_abouts()
    if selected == "Tài khoản":
        run_accounts()

