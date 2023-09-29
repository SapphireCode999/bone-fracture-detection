import ultralytics
import streamlit as st
import tempfile
import cv2
import PIL
from ultralytics import YOLO


model_path = 'model_weight/best.pt'


# Setting page layout
st.set_page_config(
    page_title="Bone Fracture Detection",  # Setting page title
    page_icon="",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
)


#  Creating main page heading
st.sidebar.title('Custom Object Detection')
st.title("Bone Fracture Detection with YOLOv8 and Streamlit")


st.markdown('''
            :orange[**Upload an xray to detect fracture**]
            ''')


st.markdown('''
            :orange[**Click :blue[Detect] button and check the result.**]
            '''
) 



# Creating two columns on the main page
col1, col2 = st.columns(2)

col3, col4, col5 = st.columns(3)


st.divider()

with col1:
     uploaded_file = st.file_uploader("Choose a file")
     


# Adding image to the first column if image is uploaded

with col3:
    if uploaded_file:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(uploaded_file)
        # Adding the uploaded image to the page with a caption
        st.image(uploaded_file,
                 caption="Uploaded Image",
                 use_column_width=True
                 )
        

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

        

if st.sidebar.button('Detect'):
    results = model.predict(uploaded_image)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
                    
            class_id = box.cls[0].item()
            object_cls = result.names[class_id]
            prob = round(box.conf[0].item(), 2)
        
            conf = f"{prob:.0%}"
        
        res_plotted = result[0].plot()
    
    with col4:
        
        try:
            st.image(res_plotted,
                 caption='Detected Image',
                 use_column_width=True,
                 )
    
        except Exception as ex:
            st.write("No image is uploaded yet!")
    
    
    with col5:       
            st.markdown('''
                        :orange[**Detection:**]
                        ''')
            st.header(object_cls)
            # st.divider()
            st.markdown('''
                        :orange[**Confidence:**]
                        '''
            ) 
            st.subheader(conf)