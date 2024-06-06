import streamlit as st
from important_variables import logo_image
import numpy as np
import datetime

import re

import matplotlib.pyplot as plt
import pickle

import base64

from utils import find_most_similar, get_image_base64, fig_to_image, load_data_for_testing, find_all
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from matplotlib import pyplot as plt

def extract_job_titles(text):
    # matches any sequence of words that starts with a capitalized word and ends just before 'Key Responsibilities'
    job_titles = re.findall(r'([A-Z][A-Za-z]*(?:\s[A-Z][A-Za-z]*)*)\s(?=Key Responsibilities)', text)
    return ', '.join(job_titles)

def month_to_year(months):
    years = months // 12
    month = months % 12
    if month != 0:
        return f'{years} Years, {month} Months'
    else:
        return f'{years} Years'

def logging(error):
    with open('loggs/logging.txt', 'a+') as log_file:
        log_file.write(datetime.datetime.now() + ':' + error)

with open('models/label_encoder', 'rb') as file:
    le = pickle.load(file)

def main():
    # data = generate_data()
    st.sidebar.image(logo_image)

    st.sidebar.title("Control Panel")

    page_section = st.sidebar.radio("Sections: ", ('Input Requirement', 'Matching Profiles', 'Display Acceptance', 'Model Performance'))

    if (page_section == 'Input Requirement'):
        # st.title(f"CURRENTLY IN DATABASE")
        # best_profiles = find_all()
        # fa_icons = """
        #     <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        # """
        # st.markdown(fa_icons, unsafe_allow_html=True)
        # st.markdown(
        # f"""
        # <div style="display: flex; flex-direction: row;">
        #     <div class = "card" ">
        #         <i class="fas fa-users"></i>
        #         <h3>Total Candidates</h3>
        #         <p>{best_profiles.shape[0]}</p>
        #     </div>
        #     <div class = "card" ">
        #         <i class="fas fa-dollar-sign"></i>
        #         <h3>Median Salary</h3>
        #         <p>₹ {best_profiles['Current Salary'].median()}</p>
        #     </div>
        #     <div class = "card"  ">
        #         <i class="fas fa-calendar-alt"></i>
        #         <h3>Mean Experience</h3>
        #         <p>{int(best_profiles['Years'].mean())} Months</p>
        #     </div>
        # </div>
        # """,
        # unsafe_allow_html=True)
        # del best_profiles
        st.title(f"ENTER JOB DETAILS HERE")
        # Define the select box options
        experience_options = ['0-1 years', '2-5 years', '6-10 years', '>10 years']

        education_options = ['Undergraduate', 'Postgraduate', 'PhD']

        form_data = st.form(key='job_form')
        gathered_form_data = {}

        with form_data:
            cols = st.columns([3, 1])
            gathered_form_data['Company Name'] = cols[0].text_input('Company Name')
            gathered_form_data['Role'] = cols[0].text_input('Role')
            gathered_form_data['Salary'] = cols[0].number_input('Salary')
            gathered_form_data['Required Experience'] = cols[0].selectbox('Required Experience', experience_options)
            gathered_form_data['Location'] = cols[0].selectbox('Location', options = list(set([item for item in le.classes_ if type(item) != float])))
            gathered_form_data['Key Skills'] = cols[0].text_input('Key Skills (Comma sepparated)').split(',')
            gathered_form_data['Educational Qualification'] = cols[0].selectbox('Educational Qualification', education_options)
            gathered_form_data['Job Description'] = cols[1].text_area('Job Description', height=400)

            # Submit button in the form
            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            # st.write(st.session_state)
            st.session_state['Company Name'] = gathered_form_data['Company Name']
            st.session_state['Role'] = gathered_form_data['Role']
            st.session_state['Salary'] = gathered_form_data['Salary']
            st.session_state['Required Experience'] = gathered_form_data['Required Experience']
            st.session_state['Location'] = gathered_form_data['Location']
            st.session_state['Key Skills'] = gathered_form_data['Key Skills']
            st.session_state['Educational Qualification'] = gathered_form_data['Educational Qualification']
            st.session_state['Job Description'] = gathered_form_data['Job Description']
            st.success('Job requirement updated successfully, Navigate to "Matching Profile Section"')

            # Print the output in the terminal
    elif (page_section == 'Matching Profiles'):
        st.title(f"MATCHED PROFILES")
        if 'Role' in st.session_state:
            best_profiles = find_most_similar([st.session_state["Role"], " ".join(st.session_state["Key Skills"]), st.session_state["Job Description"]])
            threshold = st.number_input("Set threshold of similarity: ", value = best_profiles['Cosine sim'].median())
            best_profiles = best_profiles[best_profiles['Cosine sim'] > threshold]

            if best_profiles.shape[0] != 0:
                fa_icons = """
                    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
                """
                
                location_counts = best_profiles['Current Location'].value_counts()
                locations = location_counts.index.tolist()
                counts = location_counts.values.tolist()
                st.markdown(fa_icons, unsafe_allow_html=True)
                st.markdown(
                f"""
                <div style="display: flex; flex-direction: row;">
                    <div class = "card" ">
                        <i class="fas fa-users"></i>
                        <h3>Total Candidates</h3>
                        <p>{best_profiles.shape[0]}</p>
                    </div>
                    <div class = "card" ">
                        <i class="fas fa-dollar-sign"></i>
                        <h3>Median Salary</h3>
                        <p>₹ {best_profiles['Current Salary'].median()}</p>
                    </div>
                    <div class = "card"  ">
                        <i class="fas fa-calendar-alt"></i>
                        <h3>Mean Experience</h3>
                        <p>{int(best_profiles['Years'].mean())} Months</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True)

                # Second Card with Image
                fig, ax = plt.subplots()
                ax.pie(counts, labels = locations)

                # Set the plot background color to transparent
                fig.patch.set_alpha(0.0)
                
                # Convert Matplotlib figure to an image
                image = fig_to_image(fig)

                _, image_col, _ = st.columns([1, 5, 1])

                # Display the image in the card
                image_col.markdown(
                    """
                    <div class="card">
                        <div class="card-Image">
                            <div class="card-title">
                                <h3>Popular Locations</h3>
                            </div>
                            <div style="display: flex; justify-content: center;">
                                <img src="data:image/png;base64,{}" alt="Plot Image">
                            </div>
                        </div>
                    </div>
                    """.format(image),
                    unsafe_allow_html=True
                    )
                st.dataframe(best_profiles[['Name', 'Skills', 'Notice Period', 'Expected CTC',
                    'Current Salary', 'Current Location', 'Cosine sim']].reset_index(drop = True), use_container_width = True)
            else:
                st.error('No matching profiles found !!! try changing the requirement or change the similarity threshold.')
            
            st.session_state['best_profiles'] = best_profiles
        else:
            st.warning('You need to put the details of the requirement in the job description form!!')
    elif page_section == 'Display Acceptance':
        st.title(f"ACCEPTANCE")
        if st.session_state['best_profiles'].shape[0] != 0:
            if 'best_profiles' in st.session_state:
                final_data = st.session_state['best_profiles'][['Notice Period', 'Expected CTC', 'Offered Location', 'Offered Salary',
                    'Current Salary', 'Current Location']].copy().iloc[:10, :]
                if (st.session_state['Salary'] < final_data['Current Salary']).sum() >= 1:
                    st.warning('Because of rare search some suggestions might be abnormal!!!')

                final_data['Offered Location'] = st.session_state['Location']
                final_data['Offered Salary'] = st.session_state['Salary']

                try:
                    final_data['Current Location'] = le.transform(final_data['Current Location'])
                    final_data['Offered Location'] = le.transform(final_data['Offered Location'])
                except Exception as e:
                    logging(e)
                    st.write(e)
                
                with open('models/Logistic_Regression', 'rb') as file:
                    model = pickle.load(file)

                pred = model.predict(final_data)
                pred_proba = model.predict_proba(final_data)

                image_col, detail_col = st.columns([10, 1])

                image_base64 = get_image_base64("user_profile.png")
                for random_person in final_data.index:
                    # Display the image in the card
                    image_col.markdown(
                        f"""
                        <div class = "profile">
                            <div class="card_profile_photo">
                                <div class="card-Image">
                                    <div style="display: flex;">
                                        <img src="data:image/png;base64,{image_base64}" class="card-image" alt="User Photo">
                                    </div>
                                </div>
                            </div>
                            <div class="card_profile_info">
                                <div class="card-Info">
                                    <div class="card-title_info">
                                        <h5>{st.session_state["best_profiles"].loc[random_person, "Name"].title()}</h5>
                                        <h6>{round(pred_proba[st.session_state["best_profiles"].index.get_loc(random_person)][pred[st.session_state["best_profiles"].index.get_loc(random_person)]]*100, 1)}% Probability</h6>
                                        <h6>{st.session_state["best_profiles"].loc[random_person, "Current Role"]}</h5>
                                        <h6>Current CTC: {int(st.session_state["best_profiles"].loc[random_person, "Current Salary"])} INR</h5>
                                        <h6>Notice Period: {st.session_state["best_profiles"].loc[random_person, "Notice Period"]} Days</h5>
                                        <h6>Experience: {month_to_year(st.session_state["best_profiles"].loc[random_person, "Years"])}</h5>
                                        <h6>Current Location: {st.session_state["best_profiles"].loc[random_person, "Current Location"]}</h5>
                                        <h6>Skills: {st.session_state["best_profiles"].loc[random_person, "Skills"]}</h5>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,)
        else:
            st.warning('No profiles to show!!')
    elif page_section == 'Model Performance':
        st.title(f"PERFORMANCE OF THE CURRENT MODEL")
        test_data = load_data_for_testing()[['Notice Period', 'Expected CTC', 'Offered Location', 'Offered Salary',
                'Current Salary', 'Current Location', 'label']]

        try:
            test_data['Current Location'] = le.transform(test_data['Current Location'])
            test_data['Offered Location'] = le.transform(test_data['Offered Location'])
        except Exception as e:
            logging(e)
            st.write(e)
        
        model_name = 'Logistic_Regression'
        st.markdown(f'### Used Model is {model_name.replace("_", " ")}')

        with open(f'models/{model_name}', 'rb') as file:
                model = pickle.load(file)

        pred = model.predict(test_data[['Notice Period', 'Expected CTC', 'Offered Location', 'Offered Salary',
                'Current Salary', 'Current Location']])
        
        st.markdown(f'#### Current accuracy of the model: {round(100 * accuracy_score(test_data["label"], pred), 2)} %')

        if 100 * accuracy_score(test_data["label"], pred) < 80:
            st.warning('The model performance is below the accepted standard, please retrain the model!!!')
        
        st.markdown('### Classification Report')
        st.text('Model Report:\n    ' + classification_report(test_data['label'], pred))  # Print the classification report
        cm = confusion_matrix(test_data['label'], pred)  # Get the confusion matrix
        
        # Plot the confusion matrix
        fig = plt.figure(figsize=(10, 7))  # Set the figure size
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")  # Create a heatmap
        plt.title(f"Confusion Matrix")  # Set the title
        plt.xlabel("Predicted")  # Set the x-axis label
        plt.ylabel("Actual")  # Set the y-axis label

        st.markdown('')
        st.markdown('### Confusion Matrix')
        fig_col, _ = st.columns([4, 4])
        fig_col.pyplot(fig)  # Display the plot

def homepage():
    home_image = st.image(logo_image)

    c1, c2, c3 = st.columns([2,1,2])
    c2.markdown('')
    c2.markdown('')
    continue_forward = c2.button('Continue >>>')

    st.session_state['home_page'] = False
    

    if continue_forward:
        print('going to the application!!')
        home_image.empty()
        main()


