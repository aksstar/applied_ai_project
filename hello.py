import streamlit as st
import pickle
import pandas as pd
loaded_model = pickle.load(open('rf_model.sav', 'rb'))
scaler = pickle.load(open('scaler_model.sav', 'rb'))
pca = pickle.load(open('pca_model.sav', 'rb'))
doc2vec_model = pickle.load(open('doc2vec_model.sav', 'rb'))


column_names = ['city', 'bd', 'registered_via', 'registration_year',
       'registration_month', 'song_length', 'genre_ids', 'language', 'album',
       'artist', 'listen-with']

input_dict = {}

with st.form("my_form"):
   st.write("Inside the form")
   slider_val = st.slider("Form slider")
   checkbox_val = st.checkbox("Form checkbox")
   
   city = st.number_input('city', 1)
   bd = st.number_input('bd', 23)
   registered_via = st.number_input('registered_via', 7)
   registration_year = st.number_input('registration_year', 2011)
   registration_month = st.number_input('registration_month', 10)

   song_length = st.number_input('song_length', 2000)
   genre_ids = st.number_input('genre_ids', 458)
   language = st.number_input('language', 52)
   album = st.number_input('album', 0)
   artist = st.number_input('artist', 1)
   
   source_type = st.selectbox('source_type ?',('online-playlist', 'local-playlist', 'local-library',
   'top-hits-for-artist', 'album', 'unknown', 'song-based-playlist',
   'radio', 'song', 'listen-with', 'artist', 'topic-article-playlist','my-daily-playlist'))
   
   st.write('You selected:', source_type)

   source_system_tab = st.selectbox('source_system_tab ?',('explore', 'my library', 'search', 'discover', 'unknown', 'radio','listen with', 'notification', 'settings'))
   
   st.write('You selected:', source_system_tab)
   name_of_artist_song = st.text_input('Song_Name', 'SOng 1 ')

   input_val = [city, bd, registered_via, registration_year, registration_month, song_length, genre_ids, language
   , album, artist]

   source_type_array = ['listen-with', 'local-library', 'local-playlist', 'my-daily-playlist', 'online-playlist', 'radio', 'song', 'song-based-playlist', 'top-hits-for-artist', 'topic-article-playlist', 'unknown']

   source_system_tab_array = ['discover', 'explore', 'listen with', 'my library', 'notification', 'radio', 'search', 'settings','unknown']

   source_type_array_input_list = []
   source_system_tab_array_input_list = []

   for col in source_type_array:
       if col == source_type:
              source_type_array_input_list.append(1)
       else:
              source_type_array_input_list.append(0)

   for col in source_system_tab_array:
       if col == source_system_tab:
              source_system_tab_array_input_list.append(1)
       else: 
              source_system_tab_array_input_list.append(0)


   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       print(input_val, len(input_val))
       print(source_type_array_input_list, len(source_type_array_input_list))
       input_val.extend(source_type_array_input_list)
       print(source_system_tab_array_input_list, len(source_system_tab_array_input_list))
       input_val.extend(source_system_tab_array_input_list)
       print(input_val, len(input_val))
       embeddings = doc2vec_model.infer_vector(name_of_artist_song.split())
       print(embeddings)
       input_val.extend(embeddings)
       print(input_val, len(input_val))
       X_train_trans = scaler.transform([input_val])
       X_train_pca = pca.transform(X_train_trans)
       y_hat = loaded_model.predict(X_train_pca)
       print(y_hat[0])
       
       st.write("slider", slider_val, "checkbox", checkbox_val, "output is : ", y_hat[0])

st.write("Outside the form")