# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:27:48 2022

@author: user
"""

import kivy
kivy.require('1.0.7')

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget
import sqlite3
from kivy.lang import Builder
from kivy.uix.popup import Popup
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import string
from sklearn.naive_bayes import MultinomialNB
import pickle

train=pd.read_csv('training.csv', ",")
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
file='vocab2.pkl'
transformer=pickle.load(open(file, 'rb'))

def learning(data): 
    bow=transformer.transform(data)
    trans_tfidf=TfidfTransformer().fit(bow)
    tfidf=trans_tfidf.transform(bow)
    return tfidf 

def pre_processing(data):
    punct= [i for i in data if i not in string.punctuation]
    punct=''.join(punct)
    return [i for i in punct.split() if i.lower() not in stopwords.words('english')]

text_label="init2"
def Art_Int(x):
        x={'text':[x]}
        x=pd.DataFrame(x)
        pred=loaded_model.predict(learning(x['text']))
        print(pred)
        global t
        if pred==0:
            t='sad'
        elif pred==1:
            t='joy'
        elif pred==2:
            t='love'
        elif pred==3:
            t='anger'
        else:
            t='fear'
        return t

        
        
class ScreenManager(ScreenManager):
    pass

class Home(Screen):
    feeling=ObjectProperty(None)
    def on_click(self):
        self.t=Art_Int(self.feeling.text)
        self.feeling.text=""

        if (t != None):
            popup = Popup(title='Are you feeling...', content=Label(text=self.t),
              auto_dismiss=False)
            popup.open()
        

class MyMood(App):
    def build(self):
        return Builder.load_file('mymood.kv')
    
MyMood().run()