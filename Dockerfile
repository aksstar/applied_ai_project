FROM python:3.6-alpine
RUN apk add g++ 
RUN pip install numpy
RUN pip install pandas
RUN pip install sklearn
RUN pip install gensim