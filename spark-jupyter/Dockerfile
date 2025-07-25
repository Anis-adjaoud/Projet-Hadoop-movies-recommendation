FROM jupyter/pyspark-notebook:latest

# ---- ENV ----
ENV HADOOP_VERSION=3.3.6
ENV HADOOP_HOME=/opt/hadoop
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

# ---- ROOT INSTALL ----
USER root

RUN apt-get update && \
    apt-get install -y openjdk-11-jdk wget ssh && \
    apt-get clean

# ---- HADOOP INSTALL ----
RUN wget https://dlcdn.apache.org/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
    tar -xzf hadoop-${HADOOP_VERSION}.tar.gz -C /opt/ && \
    mv /opt/hadoop-${HADOOP_VERSION} $HADOOP_HOME && \
    rm hadoop-${HADOOP_VERSION}.tar.gz

# ---- HADOOP CONFIG ----
COPY hadoop-config/* $HADOOP_HOME/etc/hadoop/

# ---- HDFS DIRECTORIES ----
RUN mkdir -p /tmp/hadoop-jovyan/namenode && \
    mkdir -p /tmp/hadoop-jovyan/datanode && \
    chown -R jovyan:users /tmp/hadoop-jovyan

# ---- USER SETUP ----
USER $NB_UID

# ---- LIBS FOR NOTEBOOKS ----
RUN pip install matplotlib seaborn
