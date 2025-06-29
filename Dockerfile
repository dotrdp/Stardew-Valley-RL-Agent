FROM ubuntu:22.04
RUN apt update 
RUN apt install lib32gcc-s1 -y
RUN apt install curl -y
RUN apt install lib32stdc++6 -y
RUN apt install lib32tinfo6 -y
RUN mkdir ~/Steam && cd ~/Steam
RUN curl -sqL "https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz" | tar zxvf -
RUN ./steamcmd.sh 
