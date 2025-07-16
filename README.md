### Outline
```
███████╗████████╗ █████╗ ██████╗ ██████╗ ███████╗██╗    ██╗    ██╗   ██╗ █████╗ ██╗     ██╗     ███████╗██╗   ██╗    ██████╗ ██╗          █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██║    ██║    ██║   ██║██╔══██╗██║     ██║     ██╔════╝╚██╗ ██╔╝    ██╔══██╗██║         ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
███████╗   ██║   ███████║██████╔╝██║  ██║█████╗  ██║ █╗ ██║    ██║   ██║███████║██║     ██║     █████╗   ╚████╔╝     ██████╔╝██║         ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
╚════██║   ██║   ██╔══██║██╔══██╗██║  ██║██╔══╝  ██║███╗██║    ╚██╗ ██╔╝██╔══██║██║     ██║     ██╔══╝    ╚██╔╝      ██╔══██╗██║         ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
███████║   ██║   ██║  ██║██║  ██║██████╔╝███████╗╚███╔███╔╝     ╚████╔╝ ██║  ██║███████╗███████╗███████╗   ██║       ██║  ██║███████╗    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝ ╚══╝╚══╝       ╚═══╝  ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝   ╚═╝       ╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   
```
This is a project with the focus to train a RL agent for stardew valley.
# MOD
The built mod exposes a reverse proxy through http with RPC capabilities through a set of methods, namely the reflection api available under api/exec <- {json with Target, Method, Parameters}, api/map, for usage refer to a decompiled version of the game, and go through your desired object properties, methods and such.
the reflection api cant really do string to objects, so instead a set of commonly used objects are aliased.
those objects are.
```
player
game1
helper	
data
input	
reflection	
```
the reflection api allows to call, set, get, and invoke anything within those mentioned objects. Therefore a lot of stuff can be done with those objects.
```
getproperty	Get object property value	target, property_name, [required]
setproperty	Set object property value	target, property_name, value, [required]
getfield	Get object field value	target, field_name, [required]
setfield	Set object field value	target, field_name, value, [required]
getmethod	Get method information	target, method_name, [required]
invokemethod	Invoke object method	target, method_name, [parameters...]
```
The exposed api also allows for keybind simulation through api/keyboard/hold <- {json with Key, and DurationMs}
some example of commands to interact with the api using curl are;
```bash
curl -X POST http://localhost:8080/api/execute \
  -H "Content-Type: application/json" \
  -d '{"Target": "reflection", "Method": "getproperty", "Parameters": ["player", "Name"]}'

curl -X POST http://localhost:8080/api/execute \
  -H "Content-Type: application/json" \
  -d '{"Target": "reflection", "Method": "getproperty", "Parameters": ["player", "Money"]}'

curl -X POST http://localhost:8080/api/execute \
  -H "Content-Type: application/json" \
  -d '{"Target": "reflection", "Method": "getproperty", "Parameters": ["player", "Position"]}'

curl -X POST http://localhost:8080/api/execute \
  -H "Content-Type: application/json" \
  -d '{"Target": "reflection", "Method": "getfield", "Parameters": ["player", "health"]}'

curl -X POST http://localhost:8080/api/execute \
  -H "Content-Type: application/json" \
  -d '{"Target": "reflection", "Method": "getfield", "Parameters": ["player", "maxHealth"]}'


curl -X POST http://localhost:8080/api/execute \
  -H "Content-Type: application/json" \
  -d '{"Target": "reflection", "Method": "getproperty", "Parameters": ["game1", "currentSeason"]}'

curl -X POST http://localhost:8080/api/execute \
  -H "Content-Type: application/json" \
  -d '{"Target": "reflection", "Method": "getproperty", "Parameters": ["game1", "timeOfDay"]}'
```
For further reference please read a decompiled version of the game(they're sometimes available through github... AI can easily figure stuff out, just inquire it into anything providing this docs!)
# NOTE
Binary MSGPACKS over base 64 are also given in the callback
```
██████╗ ██████╗  ██████╗    ██████╗ ██████╗  ██████╗ ██╗  ██╗██╗   ██╗
██╔══██╗██╔══██╗██╔════╝    ██╔══██╗██╔══██╗██╔═══██╗╚██╗██╔╝╚██╗ ██╔╝
██████╔╝██████╔╝██║         ██████╔╝██████╔╝██║   ██║ ╚███╔╝  ╚████╔╝ 
██╔══██╗██╔═══╝ ██║         ██╔═══╝ ██╔══██╗██║   ██║ ██╔██╗   ╚██╔╝  
██║  ██║██║     ╚██████╗    ██║     ██║  ██║╚██████╔╝██╔╝ ██╗   ██║   
╚═╝  ╚═╝╚═╝      ╚═════╝    ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   
```
### Installation
Required mods available in the releases tab for this repo.
# Docker
- git clone https://github.com/dotrdp/sv-GUI-less-server
- cd sv-GUI-less-server
- (open the .env.example, configure it, and rename it to .env)
- docker compose up -d
- if steamguard is enabled, run docker exec -it sdvd-server bash ; after that run the steamcmd login by steamcmd +login yourusername , and go through the steamcmd login, afterwards reload the container.
- open the docker desktop app, go into the container, click the 3 dots. Then delete junimoserver mod and upload the unzipped modfolder provided in these paths:
- data/mods, data/StardewValley/Mods.
- reload the container!, the API should automatically detect it.
# TTY
- install the game
- install SMAPI
- upload the unzipped modfolder provided in your StardewValley/Mods dir, it will exist after installing SMAPI, for more info refer to SMAPI installation guide. https://smapi.io
- ready to go!
# SSH + TTY
- set up a basic ssh server on the server machine (sshd for windows, open-ssh for MacOS and Linux, note macos already has it out of the box).
- generate RSA/ed_2519 keys on the machine running the API(you can use your existing keys if you have already generated keys before).
- set up the client(machine running the API) to use a ssh-agent to provide the keys on the ssh connection
- clone your client public key into the server machine directory called .ssh/authorized_keys you may make the authorized_keys dir if it doesnt exist.
- configure your machine hostname in the API ssh_wrap variable.
- Do the installation for TTY on the host machine.
# Python dev env
```
██████╗ ██╗   ██╗████████╗██╗  ██╗ ██████╗ ███╗   ██╗    ████████╗ ██████╗  ██████╗ ██╗     ███████╗
██╔══██╗╚██╗ ██╔╝╚══██╔══╝██║  ██║██╔═══██╗████╗  ██║    ╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██╔════╝
██████╔╝ ╚████╔╝    ██║   ███████║██║   ██║██╔██╗ ██║       ██║   ██║   ██║██║   ██║██║     ███████╗
██╔═══╝   ╚██╔╝     ██║   ██╔══██║██║   ██║██║╚██╗██║       ██║   ██║   ██║██║   ██║██║     ╚════██║
██║        ██║      ██║   ██║  ██║╚██████╔╝██║ ╚████║       ██║   ╚██████╔╝╚██████╔╝███████╗███████║
╚═╝        ╚═╝      ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚══════╝
```
An api for interaction with the http proxy is provided, along with an environment class, player class, map class, allowing for a seamless integration with python. You instantiate the API telling it where and how to communicate with the http proxy.
currently TTY(http proxy in local host), ssh+tty(http proxy in a ssh server, note you must have keys or use sshpass in the API ssh_wrap declaration), and docker(http proxy inside a docker container). For the docker container we forked a popular stardew valley server solution. Note it also provides a x11 VNC server 8090 to serve a display with nginx.
Note, binary messagepacks over base 64 are also returned on the api.

git clone https://github.com/dotrdp/Stardew-Valley-RL-Agent/
cd Stardew-Valley-RL-Agent
git checkout master
you're free to go!

#Extras
```
██████╗ ███████╗██████╗ ██╗   ██╗ ██████╗  ██████╗ ██╗███╗   ██╗ ██████╗     ████████╗ ██████╗  ██████╗ ██╗     ███████╗
██╔══██╗██╔════╝██╔══██╗██║   ██║██╔════╝ ██╔════╝ ██║████╗  ██║██╔════╝     ╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██╔════╝
██║  ██║█████╗  ██████╔╝██║   ██║██║  ███╗██║  ███╗██║██╔██╗ ██║██║  ███╗       ██║   ██║   ██║██║   ██║██║     ███████╗
██║  ██║██╔══╝  ██╔══██╗██║   ██║██║   ██║██║   ██║██║██║╚██╗██║██║   ██║       ██║   ██║   ██║██║   ██║██║     ╚════██║
██████╔╝███████╗██████╔╝╚██████╔╝╚██████╔╝╚██████╔╝██║██║ ╚████║╚██████╔╝       ██║   ╚██████╔╝╚██████╔╝███████╗███████║
╚═════╝ ╚══════╝╚═════╝  ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝        ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚══════╝
```
A python cli, along with manual controls over TUI are found in this repo.
<img width="1704" height="1594" alt="image" src="https://github.com/user-attachments/assets/31b1ade9-55e5-4af2-bfe2-4426740d14db" />
python cli tool, cli.py
and Stardew Valley TUI with vim motions for tp, manual_controls.py
<img width="1064" height="1712" alt="image" src="https://github.com/user-attachments/assets/a5d3ee05-4df0-41a3-9809-3a25c77015f5" />
#credits and acknowledgements
https://github.com/Pathoschild/SMAPI
https://www.junimohost.com




