# Stardew-Valley-RL-Agent
Artificial Inteligence(AI) agent to playthrough stardew valley.
- Currently implemented .NET 6 RPC protocol over a stardew valley mod.
- Provided API in python for interaction with the environment,
# TO DO?,
Currently the agent itself hasnt been implemented into the environment, allowing a RPC API for reading the game status, however once the agent is wrapped using the RPC protocol, the agent can be then architectured and trained.

To use the StardewModdingAPI over python using the RPC protocol, create a class of it and tell it which method will it use to execute the commands.
simple snippet
    from API import StardewModdingAPI
    stardewapi = StardewModdingAPI(method="tty") # or "ssh+tty" or "docker" 
    response = stardewapi.gamecontenthelper(target="Game1", function="DoesAssetExist", args=["assets/terrainFeatures/grass.png"])

you can see all functions available by:
    print(stardewapi)

also you can see the documentation of each function by:
    print(stardewapi.gamecontenthelper) # for example



FOR DOCKER YOULL NEED TO GIVE IT THE NAME OF THE DOCKER IMAGE
    stardewapi = StardewModdingAPI(method="docker", docker_image_name="menacing racer") #idk man the docker images name are funny

for the rest it works like any other method

Make sure there is an instance of the game running (duh)



