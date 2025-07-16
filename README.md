## Setup for Development


1. Pull and create new container from docker image python:3.9-slim

```
docker run --name portfolioAppDevelopment -it python:3.9-slim  /bin/bash
```

2. Update apt and install sudo and git

```
apt-get update 
apt-get install sudo
sudo apt install git
```

\[Optional\]: Install fish and set as default

```
apt-get install fish
chsh -s /usr/bin/fish
```

2. Install Dependencies

```
apt-get install -y --no-install-recommends python3-dev
```

3. Clone Project

```
git clone https://[username]:[token]@github.com/oscarseah/portfolioApp
cd portfolioApp
```

4. Install Python Dependencies

```
pip install --no-cache-dir -r requirements.txt
```

5. Setup Flask Environments

bash:
```
export FLASK_APP=app FLASK_DEBUG=1
```

fish:
```
set -x FLASK_APP app
set -x FLASK_DEBUG 1
```

6. Start Flask Server

```
flask run --port=[portNumber]
```