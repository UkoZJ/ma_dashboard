# %%

from src import app
import panel as pn


# %%

# Update dataset
update_on = False

# if update_on:
# Update the entire dataset (daily operation)
# etl.update()

maEN = app.MosquitoAlertExplorer(name="MA-Dashboard").view

# Select multi-language server or default english
all_lang = False
# Initialize Tornado server as instance or just start it
server_init = "servable"

if all_lang:
    # Start the application(s)
    maCA = app.MosquitoAlertExplorerCA(name="MA-Dashboard").view
    maES = app.MosquitoAlertExplorerES(name="MA-Dashboard").view

    # Servers for different languages
    app_lang = {"English": maEN, "Catalan": maCA, "Spanish": maES}
else:
    # English server only
    app_lang = maEN

# Setup
title = "Mosquito-Alert explorer application"
port = 5006
threaded = False
num_procs = 1  # lunch independent processes on the same port (theaded should be False)
ws = [f"localhost:{port}", "018310f46edb.ngrok.io", "vm148.pub.cloud.ifca.es"]

if server_init == "serve_tornado":
    # Setup Tornado instance
    server = pn.serve(
        app_lang,
        title=title,
        port=port,
        websocket_origin=ws,
        threaded=threaded,
        num_procs=num_procs,
        show=False,
        start=False,
    )
    server.start()
    server.io_loop.start()
elif server_init == "serve":
    pn.serve(
        app_lang,
        title=title,
        port=port,
        websocket_origin=ws,
        threaded=threaded,
        num_procs=num_procs,
        show=True,
        start=True,
        # autoreload=True,
        location=True,
    )
elif server_init == "servable":
    app_lang.servable()
else:
    raise KeyError(server_init)

# %%
